"""Dataset loading + augmentation pipelines for the Pet Emotion Classifier.

Source dataset: Kaggle "Pets Facial Expression Recognition" by anshtanwar.
We expect the data laid out as:

    data/raw/
        Angry/   *.jpg
        Sad/     *.jpg
        happy/   *.jpg
        Other/   *.jpg

We collapse the 4 classes into a binary task:
    happy   -> "happy"
    Angry, Sad -> "unhappy"
    Other -> dropped (ambiguous / surprised faces)

Binary framing makes the augmentation story crisper for the 5-min pitch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

CLASS_NAMES = ["unhappy", "happy"]  # index 0, 1


# ---------- Augmentation pipelines ----------
#
# Each augmentation maps to a real-world variation in user-uploaded pet photos.
# We document the WHY inline so the pitch can reference it.

def _eval_transform() -> Callable:
    """Plain center-crop preprocessing (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _baseline_train_transform() -> Callable:
    """Bare-minimum training transform: resize + flip only.

    This is the "no thoughtful augmentation" control we compare against.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _augmented_train_transform() -> Callable:
    """Augmentations tied to real-world pet-photo variation.

    Each transform corresponds to a failure mode the model must survive:
      RandomResizedCrop  -> pet close-ups vs. full-body shots, off-center framing
      RandomHorizontalFlip -> mirroring (cheap free invariance)
      RandomRotation     -> users tilting phones
      RandomPerspective  -> non-frontal camera angles
      ColorJitter        -> indoor warm light vs. outdoor sun vs. dim rooms
      GaussianBlur       -> motion blur / out-of-focus / low-end phone cameras
      RandomErasing      -> partial occlusion (collar, hand, food bowl, toy)
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_transforms(mode: str) -> Callable:
    """Return a transform pipeline.

    mode: "baseline" (minimal aug) | "augmented" (full aug) | "eval" (no aug)
    """
    if mode == "baseline":
        return _baseline_train_transform()
    if mode == "augmented":
        return _augmented_train_transform()
    if mode == "eval":
        return _eval_transform()
    raise ValueError(f"unknown transform mode: {mode}")


# ---------- Dataset ----------

# Folder names in the Kaggle release -> our binary label.
# Keys are lower-cased before lookup so capitalization quirks don't matter.
_CLASS_FOLDER_MAP = {
    "happy": 1,
    "sad": 0,
    "angry": 0,
}


@dataclass
class PetEmotionDataset(Dataset):
    """Loads images from a directory of class-named subfolders."""

    root: Path
    transform: Callable | None = None

    def __post_init__(self):
        self.root = Path(self.root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root {self.root} not found. "
                "See README for Kaggle download instructions."
            )
        self.samples: list[tuple[Path, int]] = []
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = _CLASS_FOLDER_MAP.get(class_dir.name.lower())
            if label is None:
                continue  # skip "Other" and any unexpected folders
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label))
        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. "
                f"Expected subfolders named: {list(_CLASS_FOLDER_MAP)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_loaders(
    root: str | Path,
    batch_size: int = 32,
    train_mode: str = "augmented",
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders.

    Train uses the requested augmentation mode; val and test use eval transform.
    Splits are deterministic given `seed` so the baseline vs. augmented run
    sees the same val/test images.
    """
    base = PetEmotionDataset(root=Path(root), transform=None)
    n = len(base)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        base, [n_train, n_val, n_test], generator=gen
    )

    # Subsets share the underlying dataset, so we wrap them to apply
    # mode-specific transforms without mutating each other.
    train_ds = _TransformedSubset(train_subset, get_transforms(train_mode))
    val_ds = _TransformedSubset(val_subset, get_transforms("eval"))
    test_ds = _TransformedSubset(test_subset, get_transforms("eval"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


class _TransformedSubset(Dataset):
    """Wraps a Subset and applies a transform at __getitem__ time."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Underlying dataset returns (PIL image, label) because we passed
        # transform=None when constructing it.
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
