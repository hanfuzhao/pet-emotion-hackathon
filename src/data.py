from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

CLASS_NAMES = ["unhappy", "happy"]


def _eval_tx():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _baseline_tx():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _augmented_tx():
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


def get_transforms(mode):
    if mode == "baseline":
        return _baseline_tx()
    if mode == "augmented":
        return _augmented_tx()
    if mode == "eval":
        return _eval_tx()
    raise ValueError(f"unknown mode: {mode}")


_LABELS = {
    "happy": 1,
    "sad": 0,
    "angry": 0,
}


@dataclass
class PetEmotionDataset(Dataset):
    root: Path
    transform: object = None

    def __post_init__(self):
        self.root = Path(self.root)
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root {self.root} not found")
        self.samples = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            label = _LABELS.get(d.name.lower())
            if label is None:
                continue
            for f in d.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((f, label))
        if not self.samples:
            raise RuntimeError(f"no images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class _Wrapped(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, label = self.subset[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_loaders(root, batch_size=32, train_mode="augmented",
                  val_split=0.15, test_split=0.15, seed=42, num_workers=2):
    base = PetEmotionDataset(root=Path(root), transform=None)
    n = len(base)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(base, [n_train, n_val, n_test], generator=gen)

    train_ds = _Wrapped(train_sub, get_transforms(train_mode))
    val_ds = _Wrapped(val_sub, get_transforms("eval"))
    test_ds = _Wrapped(test_sub, get_transforms("eval"))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )
