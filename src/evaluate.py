"""Robustness evaluation: how well do baseline vs. augmented models hold up
under realistic image corruptions?

We define a handful of corruption families (brightness, blur, rotation, jpeg,
occlusion) at multiple severities. We then evaluate every checkpoint against
each corruption x severity and report accuracy.

This is the *headline number* for the pitch: "augmented model holds 80% under
heavy blur, baseline drops to 55%".
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

from .data import CLASS_NAMES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, PetEmotionDataset, build_loaders
from .model import build_model


# Severities 0..4 (0 = clean). Tuned so severity-4 is visibly degraded but
# still humanly recognisable.

def corrupt_brightness(img: Image.Image, sev: int) -> Image.Image:
    factor = [1.0, 1.4, 1.8, 2.2, 2.6][sev]
    arr = np.asarray(img).astype(np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def corrupt_darkness(img: Image.Image, sev: int) -> Image.Image:
    factor = [1.0, 0.7, 0.5, 0.35, 0.25][sev]
    arr = np.asarray(img).astype(np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def corrupt_blur(img: Image.Image, sev: int) -> Image.Image:
    radius = [0, 1.5, 3.0, 5.0, 7.0][sev]
    if radius == 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def corrupt_rotation(img: Image.Image, sev: int) -> Image.Image:
    deg = [0, 10, 20, 30, 45][sev]
    if deg == 0:
        return img
    return img.rotate(deg, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def corrupt_jpeg(img: Image.Image, sev: int) -> Image.Image:
    quality = [100, 50, 30, 15, 8][sev]
    if quality >= 95:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def corrupt_occlusion(img: Image.Image, sev: int) -> Image.Image:
    """Drop a random gray rectangle on the image (simulates a hand / collar)."""
    if sev == 0:
        return img
    frac = [0, 0.10, 0.18, 0.28, 0.40][sev]
    w, h = img.size
    bw, bh = int(w * frac), int(h * frac)
    # Seeded RNG so a single image looks the same across model runs.
    rng = np.random.default_rng(seed=hash(img.tobytes()) % (2**32))
    x = rng.integers(0, max(1, w - bw))
    y = rng.integers(0, max(1, h - bh))
    out = img.copy()
    out.paste((128, 128, 128), (int(x), int(y), int(x + bw), int(y + bh)))
    return out


CORRUPTIONS: dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "brightness": corrupt_brightness,
    "darkness": corrupt_darkness,
    "blur": corrupt_blur,
    "rotation": corrupt_rotation,
    "jpeg": corrupt_jpeg,
    "occlusion": corrupt_occlusion,
}


_eval_tx = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_checkpoint(ckpt_path: str | Path, device: str | None = None) -> torch.nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def accuracy_under_corruption(
    model: torch.nn.Module,
    dataset: PetEmotionDataset,
    corruption: str,
    severity: int,
    device: str | None = None,
    batch_size: int = 32,
) -> float:
    """Apply corruption -> preprocess -> forward, return accuracy."""
    device = device or next(model.parameters()).device
    fn = CORRUPTIONS[corruption]
    correct, total = 0, 0
    batch_imgs, batch_labels = [], []

    def flush():
        nonlocal correct, total
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        y = torch.tensor(batch_labels).to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
        batch_imgs.clear()
        batch_labels.clear()

    for path, label in dataset.samples:
        img = Image.open(path).convert("RGB")
        img = fn(img, severity)
        batch_imgs.append(_eval_tx(img))
        batch_labels.append(label)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return correct / max(total, 1)


def robustness_matrix(
    model: torch.nn.Module,
    dataset: PetEmotionDataset,
    severities: tuple[int, ...] = (0, 1, 2, 3, 4),
    device: str | None = None,
) -> dict[str, dict[int, float]]:
    """Compute accuracy for every (corruption, severity) pair."""
    out: dict[str, dict[int, float]] = {}
    for name in CORRUPTIONS:
        out[name] = {}
        for sev in severities:
            acc = accuracy_under_corruption(model, dataset, name, sev, device=device)
            out[name][sev] = acc
            print(f"  {name:12s} sev={sev}  acc={acc:.3f}")
    return out


def compare_models(
    baseline_ckpt: str | Path,
    augmented_ckpt: str | Path,
    data_dir: str | Path,
    seed: int = 42,
) -> dict:
    """Build the held-out test set (same seed as training) and score both models.

    Returns a dict suitable for plotting / dumping to JSON.
    """
    # Re-use build_loaders just to recover the same test split.
    _, _, test_loader = build_loaders(root=data_dir, seed=seed)
    test_subset = test_loader.dataset  # _TransformedSubset
    # Pull the raw PetEmotionDataset.samples for the indices in this split.
    raw = test_subset.subset.dataset
    indices = test_subset.subset.indices
    held_out = PetEmotionDataset.__new__(PetEmotionDataset)
    held_out.root = raw.root
    held_out.transform = None
    held_out.samples = [raw.samples[i] for i in indices]

    print(f"[robustness] held-out test set size: {len(held_out)}")
    print("\n[baseline]")
    base_model = load_checkpoint(baseline_ckpt)
    base_results = robustness_matrix(base_model, held_out)

    print("\n[augmented]")
    aug_model = load_checkpoint(augmented_ckpt)
    aug_results = robustness_matrix(aug_model, held_out)

    return {"baseline": base_results, "augmented": aug_results}
