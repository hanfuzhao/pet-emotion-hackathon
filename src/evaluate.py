import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from .data import CLASS_NAMES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, PetEmotionDataset, build_loaders
from .model import build_model


def corrupt_brightness(img, sev):
    f = [1.0, 1.4, 1.8, 2.2, 2.6][sev]
    a = np.asarray(img).astype(np.float32) * f
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def corrupt_darkness(img, sev):
    f = [1.0, 0.7, 0.5, 0.35, 0.25][sev]
    a = np.asarray(img).astype(np.float32) * f
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def corrupt_blur(img, sev):
    r = [0, 1.5, 3.0, 5.0, 7.0][sev]
    if r == 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def corrupt_rotation(img, sev):
    d = [0, 10, 20, 30, 45][sev]
    if d == 0:
        return img
    return img.rotate(d, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def corrupt_jpeg(img, sev):
    q = [100, 50, 30, 15, 8][sev]
    if q >= 95:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def corrupt_occlusion(img, sev):
    if sev == 0:
        return img
    frac = [0, 0.10, 0.18, 0.28, 0.40][sev]
    w, h = img.size
    bw, bh = int(w * frac), int(h * frac)
    rng = np.random.default_rng(seed=hash(img.tobytes()) % (2**32))
    x = rng.integers(0, max(1, w - bw))
    y = rng.integers(0, max(1, h - bh))
    out = img.copy()
    out.paste((128, 128, 128), (int(x), int(y), int(x + bw), int(y + bh)))
    return out


CORRUPTIONS = {
    "brightness": corrupt_brightness,
    "darkness": corrupt_darkness,
    "blur": corrupt_blur,
    "rotation": corrupt_rotation,
    "jpeg": corrupt_jpeg,
    "occlusion": corrupt_occlusion,
}


_tx = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(ckpt_path, device=None):
    device = device or _pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = build_model(num_classes=len(ckpt["class_names"]))
    m.load_state_dict(ckpt["state_dict"])
    m.to(device).eval()
    return m


@torch.no_grad()
def accuracy_under_corruption(model, dataset, corruption, severity, device=None, batch_size=32):
    device = device or next(model.parameters()).device
    fn = CORRUPTIONS[corruption]
    correct, total = 0, 0
    bx, by = [], []

    def flush():
        nonlocal correct, total
        if not bx:
            return
        x = torch.stack(bx).to(device)
        y = torch.tensor(by).to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
        bx.clear()
        by.clear()

    for path, label in dataset.samples:
        img = Image.open(path).convert("RGB")
        bx.append(_tx(fn(img, severity)))
        by.append(label)
        if len(bx) >= batch_size:
            flush()
    flush()
    return correct / max(total, 1)


def robustness_matrix(model, dataset, severities=(0, 1, 2, 3, 4), device=None):
    out = {}
    for name in CORRUPTIONS:
        out[name] = {}
        for sev in severities:
            acc = accuracy_under_corruption(model, dataset, name, sev, device=device)
            out[name][sev] = acc
            print(f"  {name:12s} sev={sev}  acc={acc:.3f}")
    return out


def compare_models(baseline_ckpt, augmented_ckpt, data_dir, seed=42):
    _, _, test_loader = build_loaders(root=data_dir, seed=seed)
    test_subset = test_loader.dataset
    raw = test_subset.subset.dataset
    indices = test_subset.subset.indices
    held = PetEmotionDataset.__new__(PetEmotionDataset)
    held.root = raw.root
    held.transform = None
    held.samples = [raw.samples[i] for i in indices]

    print(f"[robustness] held-out test set size: {len(held)}")
    print("\n[baseline]")
    base = load_checkpoint(baseline_ckpt)
    base_results = robustness_matrix(base, held)
    print("\n[augmented]")
    aug = load_checkpoint(augmented_ckpt)
    aug_results = robustness_matrix(aug, held)
    return {"baseline": base_results, "augmented": aug_results}
