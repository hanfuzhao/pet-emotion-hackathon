"""Run the robustness comparison after both models are trained.

Saves:
  - assets/robustness_results.json   (raw per-corruption-x-severity accuracies)
  - assets/robustness_comparison.png (six-panel comparison plot)
  - assets/aug_samples.png           (random augmented training samples)
  - assets/summary.json              (headline numbers for README/pitch)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.data import PetEmotionDataset, get_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.evaluate import compare_models, CORRUPTIONS


def main():
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    models_dir = ROOT / "models"

    # 1) Sample augmentation grid for the pitch
    print("[viz] augmentation samples")
    ds = PetEmotionDataset(root=ROOT / "data" / "raw", transform=None)
    tx = get_transforms("augmented")
    rng = np.random.default_rng(0)

    def denorm(t):
        t = t.clone()
        for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            t[c] = t[c] * s + m
        return t.clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax in axes.flat:
        img, label = ds[int(rng.integers(len(ds)))]
        aug = tx(img)
        ax.imshow(denorm(aug))
        ax.set_title(["unhappy", "happy"][label])
        ax.axis("off")
    plt.suptitle("Random samples through the augmented training pipeline")
    plt.tight_layout()
    plt.savefig(assets / "aug_samples.png", dpi=110, bbox_inches="tight")
    plt.close()

    # 2) Robustness matrix
    print("[eval] robustness matrix (baseline vs augmented)")
    results = compare_models(
        baseline_ckpt=models_dir / "baseline.pt",
        augmented_ckpt=models_dir / "augmented.pt",
        data_dir=ROOT / "data" / "raw",
    )
    (assets / "robustness_results.json").write_text(json.dumps(results, indent=2))

    # 3) Comparison plot
    print("[viz] comparison plot")
    corruptions = list(results["baseline"].keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, name in zip(axes.flat, corruptions):
        sevs = sorted(int(k) for k in results["baseline"][name].keys())
        base_acc = [results["baseline"][name][s] for s in sevs]
        aug_acc = [results["augmented"][name][s] for s in sevs]
        ax.plot(sevs, base_acc, marker="o", label="baseline", color="#d9534f")
        ax.plot(sevs, aug_acc, marker="s", label="augmented", color="#5cb85c")
        ax.set_title(name)
        ax.set_xlabel("severity")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.suptitle("Robustness under image corruptions (held-out test set)")
    plt.tight_layout()
    plt.savefig(assets / "robustness_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()

    # 4) Headline summary
    summary = {}
    for name in ("baseline", "augmented"):
        all_accs = [v for c in results[name].values() for v in c.values()]
        summary[f"{name}_clean_acc"] = results[name]["brightness"][0]  # sev=0 ~ clean
        summary[f"{name}_mean_robust_acc"] = float(np.mean(all_accs))
    summary["delta_mean_robust_acc"] = (
        summary["augmented_mean_robust_acc"] - summary["baseline_mean_robust_acc"]
    )
    summary["per_corruption_severity4"] = {
        c: {
            "baseline": results["baseline"][c][4],
            "augmented": results["augmented"][c][4],
        }
        for c in CORRUPTIONS
    }
    (assets / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[summary]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
