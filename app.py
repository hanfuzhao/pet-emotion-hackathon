"""Gradio app: Pet Emotion Classifier with live robustness demo.

The app lets a user upload a pet photo, then dial in a corruption type +
severity to see how the trained model holds up. If both the baseline and
augmented checkpoints are present, it shows them side by side -- that's the
"see the augmentation pay off in real time" pitch moment.

Deployed on Hugging Face Spaces. See README for upload instructions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# app.py sits at the repo root next to the `src/` folder (HF Spaces convention).
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from src.evaluate import CORRUPTIONS, load_checkpoint
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = ROOT / "models"

# Lazy-load checkpoints so the app starts even if one is missing.
_eval_tx = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_loaded: dict[str, torch.nn.Module] = {}

def _model(name: str):
    if name in _loaded:
        return _loaded[name]
    path = MODELS_DIR / f"{name}.pt"
    if not path.exists():
        return None
    _loaded[name] = load_checkpoint(path, device=DEVICE)
    return _loaded[name]


@torch.no_grad()
def _predict(model, pil_img: Image.Image) -> dict:
    x = _eval_tx(pil_img).unsqueeze(0).to(DEVICE)
    probs = F.softmax(model(x), dim=1)[0].cpu().tolist()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def predict(image, corruption: str, severity: int):
    """Apply corruption then run both models (if available)."""
    if image is None:
        return None, {}, {}, "Upload a pet photo to start."

    if image.mode != "RGB":
        image = image.convert("RGB")

    corrupted = CORRUPTIONS[corruption](image, int(severity)) if corruption != "none" else image

    aug = _model("augmented")
    base = _model("baseline")

    aug_pred = _predict(aug, corrupted) if aug is not None else {}
    base_pred = _predict(base, corrupted) if base is not None else {}

    if aug is None and base is None:
        msg = "No model checkpoints found in `models/`. Train one first (see README)."
    elif base is None:
        msg = "Baseline checkpoint missing — showing augmented model only."
    elif aug is None:
        msg = "Augmented checkpoint missing — showing baseline only."
    else:
        msg = "Both models loaded. Crank up the severity to see how augmentation buys robustness."

    return corrupted, aug_pred, base_pred, msg


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pet Emotion Robustness") as demo:
        gr.Markdown(
            "# Pet Emotion Classifier — Robustness Demo\n"
            "Upload a pet photo, pick a real-world corruption, drag the severity slider.\n"
            "Watch the **augmented** model (trained with thoughtful augmentations) hold its confidence "
            "while the **baseline** model gets confused."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Pet photo", height=320)
                corruption = gr.Dropdown(
                    choices=["none", *CORRUPTIONS.keys()],
                    value="none",
                    label="Corruption",
                )
                severity = gr.Slider(0, 4, value=0, step=1, label="Severity (0=clean, 4=heavy)")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                preview = gr.Image(label="Model input (after corruption)", height=320)
                aug_out = gr.Label(label="Augmented model")
                base_out = gr.Label(label="Baseline model")
                status = gr.Markdown()

        btn.click(predict, [inp, corruption, severity], [preview, aug_out, base_out, status])
        # Live update as the user drags sliders / changes inputs.
        for trig in (corruption.change, severity.change, inp.change):
            trig(predict, [inp, corruption, severity], [preview, aug_out, base_out, status])

        gr.Markdown(
            "### How to read this\n"
            "- **Severity 0** is the original image; both models should agree.\n"
            "- As severity grows (blur, low light, occlusion, etc.) the **baseline** model's "
            "confidence usually swings or flips, while the **augmented** model stays closer to its "
            "clean-image prediction.\n"
            "- That gap is the value of choosing augmentations that match the real-world noise "
            "your users will throw at the model."
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
