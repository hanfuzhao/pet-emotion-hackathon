import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from src.evaluate import CORRUPTIONS, load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = ROOT / "models"

_tx = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_cache = {}


def _model(name):
    if name in _cache:
        return _cache[name]
    p = MODELS_DIR / f"{name}.pt"
    if not p.exists():
        return None
    _cache[name] = load_checkpoint(p, device=DEVICE)
    return _cache[name]


@torch.no_grad()
def _infer(model, img):
    x = _tx(img).unsqueeze(0).to(DEVICE)
    probs = F.softmax(model(x), dim=1)[0].cpu().tolist()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def predict(image, corruption, severity):
    if image is None:
        return None, {}, {}, "Upload a pet photo to start."
    if image.mode != "RGB":
        image = image.convert("RGB")

    if corruption != "none":
        image_in = CORRUPTIONS[corruption](image, int(severity))
    else:
        image_in = image

    aug = _model("augmented")
    base = _model("baseline")
    aug_pred = _infer(aug, image_in) if aug else {}
    base_pred = _infer(base, image_in) if base else {}

    if not aug and not base:
        msg = "No model checkpoints found in models/."
    elif not base:
        msg = "Baseline checkpoint missing."
    elif not aug:
        msg = "Augmented checkpoint missing."
    else:
        msg = "Drag the severity slider to see how augmentation holds up."

    return image_in, aug_pred, base_pred, msg


def build_ui():
    with gr.Blocks(title="Pet Emotion Robustness") as demo:
        gr.Markdown(
            "# Pet Emotion Classifier — Robustness Demo\n"
            "Upload a pet photo, pick a corruption, drag the severity slider. "
            "Watch the augmented model hold its confidence while the baseline drifts."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Pet photo", height=320)
                corruption = gr.Dropdown(
                    choices=["none", *CORRUPTIONS.keys()],
                    value="none",
                    label="Corruption",
                )
                severity = gr.Slider(0, 4, value=0, step=1, label="Severity (0=clean, 4=heavy)")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                preview = gr.Image(label="Model input (after corruption)", height=320)
                aug_out = gr.Label(label="Augmented model")
                base_out = gr.Label(label="Baseline model")
                status = gr.Markdown()

        btn.click(predict, [inp, corruption, severity], [preview, aug_out, base_out, status])
        for trig in (corruption.change, severity.change, inp.change):
            trig(predict, [inp, corruption, severity], [preview, aug_out, base_out, status])

        gr.Markdown(
            "Severity 0 is the original photo. As severity rises (blur, low light, occlusion, etc.) "
            "the baseline's confidence swings; the augmented model stays closer to its clean-image call. "
            "That gap is what thoughtful augmentation buys you."
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
