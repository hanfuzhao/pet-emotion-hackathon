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
EXAMPLES_DIR = ROOT / "examples"

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
        return None, {}, {}, "👈 Pick an example below or upload your own pet photo."
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

    if not aug or not base:
        msg = "⚠️ Missing a model checkpoint."
    elif corruption == "none" or int(severity) == 0:
        msg = "Now pick a corruption and drag severity up to 4. Watch the **baseline** drift while the **augmented** model holds steady."
    else:
        top_a = max(aug_pred, key=aug_pred.get) if aug_pred else "?"
        top_b = max(base_pred, key=base_pred.get) if base_pred else "?"
        if top_a == top_b:
            msg = f"Both models still agree: **{top_a}**. Push severity higher to see them diverge."
        else:
            msg = f"📊 **They disagree!** Augmented says **{top_a}**, baseline says **{top_b}**. This is exactly the kind of split augmentation is meant to prevent."

    return image_in, aug_pred, base_pred, msg


CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

#hero {
    background: linear-gradient(135deg, #FF6B9D 0%, #FF8E53 100%);
    border-radius: 16px;
    padding: 36px 44px;
    margin-bottom: 8px;
    color: white;
    box-shadow: 0 4px 24px rgba(255, 107, 157, 0.25);
}
#hero h1 {
    font-size: 38px !important;
    font-weight: 800 !important;
    margin: 0 0 10px 0 !important;
    color: white !important;
    line-height: 1.15;
}
#hero p {
    font-size: 16px;
    margin: 0;
    opacity: 0.95;
    line-height: 1.5;
}
#hero .accent { color: #FFE66D; font-weight: 700; }

.gr-button-primary {
    background: linear-gradient(135deg, #FF6B9D 0%, #FF8E53 100%) !important;
    border: none !important;
    font-weight: 600 !important;
}
.gr-button-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(255, 107, 157, 0.3); }

#aug-label .label-name { color: #2d8a45 !important; }
#base-label .label-name { color: #c43d3a !important; }

#status-msg {
    background: #FFF8E7;
    border-left: 4px solid #FFD93D;
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 15px;
    line-height: 1.5;
    margin-top: 8px;
}

#aug-card, #base-card {
    border-radius: 12px;
    padding: 4px;
    border: 2px solid transparent;
}
#aug-card { border-color: #6BCB77; background: #F0FBF2; }
#base-card { border-color: #D9534F; background: #FDF2F2; }

.model-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.aug-tag { background: #6BCB77; color: white; }
.base-tag { background: #D9534F; color: white; }

footer { display: none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.pink,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#FAFAFA",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="0 1px 3px rgba(0,0,0,0.04)",
    block_radius="12px",
    input_border_color="#E5E7EB",
)


def build_ui():
    with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="Pet Emotion Robustness") as demo:
        gr.HTML("""
            <div id="hero">
                <h1>🐾 Pet Emotion Robustness</h1>
                <p>Two ResNet18 models trained on the same pet photos. One saw augmentations
                that mimic <span class="accent">messy real-world variation</span> &mdash;
                lighting, blur, occlusion. The other didn't. Drag the severity slider and
                <span class="accent">watch the gap open up</span>.</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 1. Pick a photo")
                inp = gr.Image(
                    type="pil",
                    label=None,
                    show_label=False,
                    height=300,
                    sources=["upload", "clipboard"],
                )
                example_paths = [
                    str(p) for p in (EXAMPLES_DIR.glob("*.jpg") if EXAMPLES_DIR.exists() else [])
                ]
                if example_paths:
                    gr.Examples(
                        examples=example_paths,
                        inputs=inp,
                        label="examples",
                        examples_per_page=8,
                    )

                gr.Markdown("### 2. Break the image")
                corruption = gr.Dropdown(
                    choices=["none", *CORRUPTIONS.keys()],
                    value="blur",
                    label="Corruption type",
                    info="Each maps to a real failure mode",
                )
                severity = gr.Slider(
                    0, 4, value=0, step=1,
                    label="Severity",
                    info="0 = clean original · 4 = heavy degradation",
                )
                btn = gr.Button("Run prediction", variant="primary", size="lg")

            with gr.Column(scale=7):
                gr.Markdown("### 3. Compare the models")
                preview = gr.Image(
                    label="What the model actually sees",
                    height=300,
                    interactive=False,
                )

                with gr.Row():
                    with gr.Column(elem_id="aug-card"):
                        gr.HTML('<div class="model-tag aug-tag">✓ Augmented</div>')
                        aug_out = gr.Label(
                            label="",
                            num_top_classes=4,
                            show_label=False,
                            elem_id="aug-label",
                        )
                    with gr.Column(elem_id="base-card"):
                        gr.HTML('<div class="model-tag base-tag">✗ Baseline</div>')
                        base_out = gr.Label(
                            label="",
                            num_top_classes=4,
                            show_label=False,
                            elem_id="base-label",
                        )

                status = gr.Markdown(
                    "👈 Pick an example or upload your own pet photo to start.",
                    elem_id="status-msg",
                )

        gr.Markdown(
            "**How to read it:** at severity 0 both models usually agree. "
            "As severity climbs, the baseline's top-class confidence swings or flips, "
            "while the augmented model stays closer to its clean-image call. "
            "That gap is what augmentation actually buys you in the wild."
        )

        btn.click(predict, [inp, corruption, severity], [preview, aug_out, base_out, status])
        for trig in (corruption.change, severity.change, inp.change):
            trig(predict, [inp, corruption, severity], [preview, aug_out, base_out, status])

    return demo


if __name__ == "__main__":
    build_ui().launch()
