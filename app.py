import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio_client.utils as _gcu
_orig_get_type = _gcu.get_type
_orig_jsonschema = _gcu._json_schema_to_python_type

def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _orig_get_type(schema)

def _safe_jsonschema(schema, defs=None):
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _orig_jsonschema(schema, defs)

_gcu.get_type = _safe_get_type
_gcu._json_schema_to_python_type = _safe_jsonschema

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

EMOJI = {"angry": "😠", "happy": "😊", "sad": "😢", "other": "🤔"}

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
    return [(CLASS_NAMES[i], float(probs[i])) for i in range(len(CLASS_NAMES))]


def _render_bars(probs, accent):
    if not probs:
        return '<div style="color:#999;font-size:13px;padding:8px;">no model loaded</div>'
    ranked = sorted(probs, key=lambda x: -x[1])
    top_class = ranked[0][0]
    rows = []
    for cls, p in ranked:
        pct = int(round(p * 100))
        is_top = cls == top_class
        bar_color = accent if is_top else "#D5D5D5"
        text_weight = "700" if is_top else "500"
        emoji = EMOJI.get(cls, "•")
        rows.append(f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:{text_weight};color:#333;margin-bottom:3px;">
                <span>{emoji}&nbsp;&nbsp;{cls}</span>
                <span>{pct}%</span>
              </div>
              <div style="background:#EEE;border-radius:4px;height:8px;overflow:hidden;">
                <div style="background:{bar_color};height:100%;width:{pct}%;transition:width 0.3s;"></div>
              </div>
            </div>
        """)
    return f'<div style="padding:8px 4px;">{"".join(rows)}</div>'


def predict(image, corruption, severity):
    if image is None:
        empty = '<div style="color:#999;font-size:13px;padding:20px;text-align:center;">waiting for a photo…</div>'
        return None, empty, empty, "👈 Pick an example below or upload your own pet photo."

    if image.mode != "RGB":
        image = image.convert("RGB")

    if corruption != "none":
        image_in = CORRUPTIONS[corruption](image, int(severity))
    else:
        image_in = image

    aug = _model("augmented")
    base = _model("baseline")
    aug_probs = _infer(aug, image_in) if aug else []
    base_probs = _infer(base, image_in) if base else []

    aug_html = _render_bars(aug_probs, "#5CB85C")
    base_html = _render_bars(base_probs, "#D9534F")

    top_a = max(aug_probs, key=lambda x: x[1])[0] if aug_probs else "?"
    top_b = max(base_probs, key=lambda x: x[1])[0] if base_probs else "?"

    if corruption == "none" or int(severity) == 0:
        msg = "Now pick a corruption and drag severity up to 4. Watch the **baseline** drift while the **augmented** model holds steady."
    elif top_a == top_b:
        msg = f"Both models still agree: **{top_a}**. Push severity higher to see them split."
    else:
        msg = f"📊 **They disagree.** Augmented says **{top_a}**, baseline says **{top_b}**. That split is exactly what augmentation is meant to prevent."

    return image_in, aug_html, base_html, msg


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

#aug-card, #base-card {
    border-radius: 12px;
    padding: 16px;
    border: 2px solid transparent;
    margin-top: 0;
}
#aug-card { border-color: #5CB85C; background: #F0FBF2; }
#base-card { border-color: #D9534F; background: #FDF2F2; }

.model-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.aug-tag { background: #5CB85C; color: white; }
.base-tag { background: #D9534F; color: white; }

#status-msg {
    background: #FFF8E7;
    border-left: 4px solid #FFD93D;
    padding: 14px 18px;
    border-radius: 8px;
    font-size: 15px;
    line-height: 1.5;
    margin-top: 12px;
}

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
)


EMPTY_BARS = '<div style="color:#999;font-size:13px;padding:20px;text-align:center;">waiting for a photo…</div>'


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
                    show_label=False,
                    height=300,
                    sources=["upload", "clipboard"],
                )
                example_paths = sorted(
                    str(p) for p in (EXAMPLES_DIR.glob("*.jpg") if EXAMPLES_DIR.exists() else [])
                )
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
                        aug_out = gr.HTML(EMPTY_BARS)
                    with gr.Column(elem_id="base-card"):
                        gr.HTML('<div class="model-tag base-tag">✗ Baseline</div>')
                        base_out = gr.HTML(EMPTY_BARS)
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
    build_ui().launch(ssr_mode=False)
