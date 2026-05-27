"""Create a Hugging Face Space and upload the project files + weights.

Usage:
    python scripts/deploy_hf_space.py --username HanfuZhao781 --space pet-emotion-robustness

The script:
  1. Creates the Space (gradio SDK) if it doesn't exist.
  2. Uploads app.py, requirements.txt, src/, and models/*.pt.
  3. Prints the Space URL.

Requires that you've already logged in via `huggingface-cli login` (or that a
valid token is present at ~/.cache/huggingface/token).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

ROOT = Path(__file__).resolve().parent.parent


def deploy(username: str, space_name: str, private: bool = False) -> str:
    repo_id = f"{username}/{space_name}"
    api = HfApi()

    print(f"[hf] creating space {repo_id} (if missing)...")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
    )

    # Stage a clean folder so we don't upload .git, data/, notebooks/, etc.
    stage_dir = ROOT / ".hf_space_stage"
    if stage_dir.exists():
        import shutil
        shutil.rmtree(stage_dir)
    stage_dir.mkdir()

    import shutil
    # README has the YAML frontmatter HF Spaces needs.
    shutil.copy(ROOT / "README.md", stage_dir / "README.md")
    shutil.copy(ROOT / "app.py", stage_dir / "app.py")
    shutil.copy(ROOT / "requirements.txt", stage_dir / "requirements.txt")
    shutil.copytree(ROOT / "src", stage_dir / "src")

    models_src = ROOT / "models"
    models_dst = stage_dir / "models"
    models_dst.mkdir()
    weights = sorted(models_src.glob("*.pt"))
    if not weights:
        print("[hf] WARNING: no .pt weights found in models/ -- Space will run but predictions will fail.")
    for w in weights:
        print(f"[hf]   staging weights: {w.name} ({w.stat().st_size / 1e6:.1f} MB)")
        shutil.copy(w, models_dst / w.name)

    print(f"[hf] uploading {stage_dir} -> {repo_id}")
    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy: app + trained weights",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"[hf] done: {url}")

    shutil.rmtree(stage_dir)
    return url


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--username", required=True)
    p.add_argument("--space", default="pet-emotion-robustness")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()
    deploy(args.username, args.space, args.private)


if __name__ == "__main__":
    _cli()
