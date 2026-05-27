import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent


def deploy(username, space_name, private=False):
    repo_id = f"{username}/{space_name}"
    api = HfApi()

    print(f"[hf] creating space {repo_id} (if missing)")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
    )

    stage = ROOT / ".hf_space_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()

    shutil.copy(ROOT / "README.md", stage / "README.md")
    shutil.copy(ROOT / "app.py", stage / "app.py")
    shutil.copy(ROOT / "requirements.txt", stage / "requirements.txt")
    shutil.copytree(
        ROOT / "src", stage / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    examples_src = ROOT / "examples"
    if examples_src.exists():
        shutil.copytree(examples_src, stage / "examples")

    models_dst = stage / "models"
    models_dst.mkdir()
    weights = sorted((ROOT / "models").glob("*.pt"))
    if not weights:
        print("[hf] WARNING: no .pt weights in models/")
    for w in weights:
        print(f"[hf]   staging weights: {w.name} ({w.stat().st_size / 1e6:.1f} MB)")
        shutil.copy(w, models_dst / w.name)

    print(f"[hf] uploading {stage} -> {repo_id}")
    api.upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Update app + weights",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"[hf] done: {url}")
    shutil.rmtree(stage)
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
