---
title: Pet Emotion Robustness
emoji: 🐾
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: "5.0.1"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# Pet Emotion Classifier (Transfer Learning + Augmentation Robustness)

Mini-Hackathon #1: How Can Machines See What Matters?

A 4-class pet-emotion classifier (angry / happy / sad / other) built with ResNet18 transfer learning, used as a vehicle to show how thoughtful data augmentation buys real-world robustness instead of just chasing clean-test accuracy.

The setup: train two models on the same data with the same seed. One sees only resize + flip; the other sees augmentations that mimic real-world photo variation (lighting, blur, perspective, occlusion). Evaluate both on a held-out test set that's been corrupted in 6 different ways at 5 severity levels each. The gap between the two lines is what augmentation actually buys you.

## Links

- Live app: https://huggingface.co/spaces/HanfuZhao781/pet-emotion-robustness
- GitHub repo: https://github.com/hanfuzhao/pet-emotion-hackathon

The Space takes a minute to wake up if it's been idle.

## Problem

People photograph pets in messy conditions: dim kitchens, blurry action shots, hands and toys in frame, phones held at weird angles. A classifier that hits 90% on a clean curated test set can drop to 30% on a heavily blurred photo. We make those failure modes explicit by training and evaluating against them.

## Approach

| Decision | Choice |
|---|---|
| Pretrained backbone | ResNet18 (ImageNet) |
| Transfer learning | Two-phase: head-only first, then unfreeze layer4 |
| Data | Kaggle "Pets Facial Expression Recognition" (~1000 images, 4 classes, 250 per class) |
| Augmentations | ColorJitter, RandomRotation, RandomPerspective, RandomResizedCrop, GaussianBlur, RandomErasing |
| Eval | 6 corruption families x 5 severities, same held-out test set for both models |

Each augmentation is tied to a specific real-world failure mode:
- `ColorJitter`: indoor warm light vs. outdoor sun vs. dim rooms
- `GaussianBlur`: motion blur, cheap phone cameras
- `RandomRotation` + `RandomPerspective`: phones held off-axis
- `RandomResizedCrop`: close-ups vs. full-body shots
- `RandomErasing`: hand, collar, food bowl, toy occluding part of the pet
- `RandomHorizontalFlip`: free invariance

## Project layout

```
hackathon1/
├── app.py                      # Gradio app
├── requirements.txt
├── notebooks/train_colab.ipynb
├── scripts/
│   ├── run_robustness_eval.py
│   └── deploy_hf_space.py
├── src/
│   ├── data.py                 # dataset + augmentation pipelines
│   ├── model.py                # ResNet18 + two-phase transfer learning
│   ├── train.py                # training loop
│   └── evaluate.py             # corruption-based robustness eval
├── models/                     # weights (gitignored, ~45 MB each)
└── assets/                     # plots + result JSON
```

## How to run

### Local (Mac MPS or CUDA)

```bash
pip install -r requirements.txt

# Place dataset as data/raw/Angry/*, data/raw/happy/*, data/raw/Sad/*, data/raw/Other/*

python -m src.train --data_dir data/raw --train_mode baseline  --out models/baseline.pt
python -m src.train --data_dir data/raw --train_mode augmented --out models/augmented.pt
python scripts/run_robustness_eval.py
python app.py
```

### Colab

Open `notebooks/train_colab.ipynb`, update `REPO_URL`, select GPU runtime, run all.

### Deploy to HF Spaces

```bash
python scripts/deploy_hf_space.py --username <your-hf-username> --space pet-emotion-robustness
```

The YAML frontmatter at the top of this README is HF Spaces compatible, so linking the Space directly to this GitHub repo also works.

## Results

Both models trained on Apple-silicon MPS, ~3 minutes each. Same seeded splits (700 train / 150 val / 150 test). 4-class chance accuracy is 25%.

### Headline

| Metric | Baseline | Augmented | Delta |
|---|---:|---:|---:|
| Clean test accuracy | **0.900** | 0.853 | -4.7pp |
| Severity-4 brightness | 0.593 | **0.680** | +8.7pp |
| Severity-4 darkness  | 0.700 | **0.767** | +6.7pp |
| Severity-4 blur      | 0.293 | **0.380** | +8.7pp |
| Severity-4 JPEG (Q=8)| 0.447 | **0.513** | +6.7pp |
| Severity-4 rotation  | 0.707 | 0.660 | -4.7pp |
| Severity-4 occlusion | 0.713 | 0.693 | -2.0pp |

### Reading the table

The augmented model gives up 4.7pp of clean accuracy and gains back **6-9pp on the heaviest version of brightness, darkness, blur, and JPEG corruption**. That is exactly the trade you want when users will not be feeding the model clean curated photos.

The two losses are informative:
- Rotation: severity-4 is 45 degrees but our `RandomRotation(15)` only trained on +/- 15. The model never saw extreme rotations, so it never learned to handle them. We would push that augmentation harder in a second iteration.
- Occlusion: roughly a tie. Our `RandomErasing` patches are similar in size to the eval occlusion patches, so there is not much new information for the model to gain.

![robustness comparison](assets/robustness_comparison.png)

The brightness, darkness, and blur panels are where the augmented model (green) clearly stays above the baseline (red) as severity grows. The blur panel is the most dramatic: baseline drops from 90% to 30% (worse than random for some classes), while augmented holds at 38%.

`assets/aug_samples.png` shows random samples through the augmentation pipeline so you can see what the model was trained on.

## GitHub workflow

Trunk-based with feature branches and `--no-ff` merges, so the history graphs cleanly:

```
feat/data-and-augmentation   feat/transfer-learning      feat/robustness-eval
feat/training-notebook        feat/gradio-app             feat/training-results
feat/mps-support              fix/space-py313             fix/upgrade-gradio5
fix/pin-python311             docs/real-results           docs/live-links
```

Each branch was merged with `--no-ff` so the merge commits are visible in `git log --graph`.

## License

MIT.
