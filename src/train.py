import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .data import build_loaders, CLASS_NAMES
from .model import build_model, unfreeze_last_block, trainable_parameters, count_trainable


def _run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc="train" if train else "val ", leave=False)
    with torch.set_grad_enabled(train):
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(loss=loss_sum / total, acc=correct / total)
    return loss_sum / total, correct / total


def train(data_dir, out_path, train_mode="augmented", batch_size=32,
          head_epochs=5, finetune_epochs=5, head_lr=1e-3, finetune_lr=1e-4,
          seed=42, device=None, num_workers=0):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[device] {device}")
    print(f"[mode]   {train_mode}")

    train_loader, val_loader, test_loader = build_loaders(
        root=data_dir, batch_size=batch_size, train_mode=train_mode,
        seed=seed, num_workers=num_workers,
    )
    print(f"[data]   train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    history = {"phase1": [], "phase2": []}

    print(f"[phase1] head only, trainable params = {count_trainable(model):,}")
    optimizer = AdamW(trainable_parameters(model), lr=head_lr, weight_decay=1e-4)
    for e in range(head_epochs):
        _, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        _, va_acc = _run_epoch(model, val_loader, criterion, None, device, False)
        print(f"  epoch {e+1}/{head_epochs}  train acc={tr_acc:.3f}  val acc={va_acc:.3f}")
        history["phase1"].append({"epoch": e + 1, "train_acc": tr_acc, "val_acc": va_acc})

    unfreeze_last_block(model)
    print(f"[phase2] fine-tune layer4, trainable params = {count_trainable(model):,}")
    optimizer = AdamW(trainable_parameters(model), lr=finetune_lr, weight_decay=1e-4)
    for e in range(finetune_epochs):
        _, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        _, va_acc = _run_epoch(model, val_loader, criterion, None, device, False)
        print(f"  epoch {e+1}/{finetune_epochs}  train acc={tr_acc:.3f}  val acc={va_acc:.3f}")
        history["phase2"].append({"epoch": e + 1, "train_acc": tr_acc, "val_acc": va_acc})

    _, te_acc = _run_epoch(model, test_loader, criterion, None, device, False)
    print(f"[test]   clean test acc = {te_acc:.3f}")
    history["clean_test_acc"] = te_acc
    history["train_mode"] = train_mode

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "train_mode": train_mode,
        "history": history,
    }, out_path)
    print(f"[save]   {out_path}")
    out_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    return history


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--out", default="models/augmented.pt")
    p.add_argument("--train_mode", default="augmented", choices=["baseline", "augmented"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--head_epochs", type=int, default=5)
    p.add_argument("--finetune_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(
        data_dir=args.data_dir,
        out_path=args.out,
        train_mode=args.train_mode,
        batch_size=args.batch_size,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
