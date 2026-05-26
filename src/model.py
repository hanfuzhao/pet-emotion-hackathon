"""ResNet18 transfer learning for binary pet emotion classification.

Transfer learning strategy (two-phase):
  Phase 1 (head-only): freeze the entire backbone, train just the new fc layer.
                       Fast, prevents the noisy random head from destroying
                       pretrained features.
  Phase 2 (fine-tune):  unfreeze layer4 (last residual block) and continue
                        training with a 10x lower LR. Lets the high-level
                        features adapt to "pet face" semantics.

Lower-level features (edges, textures) stay frozen throughout -- they generalize
fine from ImageNet and we don't have enough pet data to retrain them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


NUM_CLASSES = 2


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a ResNet18 with a fresh classifier head.

    Returns the model with ALL parameters frozen except the new fc layer.
    Call `unfreeze_last_block` before phase-2 fine-tuning.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    for p in model.parameters():
        p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # New head is trainable by default; explicit for clarity.
    for p in model.fc.parameters():
        p.requires_grad = True

    return model


def unfreeze_last_block(model: nn.Module) -> None:
    """Unfreeze the last residual stage (layer4) for fine-tuning."""
    for p in model.layer4.parameters():
        p.requires_grad = True


def trainable_parameters(model: nn.Module):
    """Iterable of params with requires_grad=True (for the optimizer)."""
    return (p for p in model.parameters() if p.requires_grad)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
