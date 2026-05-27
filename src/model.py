import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4


def build_model(num_classes=NUM_CLASSES):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=weights)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    for p in m.fc.parameters():
        p.requires_grad = True
    return m


def unfreeze_last_block(m):
    for p in m.layer4.parameters():
        p.requires_grad = True


def trainable_parameters(m):
    return (p for p in m.parameters() if p.requires_grad)


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
