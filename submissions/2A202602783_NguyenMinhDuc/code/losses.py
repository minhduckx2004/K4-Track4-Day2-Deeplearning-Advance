"""Losses and Mixup/CutMix utilities."""
from __future__ import annotations

import math


def build_criterion(kind: str = "ce", **kw):
    import torch.nn as nn

    kind = (kind or "ce").lower()
    weight = kw.get("weight", None)
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", kw.get("label_smoothing", 0.1)), weight=weight)
    if kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha", weight))
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted requires weight=")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown loss kind: {kind}")


class LabelSmoothingCE:
    def __init__(self, smoothing: float = 0.1, weight=None):
        import torch.nn as nn

        self.impl = nn.CrossEntropyLoss(label_smoothing=float(smoothing), weight=weight)

    def __call__(self, logits, target):
        return self.impl(logits, target)


class FocalLoss:
    def __init__(self, gamma: float = 2.0, alpha=None):
        import torch

        self.gamma = float(gamma)
        self.alpha = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)

    def __call__(self, logits, target):
        import torch.nn.functional as F

        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = loss * alpha.gather(0, target)
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    import torch

    counts = torch.as_tensor(counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    if beta and beta > 0:
        beta = float(beta)
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
    else:
        weights = 1.0 / counts
    return weights * (len(weights) / weights.sum())


def _rand_bbox(width: int, height: int, lam: float):
    import torch

    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = int(torch.randint(0, width, (1,)).item())
    cy = int(torch.randint(0, height, (1,)).item())
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    import torch

    if alpha <= 0:
        return x, (y, y, 1.0)
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    mode = mode.lower()

    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        x_mix = x.clone()
        _, _, h, w = x.shape
        x1, y1, x2, y2 = _rand_bbox(w, h, lam)
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        lam = 1.0 - box_area / float(w * h)
    else:
        raise ValueError(f"Unknown mix mode: {mode}")
    return x_mix, (y_a, y_b, float(lam))


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
