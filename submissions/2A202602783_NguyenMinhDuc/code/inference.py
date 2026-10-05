"""Inference utilities for validation/test-time experiments."""
from __future__ import annotations

import copy

import numpy as np


def predict_logits(model, loader, device, view=None):
    import torch

    model.eval()
    filenames, labels, logits = [], [], []
    with torch.inference_mode():
        for x, y, fn in loader:
            x = x.to(device, non_blocking=True)
            if view is not None:
                x = view(x)
            out = model(x)
            filenames.extend(list(fn))
            labels.append(y.cpu().numpy())
            logits.append(out.detach().cpu().numpy())
    return filenames, np.concatenate(labels), np.concatenate(logits)


def view_identity(x):
    return x


def view_hflip(x):
    import torch
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    _, _, h, w = x.shape
    if crop > h or crop > w:
        raise ValueError("crop must be <= input height/width")
    coords = [
        (0, 0),
        (0, w - crop),
        (h - crop, 0),
        (h - crop, w - crop),
        ((h - crop) // 2, (w - crop) // 2),
    ]
    return [x[:, :, y:y + crop, xx:xx + crop] for y, xx in coords]


def views_multiscale(x, sizes):
    import torch.nn.functional as F
    return [F.interpolate(x, size=(int(s), int(s)), mode="bilinear", align_corners=False) for s in sizes]


def _softmax_np(z):
    z = np.asarray(z)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    arr = [np.asarray(z) for z in logits_per_view]
    if not arr:
        raise ValueError("Need at least one view")
    space = space.lower()
    if space == "prob":
        probs = np.mean([_softmax_np(z) for z in arr], axis=0)
    elif space == "logit":
        probs = _softmax_np(np.mean(arr, axis=0))
    else:
        raise ValueError(f"Unknown aggregation space: {space}")
    return probs / probs.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs):
    arr = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    if not arr:
        raise ValueError("Need at least one probability array")
    probs = np.mean(arr, axis=0)
    return probs / probs.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    import torch
    import torch.nn.functional as F

    logits = torch.as_tensor(val_logits, dtype=torch.float32)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(1e-3, 100.0)
        loss = F.cross_entropy(logits / t, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(1e-3, 100.0).detach().item())


def apply_temperature(logits, T: float):
    if T <= 0:
        raise ValueError("Temperature must be positive")
    return _softmax_np(np.asarray(logits) / float(T))


def _fuse_pair(conv, bn):
    import torch
    import torch.nn as nn

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(conv.weight.device)
    w = conv.weight.reshape(conv.out_channels, -1)
    b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data.copy_((w * scale.reshape(-1, 1)).reshape_as(conv.weight))
    fused.bias.data.copy_(bn.bias + (b - bn.running_mean) * scale)
    return fused


def _fuse_children(module):
    import torch.nn as nn

    prev_name, prev_child = None, None
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d) and isinstance(prev_child, nn.Conv2d):
            setattr(module, prev_name, _fuse_pair(prev_child, child))
            setattr(module, name, nn.Identity())
            prev_name, prev_child = None, None
        else:
            _fuse_children(child)
            prev_name, prev_child = name, child


def fuse_conv_bn(model):
    fused = copy.deepcopy(model).eval()
    _fuse_children(fused)
    return fused
