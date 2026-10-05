"""Model construction and optimizer parameter grouping."""
from __future__ import annotations

from copy import deepcopy

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def _canonical_name(name: str) -> str:
    return SUGGESTED_BACKBONES.get(name, name)


def _classifier_param_ids(model) -> set[int]:
    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    if head is None:
        return set()
    if isinstance(head, tuple):
        modules = [m for m in head if hasattr(m, "parameters")]
    elif hasattr(head, "parameters"):
        modules = [head]
    else:
        modules = []
    return {id(p) for m in modules for p in m.parameters()}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    import timm

    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"Unknown init: {init}")
    model = timm.create_model(
        _canonical_name(name),
        pretrained=(pretrained and init != "scratch"),
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    model.lab_pretrained_cfg = deepcopy(getattr(model, "pretrained_cfg", {}))
    model.lab_init = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    head_ids = _classifier_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.lab_backbone_frozen = True
    _set_frozen_backbone_bn_eval(model)


def _set_frozen_backbone_bn_eval(model) -> None:
    if not getattr(model, "lab_backbone_frozen", False):
        return
    import torch.nn as nn

    head_ids = _classifier_param_ids(model)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            params = list(module.parameters(recurse=False))
            if not params or all(id(p) not in head_ids for p in params):
                module.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head_ids = _classifier_param_ids(model)
    backbone_decay, backbone_nodecay, head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids or name.startswith(("head.", "fc.", "classifier.")):
            head.append(p)
        elif p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            backbone_nodecay.append(p)
        else:
            backbone_decay.append(p)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_nodecay:
        groups.append({"params": backbone_nodecay, "lr": lr_backbone, "weight_decay": 0.0})
    if head:
        groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
    return groups


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    try:
        from thop import profile
        import torch

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        model.train(was_training)
        return float(macs) / 1e9
    except Exception:
        return float("nan")
