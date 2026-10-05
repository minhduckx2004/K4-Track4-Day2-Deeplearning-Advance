"""Run planned Lab Day 2 experiments.

Examples:
    python experiments.py --mode smoke
    python experiments.py --mode backbones --epochs 10 --batch-size 16
    python experiments.py --mode training --backbone resnet50 --epochs 10 --batch-size 16
    python experiments.py --mode final --backbone resnet50 --epochs 10 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from train import Config, run


BACKBONES = [
    ("B01", "resnet50"),
    ("B02", "resnext50_32x4d"),
    ("B03", "convnext_tiny"),
    ("B04", "deit_small_patch16_224"),
    ("B05", "swin_tiny_patch4_window7_224"),
    ("B06", "efficientnet_b0"),
]


def base(args, exp_id: str, backbone: str, seed: int = 0) -> Config:
    return Config(
        exp_id=exp_id,
        backbone=backbone,
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        amp=True,
    )


def smoke(args):
    return [replace(base(args, "SMOKE", "efficientnet_b0"), epochs=1, batch_size=min(args.batch_size, 8), num_workers=0)]


def backbones(args):
    return [base(args, exp_id, backbone) for exp_id, backbone in BACKBONES]


def training(args):
    b = args.backbone
    return [
        base(args, "T00", b),
        replace(base(args, "T01_frozen", b), init="frozen"),
        replace(base(args, "T02_color", b), aug="color"),
        replace(base(args, "T03_ls", b), loss="ls", label_smoothing=0.1),
        replace(base(args, "T04_focal", b), loss="focal", focal_gamma=2.0),
        replace(base(args, "T05_cutmix", b), mix="cutmix", mix_alpha=1.0),
        replace(base(args, "T06_balanced", b), sampler="balanced"),
        replace(base(args, "T07_ema", b), ema_decay=0.999),
        replace(base(args, "T08_combo", b), aug="color", loss="ls", label_smoothing=0.1, mix="cutmix", mix_alpha=1.0, ema_decay=0.999),
    ]


def final(args):
    cfgs = []
    for seed in (0, 1, 2):
        cfgs.append(replace(base(args, "F01", args.backbone, seed), save_test_predictions=True))
        cfgs.append(replace(base(args, "T00FINAL", "resnet50", seed), save_test_predictions=True))
    return cfgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "backbones", "training", "final", "all"], default="smoke")
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--images-dir", default="../../../data/images")
    p.add_argument("--labels-dir", default="../../../data/labels")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    plans = []
    modes = ["smoke", "backbones", "training", "final"] if args.mode == "all" else [args.mode]
    for mode in modes:
        plans.extend(globals()[mode](args))

    print(json.dumps([asdict(c) for c in plans], indent=2))
    if args.dry_run:
        return
    for cfg in plans:
        print(f"\n=== RUN {cfg.exp_id} seed={cfg.seed} backbone={cfg.backbone} ===", flush=True)
        summary = run(cfg)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
