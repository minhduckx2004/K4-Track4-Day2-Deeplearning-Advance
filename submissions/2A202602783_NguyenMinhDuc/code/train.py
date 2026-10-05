"""Training loop shared by all DeepWeeds experiments."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import compute_metrics, save_predictions  # noqa: E402
from dataset import NUM_CLASSES, build_transforms, check_split, load_split, make_loader  # noqa: E402
from losses import build_criterion, class_weights, mix_batch, mixed_loss  # noqa: E402
from model import build_model, count_gmacs, count_params, param_groups  # noqa: E402


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "../runs"
    pred_dir: str = "../predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    import torch

    return torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    import torch

    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    def __init__(self, model, decay: float):
        import copy

        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model) -> None:
        with _torch_no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                src = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_(src.to(v.device), alpha=1.0 - self.decay)
                else:
                    v.copy_(src.to(v.device))


class _torch_no_grad:
    def __enter__(self):
        import torch
        self.ctx = torch.no_grad()
        return self.ctx.__enter__()

    def __exit__(self, *args):
        return self.ctx.__exit__(*args)


def _keep_frozen_bn_eval(model) -> None:
    try:
        from model import _set_frozen_backbone_bn_eval
        _set_frozen_backbone_bn_eval(model)
    except Exception:
        pass


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    import torch

    model.train()
    _keep_frozen_bn_eval(model)
    losses = []
    t0 = time.perf_counter()
    use_amp = bool(cfg.amp and device.type == "cuda")

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        targets = y
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        losses.append(float(loss.detach().cpu()))

    return {
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "epoch_time_s": time.perf_counter() - t0,
    }


def evaluate(model, loader, criterion, device):
    import torch

    model.eval()
    filenames, labels, logits_all, losses = [], [], [], []
    with torch.inference_mode():
        for x, y, fn in loader:
            x = x.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y_dev)
            filenames.extend(list(fn))
            labels.append(y.numpy())
            logits_all.append(logits.detach().cpu().numpy())
            losses.append(float(loss.detach().cpu()) * len(y))
    y_true = np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)
    logits_np = np.concatenate(logits_all) if logits_all else np.empty((0, NUM_CLASSES))
    loss_mean = float(np.sum(losses) / max(1, len(y_true)))
    return filenames, y_true, logits_np, loss_mean


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if not history:
        return
    df = pd.DataFrame(history)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=140)
    ax1.plot(df["epoch"], df["train_loss"], label="train loss")
    ax1.plot(df["epoch"], df["val_loss"], label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_macro_f1"], color="tab:green", label="val macro-F1")
    ax2.set_ylabel("macro-F1")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _criterion_for_cfg(cfg: Config, train_df, device):
    weight = None
    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).to_numpy()
        beta = 0.0 if cfg.class_weight_beta is None else cfg.class_weight_beta
        weight = class_weights(counts, beta=beta).to(device)
    return build_criterion(
        cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=weight,
    )


def run(cfg: Config) -> dict:
    import torch

    set_seed(cfg.seed)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    split_report = check_split(train_df, val_df, test_df, cfg.images_dir)
    (out / "split_report.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")

    train_loader = make_loader(train_df, cfg.images_dir, build_transforms(True, cfg.img_size, cfg.aug),
                               cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, build_transforms(False, cfg.img_size, cfg.aug),
                             cfg.batch_size, False, None, cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[{cfg.exp_id}/seed{cfg.seed}] device={device} "
        f"torch_cuda={torch.cuda.is_available()} epochs={cfg.epochs} batch_size={cfg.batch_size}",
        flush=True,
    )
    if device.type != "cuda":
        print(
            "WARNING: CUDA is not available in this Python environment. "
            "Training will run on CPU and can be extremely slow.",
            flush=True,
        )
    model = build_model(cfg.backbone, pretrained=True, num_classes=NUM_CLASSES,
                        drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    criterion = _criterion_for_cfg(cfg, train_df, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    best_f1, best_epoch, history = -1.0, -1, []
    best_path = out / "best.pt"
    for epoch in range(cfg.epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        eval_model = ema.module.to(device) if ema is not None else model
        filenames, y_true, logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        probs = _softmax(logits)
        metrics = compute_metrics(y_true, probs.argmax(1), probs)
        row = {
            "epoch": epoch,
            **tr,
            "val_loss": val_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_top1": metrics["top1"],
            "val_ece": metrics["ece"],
        }
        history.append(row)
        print(
            f"[{cfg.exp_id}/seed{cfg.seed}] epoch {epoch + 1}/{cfg.epochs} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} val_top1={row['val_top1']:.4f} "
            f"lr={row['lr']:.2e} time={row['epoch_time_s']:.1f}s",
            flush=True,
        )
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save({
                "model": eval_model.state_dict(),
                "cfg": asdict(cfg),
                "epoch": epoch,
                "macro_f1": best_f1,
            }, best_path)

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    plot_curves(history, Path(cfg.pred_dir).parent / "curves" / f"{cfg.exp_id}_seed{cfg.seed}.png",
                f"{cfg.exp_id} seed{cfg.seed} {cfg.backbone}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    filenames, y_true, logits, val_loss = evaluate(model, val_loader, criterion, device)
    probs = _softmax(logits)
    np.save(out / "val_logits.npy", logits)
    save_predictions(pred_path(cfg, "val"), filenames, y_true, probs)
    val_metrics = compute_metrics(y_true, probs.argmax(1), probs)

    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = make_loader(test_df, cfg.images_dir, build_transforms(False, cfg.img_size, cfg.aug),
                                  cfg.batch_size, False, None, cfg.num_workers)
        filenames, y_true, logits, _ = evaluate(model, test_loader, criterion, device)
        probs = _softmax(logits)
        np.save(out / "test_logits.npy", logits)
        save_predictions(pred_path(cfg, "test"), filenames, y_true, probs)
        test_metrics = compute_metrics(y_true, probs.argmax(1), probs)

    summary = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "val_macro_f1": val_metrics["macro_f1"],
        "val_top1": val_metrics["top1"],
        "val_ece": val_metrics["ece"],
        "params_m": count_params(model),
        "gmac": count_gmacs(model, cfg.img_size),
        "device": str(device),
    }
    if test_metrics is not None:
        summary.update({"test_macro_f1": test_metrics["macro_f1"], "test_top1": test_metrics["top1"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    field_map = {f.name: f for f in fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        if key not in field_map:
            raise KeyError(f"Unknown Config field: {key}")
        cur = getattr(Config(), key)
        low = value.lower()
        if low in {"none", "null"}:
            out[key] = None
        elif isinstance(cur, bool):
            out[key] = low in {"1", "true", "yes", "y", "on"}
        elif isinstance(cur, int) and not isinstance(cur, bool):
            out[key] = int(value)
        elif isinstance(cur, float):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
