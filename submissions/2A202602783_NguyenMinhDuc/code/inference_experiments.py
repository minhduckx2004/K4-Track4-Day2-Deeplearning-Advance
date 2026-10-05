"""Run validation inference experiments for a saved checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import compute_metrics, save_predictions
from benchmark import latency_report
from dataset import NUM_CLASSES, build_transforms, load_split, make_loader
from inference import aggregate_views, apply_temperature, fit_temperature, predict_logits, view_hflip
from model import build_model
from train import Config, evaluate
from losses import build_criterion


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_run(run_root: Path):
    cfg = Config(**json.loads((run_root / "config.json").read_text(encoding="utf-8")))
    ckpt = torch.load(run_root / "best.pt", map_location="cpu")
    model = build_model(cfg.backbone, pretrained=False, num_classes=NUM_CLASSES, drop_rate=cfg.drop_rate, init="finetune")
    model.load_state_dict(ckpt["model"], strict=True)
    return cfg, model


def row(method, probs, y_true, latency=None, k=1):
    m = compute_metrics(y_true, probs.argmax(1), probs)
    out = {
        "method": method,
        "K": k,
        "macro_F1_val": m["macro_f1"],
        "top1_val": m["top1"],
        "ECE_val": m["ece"],
    }
    if latency:
        out.update({
            "latency_p50_ms": latency["p50"],
            "latency_p95_ms": latency["p95"],
            "latency_p99_ms": latency["p99"],
            "images_per_s": latency["images_per_s"],
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="../runs/B01/seed0")
    p.add_argument("--out-csv", default="../runs/inference_results.csv")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--latency-iters", type=int, default=80)
    args = p.parse_args()

    run_root = Path(args.run_root)
    cfg, model = load_run(run_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    _, val_df, _ = load_split(cfg.labels_dir, cfg.fold)
    loader = make_loader(val_df, cfg.images_dir, build_transforms(False, cfg.img_size, cfg.aug),
                         args.batch_size, False, None, cfg.num_workers)
    criterion = build_criterion("ce")

    filenames, y_true, logits, _ = evaluate(model, loader, criterion, device)
    p_i00 = softmax(logits)
    rows = []
    lat_fp32 = latency_report(model, 1, cfg.img_size, dtype="fp32", device=str(device), iters=args.latency_iters)
    rows.append(row("I00_1view", p_i00, y_true, lat_fp32, 1))
    save_predictions(Path("../predictions") / f"I00_{cfg.exp_id}_seed{cfg.seed}_val.csv", filenames, y_true, p_i00)

    _, _, logits_flip = predict_logits(model, loader, device, view=view_hflip)
    p_i01 = aggregate_views([logits, logits_flip], space="prob")
    rows.append(row("I01_hflip_prob", p_i01, y_true, None, 2))
    save_predictions(Path("../predictions") / f"I01_{cfg.exp_id}_seed{cfg.seed}_val.csv", filenames, y_true, p_i01)

    p_i03 = aggregate_views([logits, logits_flip], space="logit")
    rows.append(row("I03_hflip_logit", p_i03, y_true, None, 2))

    temp = fit_temperature(logits, y_true)
    p_i07 = apply_temperature(logits, temp)
    r = row("I07_temperature", p_i07, y_true, lat_fp32, 1)
    r["temperature"] = temp
    rows.append(r)
    save_predictions(Path("../predictions") / f"I07_{cfg.exp_id}_seed{cfg.seed}_val.csv", filenames, y_true, p_i07)

    if device.type == "cuda":
        try:
            lat_amp = latency_report(model, 1, cfg.img_size, dtype="amp", device=str(device), iters=args.latency_iters)
            rows.append({"method": "I08_amp_latency", "K": 1, "macro_F1_val": np.nan, "top1_val": np.nan,
                         "ECE_val": np.nan, "latency_p50_ms": lat_amp["p50"], "latency_p95_ms": lat_amp["p95"],
                         "latency_p99_ms": lat_amp["p99"], "images_per_s": lat_amp["images_per_s"]})
        except RuntimeError as exc:
            print(f"AMP latency skipped: {exc}")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
