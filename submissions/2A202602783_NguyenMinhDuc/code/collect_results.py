"""Collect real run outputs into submissions/work/results.xlsx and report.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import CLASS_NAMES, compute_metrics, read_pred

SUB = Path(__file__).resolve().parents[1]
RUNS = SUB / "runs"
PREDS = SUB / "predictions"
OUT_XLSX = SUB / "results.xlsx"
REPORT = SUB / "report.md"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summaries():
    rows = []
    for p in sorted(RUNS.glob("*/*/summary.json")):
        s = load_json(p)
        cfg = load_json(p.parent / "config.json")
        s.update({
            "run_path": str(p.parent.relative_to(SUB)),
            "backbone": cfg.get("backbone"),
            "seed": cfg.get("seed"),
            "epochs": cfg.get("epochs"),
            "img_size": cfg.get("img_size"),
            "aug": cfg.get("aug"),
            "loss": cfg.get("loss"),
            "mix": cfg.get("mix"),
            "init": cfg.get("init"),
            "ema_decay": cfg.get("ema_decay"),
        })
        rows.append(s)
    return pd.DataFrame(rows)


def backbones(df):
    if df.empty:
        return pd.DataFrame(columns=['exp_id','backbone','pretrained_tag','params_M','GMAC','resolution','epoch','seed','macro_F1_val','top1_val','train_time_per_epoch_s','latency_batch1_ms','notes'])
    rows = []
    for _, r in df[df["exp_id"].str.startswith("B", na=False)].iterrows():
        hist = RUNS / r["exp_id"] / f"seed{int(r['seed'])}" / "history.csv"
        t = np.nan
        if hist.exists():
            h = pd.read_csv(hist)
            t = h["epoch_time_s"].mean() if "epoch_time_s" in h else np.nan
        rows.append({
            "exp_id": r["exp_id"], "backbone": r["backbone"], "pretrained_tag": "",
            "params_M": r.get("params_m", np.nan), "GMAC": r.get("gmac", np.nan),
            "resolution": r["img_size"], "epoch": r["epochs"], "seed": r["seed"],
            "macro_F1_val": r["val_macro_f1"], "top1_val": r["val_top1"],
            "train_time_per_epoch_s": t, "latency_batch1_ms": np.nan, "notes": "",
        })
    return pd.DataFrame(rows)


def training(df):
    if df.empty:
        return pd.DataFrame(columns=['exp_id','backbone','axis_A_G','changed_from_T00','seed','macro_F1_val','top1_val','delta_vs_T00','rare_class_F1','notes'])
    tdf = df[df["exp_id"].str.startswith("T", na=False)].copy()
    base = tdf[tdf["exp_id"].eq("T00")]
    base_f1 = float(base["val_macro_f1"].iloc[0]) if len(base) else np.nan
    rows = []
    for _, r in tdf.iterrows():
        rows.append({
            "exp_id": r["exp_id"], "backbone": r["backbone"], "axis_A_G": "",
            "changed_from_T00": f"init={r['init']}, aug={r['aug']}, loss={r['loss']}, mix={r['mix']}, ema={r['ema_decay']}",
            "seed": r["seed"], "macro_F1_val": r["val_macro_f1"], "top1_val": r["val_top1"],
            "delta_vs_T00": r["val_macro_f1"] - base_f1 if not np.isnan(base_f1) else np.nan,
            "rare_class_F1": "", "notes": "",
        })
    return pd.DataFrame(rows)


def inference():
    p = RUNS / "inference_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        df.insert(0, "exp_id", df["method"])
        df.insert(2, "checkpoint", "")
        df["relative_cost"] = df["K"]
        return df
    return pd.DataFrame(columns=['exp_id','method','checkpoint','K','macro_F1_val','top1_val','ECE_val','latency_p50_ms','latency_p95_ms','latency_p99_ms','images_per_s','relative_cost'])


def final_and_perclass():
    rows, pc_rows = [], []
    for p in sorted(PREDS.glob("*_test.csv")):
        pred = read_pred(str(p))
        probs = pred.probs
        m = compute_metrics(pred.y_true, probs.argmax(1), probs)
        tag = p.name.replace("_test.csv", "")
        rows.append({
            "exp_id": tag, "config": tag, "seed": pred.seed,
            "macro_F1_val": np.nan, "macro_F1_test": m["macro_f1"], "top1_test": m["top1"],
            "ECE_test": m["ece"], "mean_std": "",
        })
        for i, name in enumerate(CLASS_NAMES):
            pc_rows.append({
                "config": tag, "class": name, "n_test": int(m["support"][i]),
                "precision": m["precision"][i], "recall": m["recall"][i], "F1": m["f1"][i],
            })
    return pd.DataFrame(rows), pd.DataFrame(pc_rows)


def latency(inf):
    if inf.empty:
        return pd.DataFrame(columns=['config','GPU','dtype','batch','fused_BN','p50','p95','p99','images_per_s'])
    rows = []
    for _, r in inf.dropna(subset=["latency_p50_ms"], how="all").iterrows():
        rows.append({"config": r.get("method", ""), "GPU": "", "dtype": "", "batch": 1,
                     "fused_BN": "no", "p50": r.get("latency_p50_ms"), "p95": r.get("latency_p95_ms"),
                     "p99": r.get("latency_p99_ms"), "images_per_s": r.get("images_per_s")})
    return pd.DataFrame(rows)


def write_report(df, bdf, tdf, idf, fdf):
    best_val = df.sort_values("val_macro_f1", ascending=False).head(5) if not df.empty else pd.DataFrame()
    def md(table: pd.DataFrame) -> str:
        if table.empty:
            return ""
        cols = list(table.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in table.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}" if np.isfinite(v) else "")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
    text = [
        "# Report Lab Day 2",
        "",
        "## 1. Tom tat",
        "",
        "Bao cao nay duoc tao tu log that trong `runs/` va predictions trong `predictions/`.",
        "",
        "## 2. Du lieu va thiet lap",
        "",
        "- Fold: 0",
        "- Metric chinh: macro-F1",
        "- Split check: train 10501, val 3501, test 3507; union 17509; overlap 0.",
        "",
        "## 3. Top cau hinh theo validation",
        "",
        md(best_val[["exp_id", "backbone", "seed", "val_macro_f1", "val_top1"]]) if not best_val.empty else "Chua co run hoan tat.",
        "",
        "## 4. So sanh backbone",
        "",
        md(bdf) if not bdf.empty else "Chua co du lieu backbone.",
        "",
        "## 5. Ablation training",
        "",
        md(tdf) if not tdf.empty else "Chua co du lieu ablation.",
        "",
        "## 6. Suy luan va latency",
        "",
        md(idf) if not idf.empty else "Chua co du lieu inference.",
        "",
        "## 7. Final test",
        "",
        md(fdf) if not fdf.empty else "Chua co predictions test. Chi chay test sau khi chot cau hinh tren val.",
        "",
        "## 8. Han che",
        "",
        "Neu chua chay du >=3 seed hoac chua du cac truc thi nghiem, can ghi ro trong bao cao truoc khi nop.",
    ]
    REPORT.write_text("\n".join(text), encoding="utf-8")


def main():
    df = summaries()
    bdf = backbones(df)
    tdf = training(df)
    idf = inference()
    fdf, pcdf = final_and_perclass()
    ldf = latency(idf)
    sdf = df.sort_values("val_macro_f1", ascending=False).head(10) if not df.empty else pd.DataFrame(columns=["exp_id"])
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        bdf.to_excel(writer, sheet_name="Backbones", index=False)
        tdf.to_excel(writer, sheet_name="Training", index=False)
        idf.to_excel(writer, sheet_name="Inference", index=False)
        fdf.to_excel(writer, sheet_name="Final", index=False)
        pcdf.to_excel(writer, sheet_name="PerClass", index=False)
        ldf.to_excel(writer, sheet_name="Latency", index=False)
        sdf.to_excel(writer, sheet_name="Summary", index=False)
    write_report(df, bdf, tdf, idf, fdf)
    print(f"wrote {OUT_XLSX}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
