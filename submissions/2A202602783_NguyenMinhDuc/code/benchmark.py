"""Latency benchmarking helpers."""
from __future__ import annotations

import time

import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "n": int(iters),
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    dtype = dtype.lower()
    autocast_enabled = False
    if dtype == "fp16":
        if dev.type != "cuda":
            raise ValueError("fp16 latency is intended for CUDA")
        model = model.half()
        x = x.half()
    elif dtype == "amp":
        autocast_enabled = dev.type == "cuda"
    elif dtype != "fp32":
        raise ValueError(f"Unknown dtype: {dtype}")

    def fn():
        with torch.inference_mode():
            if autocast_enabled:
                with torch.cuda.amp.autocast():
                    model(x)
            else:
                model(x)

    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    stats = bench(fn, warmup=warmup, iters=iters, sync=sync)
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    stats.update({
        "gpu": gpu,
        "dtype": dtype,
        "batch": int(batch_size),
        "img_size": int(img_size),
        "images_per_s": float(batch_size / (stats["p50"] / 1000.0)),
        "torch": torch.__version__,
        "device": str(dev),
    })
    return stats


def tta_latency(model, k_views: int, **kw) -> dict:
    base = latency_report(model, **kw)
    out = dict(base)
    for key in ("p50", "p95", "p99", "mean"):
        out[key] = out[key] * k_views
    out["k_views"] = int(k_views)
    out["images_per_s"] = out["batch"] / (out["p50"] / 1000.0)
    return out
