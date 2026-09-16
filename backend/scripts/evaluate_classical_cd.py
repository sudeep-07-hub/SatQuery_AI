"""
Evaluate the classical change detector against S1-AAD ground-truth change masks.

Usage (from backend/):
    python3 scripts/evaluate_classical_cd.py [--limit N]

Writes data/reports/classical_cd_s1aad_eval.json
"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raster_io import load_raster  # noqa: E402
from mc4b_temporal.classical import detect_change  # noqa: E402

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "S1-AAD Sentinel-1 Amazon Airstrip Dataset",
    "S1-AAD Sentinel-1 Amazon Airstrip Dataset",
    "Change_detection",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    names = sorted(f for f in os.listdir(os.path.join(DATASET, "before")) if f.endswith(".tif"))
    if args.limit:
        names = names[: args.limit]

    tp = fp = fn = 0
    per_pair = []
    for name in names:
        before = load_raster(os.path.join(DATASET, "before", name), "image_1")
        after = load_raster(os.path.join(DATASET, "after", name), "image_2")
        with rasterio.open(os.path.join(DATASET, "mask", name)) as src:
            gt = src.read(1) > 0
        pred = detect_change(before, after, "sar")["mask"]
        if pred.shape != gt.shape:
            continue
        p_tp = int(np.sum(pred & gt))
        p_fp = int(np.sum(pred & ~gt))
        p_fn = int(np.sum(~pred & gt))
        tp, fp, fn = tp + p_tp, fp + p_fp, fn + p_fn
        union = p_tp + p_fp + p_fn
        per_pair.append({
            "pair": name,
            "gt_changed_px": int(gt.sum()),
            "pred_changed_px": int(pred.sum()),
            "iou": round(p_tp / union, 4) if union else 1.0,
        })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    report = {
        "dataset": "S1-AAD Change_detection (Sentinel-1, before/after/mask)",
        "detector": "classical SAR log-ratio, threshold = max(Otsu, 3 dB), 5x5 smoothing, 3x3 opening, min region 20 px",
        "pairs_evaluated": len(per_pair),
        "pixel_micro": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "iou": round(iou, 4),
        },
        "mean_pair_iou": round(float(np.mean([p["iou"] for p in per_pair])), 4) if per_pair else 0.0,
        "per_pair": per_pair,
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports", "classical_cd_s1aad_eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_pair"}, indent=2))


if __name__ == "__main__":
    main()
