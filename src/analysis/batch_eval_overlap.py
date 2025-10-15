# -*- coding: utf-8 -*-
"""
Batch evaluation of model predictions vs weak labels for all videos.
Outputs overlap ratio for each video and ranks them.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def encode_labels(series):
    unique = sorted(series.dropna().unique().tolist())
    mapping = {lab: i for i, lab in enumerate(unique)}
    return series.map(mapping), mapping


def estimate_overlap(df_labels, df_pred, seq_len=64):
    """Compute overlap ratio between weak labels and predictions for one video."""
    max_frame = max(df_labels["end_frame"].max(), len(df_pred) * seq_len)
    t = np.arange(0, max_frame)

    # 标签序列
    label_series = pd.Series(np.nan, index=t)
    for _, row in df_labels.iterrows():
        start, end = int(row.start_frame), int(row.end_frame)
        label_series[start:end] = row.label

    # 预测序列
    pred_series = pd.Series(np.nan, index=t)
    if "start_frame" in df_pred.columns and "end_frame" in df_pred.columns:
        for _, row in df_pred.iterrows():
            pred_series[int(row.start_frame):int(row.end_frame)] = row.pred_label
    else:
        for i, row in enumerate(df_pred.itertuples()):
            start = i * seq_len
            end = min(start + seq_len, len(t))
            pred_series[start:end] = row.pred_label

    # 编码
    label_encoded, _ = encode_labels(label_series)
    pred_encoded, _ = encode_labels(pred_series)

    # 重叠率
    n = min(len(label_encoded), len(pred_encoded))
    valid = (~label_encoded.isna()) & (~pred_encoded.isna())
    match = (label_encoded[valid] == pred_encoded[valid]).sum()
    overlap_ratio = match / valid.sum() if valid.sum() > 0 else 0
    return overlap_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_csv", default=str(ROOT / "data" / "labels" / "weak_labels.csv"))
    ap.add_argument("--pred_csv", default=str(ROOT / "runs_dl" / "hybrid_len64_clean_finetune" / "preds_ensemble.csv"))
    ap.add_argument("--topk", type=int, default=3, help="Show top-K videos by overlap ratio")
    args = ap.parse_args()

    df_labels = pd.read_csv(args.label_csv)
    df_preds = pd.read_csv(args.pred_csv)

    results = []
    for vid in df_labels["video"].unique():
        df_lab_vid = df_labels[df_labels["video"] == vid]
        df_pred_vid = df_preds[df_preds["video"].astype(str).str.contains(Path(vid).stem, na=False)]
        if len(df_pred_vid) == 0:
            continue
        overlap = estimate_overlap(df_lab_vid, df_pred_vid)
        results.append((vid, overlap))

    results.sort(key=lambda x: x[1], reverse=True)

    print("📊 Overlap ratios (sorted):")
    for vid, overlap in results:
        print(f"{vid:40s} {overlap*100:.2f}%")

    print(f"\n✅ Top-{args.topk} videos for demo:")
    for vid, overlap in results[:args.topk]:
        print(f"   {vid:40s} {overlap*100:.2f}%")


if __name__ == "__main__":
    main()
