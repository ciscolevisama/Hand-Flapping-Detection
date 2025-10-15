"""
分析 weak_labels.csv 的标签分布、样本数量、视频覆盖率。
用于验证重新打标签后的数据质量。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import Counter

def analyze_label_distribution(csv_path, outdir=None):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV 必须包含 'label' 列（或请检查列名是否匹配）")

    total = len(df)
    label_counts = df["label"].value_counts().sort_index()
    label_ratios = label_counts / total

    print("📊 Label distribution")
    print("=" * 40)
    for label, count in label_counts.items():
        ratio = label_ratios[label] * 100
        print(f"{label:<25} {count:>7} ({ratio:5.2f}%)")

    print("=" * 40)
    print(f"Total samples: {total}")
    print(f"Unique labels: {len(label_counts)}")

    # 统计每个视频的标签覆盖
    if "video" in df.columns:
        vids = df["video"].value_counts()
        print(f"\n🎥 Videos covered: {len(vids)}")
        print(f"Median samples per video: {int(np.median(vids))}")
        print(f"Min samples: {vids.min()}, Max: {vids.max()}")

    # 如果存在 start/end，可以检查段落长度
    if {"start", "end"}.issubset(df.columns):
        df["dur"] = df["end"] - df["start"]
        print(f"\n🕒 Segment duration mean={df['dur'].mean():.2f}, median={df['dur'].median():.2f}")

    # 可选导出结果
    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out_csv = outdir / "label_distribution_summary.csv"
        label_counts.to_csv(out_csv, header=["count"])
        print(f"\n✅ Summary saved to: {out_csv}")

    return label_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze weak_labels.csv distribution")
    parser.add_argument("--csv", type=str, required=True, help="Path to weak_labels.csv")
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory for summary CSV")
    args = parser.parse_args()

    analyze_label_distribution(args.csv, args.outdir)
