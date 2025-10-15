# -*- coding: utf-8 -*-
"""
Compare DL model performances (CNN / LSTM / Hybrid) from ensemble_report.json.

Usage:
    python src/analysis/compare_dl_models.py
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs_dl"

# 要比较的模型及对应目录
MODELS = {
    "CNN": RUNS_DIR / "cnn_len64_clean_finetune" / "ensemble_report.json",
    "LSTM": RUNS_DIR / "lstm_len64_clean_finetune" / "ensemble_report.json",
    "Hybrid CNN-LSTM": RUNS_DIR / "hybrid_len64_clean_finetune" / "ensemble_report.json",
}

def load_metrics(json_path: Path):
    if not json_path.exists():
        print(f"⚠️ File not found: {json_path}")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    overall = data.get("overall", {})
    return {
        "f1_macro": overall.get("f1_macro", 0),
        "highF1": overall.get("highF1", 0),
        "acc": overall.get("acc", 0),
    }

def main():
    metrics = {}
    for name, path in MODELS.items():
        res = load_metrics(path)
        if res:
            metrics[name] = res

    if not metrics:
        print("❌ No valid ensemble_report.json found.")
        return

    print("✅ Loaded metrics for:", ", ".join(metrics.keys()))

    # 准备数据
    models = list(metrics.keys())
    f1_macro = [metrics[m]["f1_macro"] for m in models]
    highF1 = [metrics[m]["highF1"] for m in models]
    acc = [metrics[m]["acc"] for m in models]

    # 绘制柱状图
    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, f1_macro, width, label="F1_macro")
    plt.bar(x, highF1, width, label="HighF1")
    plt.bar(x + width, acc, width, label="Accuracy")

    plt.xticks(x, models, fontsize=11)
    plt.ylim(0, 1)
    plt.ylabel("Score", fontsize=12)
    plt.title("Comparison of Deep Learning Models", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # 输出路径
    out_path = RUNS_DIR / "compare_dl_models.png"
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"✅ Comparison plot saved to: {out_path}")

if __name__ == "__main__":
    main()
