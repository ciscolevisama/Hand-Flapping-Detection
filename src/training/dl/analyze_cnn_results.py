#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse CNN results across three labelling modes:
(full / clean_only / conf_weighted)

Outputs:
- runs_dl/cnn_summary.csv              ← overall macro-F1 & accuracy summary
- runs_dl/cnn_classwise_f1.csv         ← per-class F1 summary
- runs_dl/figures/cnn_summary_bar.png  ← overall performance bar chart
- runs_dl/figures/cnn_classwise_f1.png ← per-class F1 comparison chart
"""
import os, json, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Config
# -----------------------------
RESULT_ROOT = "runs_dl"
FIG_DIR = os.path.join(RESULT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------
# 1. Overall summary
# -----------------------------
rows=[]
for path in glob.glob(f"{RESULT_ROOT}/cnn_*/metrics_cnn.csv"):
    mode = os.path.basename(os.path.dirname(path)).replace("cnn_","")
    df = pd.read_csv(path)
    rows.append({
        "mode": mode,
        "f1_macro_mean": df["f1_macro"].mean(),
        "f1_macro_std": df["f1_macro"].std(),
        "acc_mean": df["accuracy"].mean(),
        "acc_std": df["accuracy"].std()
    })
summary = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
summary.to_csv(f"{RESULT_ROOT}/cnn_summary.csv", index=False)
print("✅ Saved overall summary -> runs_dl/cnn_summary.csv")
print(summary.round(4))

# -----------------------------
# 2. Classwise F1 summary
# -----------------------------
rows=[]
for path in glob.glob(f"{RESULT_ROOT}/cnn_*/report_overall.json"):
    mode = os.path.basename(os.path.dirname(path)).replace("cnn_","")
    with open(path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    for cls, vals in rep.items():
        if cls in ("accuracy","macro avg","weighted avg"): continue
        rows.append({"mode": mode, "class": cls, **vals})
df = pd.DataFrame(rows)
pivot = df.pivot_table(index="class", columns="mode", values="f1-score")
pivot.to_csv(f"{RESULT_ROOT}/cnn_classwise_f1.csv")
print("✅ Saved classwise F1 -> runs_dl/cnn_classwise_f1.csv")

# -----------------------------
# 3. Plot overall summary bar
# -----------------------------
plt.figure(figsize=(7,4))
x = np.arange(len(summary))
plt.bar(x-0.2, summary["f1_macro_mean"], width=0.4, label="Macro-F1")
plt.bar(x+0.2, summary["acc_mean"], width=0.4, label="Accuracy")
plt.xticks(x, summary["mode"], rotation=15)
plt.ylabel("Score")
plt.title("CNN Overall Performance (5-fold grouped CV)")
for i,v in enumerate(summary["f1_macro_mean"]):
    plt.text(i-0.25, v+0.005, f"{v:.3f}", fontsize=9)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cnn_summary_bar.png", dpi=300)
plt.close()
print("✅ Saved figure -> runs_dl/figures/cnn_summary_bar.png")

# -----------------------------
# 4. Plot classwise F1 bar chart
# -----------------------------
pivot = pivot.sort_index()
modes = pivot.columns.tolist()
plt.figure(figsize=(10,6))
bar_width = 0.25
x = np.arange(len(pivot))
for i,mode in enumerate(modes):
    plt.bar(x + (i-len(modes)/2)*bar_width, pivot[mode], width=bar_width, label=mode)
plt.xticks(x, pivot.index, rotation=45, ha="right")
plt.ylabel("F1-score")
plt.title("CNN Classwise F1 Comparison across Labelling Modes")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cnn_classwise_f1.png", dpi=300)
plt.close()
print("✅ Saved figure -> runs_dl/figures/cnn_classwise_f1.png")

print("\n🎯 All analysis complete. Check 'runs_dl/figures/' for figures.")
