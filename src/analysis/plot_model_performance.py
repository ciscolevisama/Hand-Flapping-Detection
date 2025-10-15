"""
plot_model_performance_final.py
---------------------------------------------------
Generates Figure 4.1 for dissertation:
Grouped bar chart comparing macro Precision, Recall, and F1
for all ML and DL models (RF, SVM, CNN1D, BiLSTM, Hybrid CNN–LSTM).
---------------------------------------------------
Author: Wang Yida
Date: 2025-10
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================================
# 1. 数据定义（直接使用你实验统计的结果）
# ==========================================================
data = {
    "Model": ["RF", "SVM", "CNN1D", "BiLSTM", "Hybrid CNN–LSTM"],
    "Precision": [0.951, 0.764, 0.617, 0.511, 0.614],
    "Recall":    [0.937, 0.747, 0.595, 0.495, 0.593],
    "F1":        [0.943, 0.755, 0.606, 0.502, 0.603]
}
df = pd.DataFrame(data)

# ==========================================================
# 2. Matplotlib 样式设置（论文风格）
# ==========================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 300
})

x = np.arange(len(df["Model"]))
bar_width = 0.22
colors = ["#4C72B0", "#55A868", "#C44E52"]  # 论文标准配色

fig, ax = plt.subplots(figsize=(8, 4.5))

# ==========================================================
# 3. 绘制三组柱状图
# ==========================================================
ax.bar(x - bar_width, df["Precision"], bar_width, label="Precision", color=colors[0])
ax.bar(x,             df["Recall"],    bar_width, label="Recall",    color=colors[1])
ax.bar(x + bar_width, df["F1"],        bar_width, label="F1",        color=colors[2])

# 坐标轴与标签
ax.set_xticks(x)
ax.set_xticklabels(df["Model"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
ax.set_title("Comparison of Model Performance (Macro Precision, Recall, F1)", fontsize=12)

# ==========================================================
# 4. 添加图例（放顶部，避免重叠）
# ==========================================================
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=3,
    frameon=False
)

# ==========================================================
# 5. 添加数值标签（自动错位避免遮挡）
# ==========================================================
for i in range(len(df)):
    metrics = ["Precision", "Recall", "F1"]
    offsets = [-bar_width, 0, bar_width]
    for metric, offset in zip(metrics, offsets):
        val = df.iloc[i][metric]
        ax.text(x[i] + offset, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, rotation=0)

# ==========================================================
# 6. 保存结果（论文图用高分辨率 PNG + PDF）
# ==========================================================
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/Figure_4_1_performance.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/Figure_4_1_performance.pdf", bbox_inches="tight")

plt.show()

print("✅ Figure saved as: figures/Figure_4_1_performance.png / .pdf")
