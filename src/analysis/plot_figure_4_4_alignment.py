"""
plot_figure_4_4_alignment.py
Final revision: Fixed overlapping between subplot titles and x-axis labels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import argparse
import os

# ---------------------------
# Parse command-line arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Plot alignment between weak labels and model predictions.")
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--pred_csv", type=str, required=True)
parser.add_argument("--rf_csv", type=str, required=True)
parser.add_argument("--output_path", type=str, default="figures/Figure_4_4_alignment.png")
args = parser.parse_args()

video_id = os.path.splitext(os.path.basename(args.video))[0]
label_csv = f"data/labels/{video_id}_labels.csv"

# ---------------------------
# Load data
# ---------------------------
df_labels = pd.read_csv(label_csv)
df_rf = pd.read_csv(args.rf_csv)
df_hyb = pd.read_csv(args.pred_csv)

df_labels = df_labels[df_labels['video'] == video_id] if 'video' in df_labels.columns else df_labels
df_rf = df_rf.iloc[:len(df_labels)]
df_hyb = df_hyb.iloc[:len(df_labels)]

# ---------------------------
# Define class order and colours
# ---------------------------
classes = [
    "no_flap", "left_only_low", "left_only_high",
    "right_only_low", "right_only_high",
    "both_symmetric_low", "both_symmetric_high", "both_asymmetric"
]
colors = plt.cm.tab10.colors[:len(classes)]
color_map = dict(zip(classes, colors))

# ---------------------------
# Helper: label → index
# ---------------------------
def map_to_index(series):
    return [classes.index(x) if x in classes else -1 for x in series]

y_label = map_to_index(df_labels['label'])
y_rf = map_to_index(df_rf['pred_label'])
y_hyb = map_to_index(df_hyb['pred_label'])

# ---------------------------
# Plot settings
# ---------------------------
plt.rc('font', family='Times New Roman', size=12)
plt.rc('axes', titlesize=12, labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)

cm = mcolors.ListedColormap(colors)
fig, axes = plt.subplots(3, 1, figsize=(12, 6))  # 稍微增高一点
plt.subplots_adjust(bottom=0.25, top=0.95, hspace=0.55)  # ✅ 加大垂直间距 hspace=0.55

# ---------------------------
# Subplots
# ---------------------------
axes[0].imshow([y_label], aspect='auto', cmap=cm)
axes[0].set_yticks([])
axes[0].set_title("(a)  Ground-truth weak labels", fontweight='bold', pad=10)

axes[1].imshow([y_rf], aspect='auto', cmap=cm)
axes[1].set_yticks([])
axes[1].set_title("(b)  Random Forest predictions", fontweight='bold', pad=10)

axes[2].imshow([y_hyb], aspect='auto', cmap=cm)
axes[2].set_yticks([])
axes[2].set_title("(c)  Hybrid CNN–LSTM predictions", fontweight='bold', pad=12)  # ✅ 多加一点 pad
axes[2].set_xlabel("Frame index (time)", fontweight='bold', labelpad=10)

# ---------------------------
# Legend
# ---------------------------
patches = [mpatches.Patch(color=color_map[c], label=c) for c in classes]
fig.legend(
    handles=patches,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.1),
    frameon=False
)

# ---------------------------
# Save
# ---------------------------
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
plt.savefig(args.output_path, dpi=300)
print(f"✅ Saved: {args.output_path}")
plt.show()
