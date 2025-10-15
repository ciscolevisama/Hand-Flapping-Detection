import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# === 路径配置 ===
video_id = "My nephew flapping his arms [UxpQpWzhOx0]"   # 👈 改成你要检查的视频 ID（不带扩展名）
angles_file = f"../../data/processed_angles/{video_id}.npy"
labels_file = f"../../data/labels/{video_id}_labels.csv"
out_dir = "../../data/analysis/qc_plots"
os.makedirs(out_dir, exist_ok=True)

# === 标签颜色映射 ===
label_colors = {
    "no_flap": "lightgrey",
    "left_only_low": "blue",
    "left_only_high": "royalblue",
    "right_only_low": "orange",
    "right_only_high": "darkorange",
    "both_symmetric_low": "green",
    "both_symmetric_high": "limegreen",
    "both_asymmetric": "red",
}

def plot_qc(angles_file, labels_file, out_dir):
    angles = np.load(angles_file)
    labels = pd.read_csv(labels_file)
    frames = np.arange(len(angles))

    left_h = angles[:, 6]
    right_h = angles[:, 7]

    plt.figure(figsize=(15, 6))
    plt.plot(frames, left_h, label="Left wrist height", color="blue")
    plt.plot(frames, right_h, label="Right wrist height", color="orange")

    # 叠加窗口标签（背景色）
    for _, row in labels.iterrows():
        start, end, label = row["start_frame"], row["end_frame"], row["label"]
        color = label_colors.get(label, "pink")
        plt.axvspan(start, end, color=color, alpha=0.2)

    # 图例（颜色 → 标签）
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.3) for c in label_colors.values()]
    plt.legend(handles, label_colors.keys(), title="Labels", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.title(f"QC Plot for {Path(angles_file).stem}")
    plt.xlabel("Frame")
    plt.ylabel("Relative Wrist Height")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{Path(angles_file).stem}_qc.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved QC plot to {out_path}")

if __name__ == "__main__":
    plot_qc(angles_file, labels_file, out_dir)