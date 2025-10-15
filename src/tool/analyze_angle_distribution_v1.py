import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ANGLE_NAMES = [
    "left_wrist", "right_wrist",
    "left_elbow", "right_elbow",
    "left_armpit", "right_armpit"
]

def analyze_distribution(processed_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    all_data = {name: [] for name in ANGLE_NAMES}

    # 读取所有 .npy
    for file in os.listdir(processed_dir):
        if file.endswith(".npy"):
            path = os.path.join(processed_dir, file)
            features = np.load(path)  # (frames, 6)

            for i, name in enumerate(ANGLE_NAMES):
                all_data[name].extend(features[:, i])

    # 统计结果
    stats = {}
    for name, values in all_data.items():
        arr = np.array(values)
        stats[name] = {
            "mean": np.mean(arr),
            "std": np.std(arr),
            "p25": np.percentile(arr, 25),
            "p50": np.percentile(arr, 50),
            "p75": np.percentile(arr, 75),
            "p90": np.percentile(arr, 90)
        }

        # 绘制直方图
        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=50, color="skyblue", edgecolor="black")
        plt.title(f"Distribution of {name}")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}_hist.png"))
        plt.close()

    # 保存统计表
    df = pd.DataFrame(stats).T
    csv_path = os.path.join(save_dir, "angle_distribution_stats.csv")
    df.to_csv(csv_path)
    print(f"✅ Saved stats to {csv_path}")

    # === 自动阈值推荐 ===
    thresholds = {}
    # 高/低幅度（用 armpit）
    thresholds["AMPLITUDE_THRESHOLD"] = (df.loc["left_armpit", "p75"] +
                                         df.loc["right_armpit", "p75"]) / 2
    # 对称性（左右差异 ≤ 20 度左右）
    thresholds["SYMMETRY_THRESHOLD"] = 20  # 文献常用范围 15–20
    # 动作 vs 无动作（方差阈值）
    all_var = []
    for name, values in all_data.items():
        arr = np.array(values)
        all_var.append(np.var(arr))
    thresholds["MOVEMENT_THRESHOLD"] = np.mean(all_var) * 0.5  # 经验：均值的一半

    thr_path = os.path.join(save_dir, "recommended_thresholds.json")
    import json
    with open(thr_path, "w") as f:
        json.dump(thresholds, f, indent=4)

    print(f"✅ Recommended thresholds saved to {thr_path}")
    print("📊 Suggested values:")
    for k, v in thresholds.items():
        print(f"   {k}: {v:.2f}")

    return df, thresholds


if __name__ == "__main__":
    processed_dir = "../../data/processed_angles"
    save_dir = "../../data/analysis_v1"
    analyze_distribution(processed_dir, save_dir)
