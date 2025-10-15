import os
import numpy as np
import pandas as pd
from extract_window_features import process_file  # 直接复用你已有的函数

# Paths
processed_dir = "../../data/processed_angles"
out_dir = "../../data/window_features"
os.makedirs(out_dir, exist_ok=True)

def compare_modes(file, window_size=12, step=6, min_valid_ratio=0.5):
    rows_strict = process_file(file, window_size, step, min_valid_ratio, mode="strict", interpolate=False)
    rows_arm = process_file(file, window_size, step, min_valid_ratio, mode="arm_only", interpolate=True)

    return {
        "video": file.replace(".npy", ""),
        "strict_windows": len(rows_strict),
        "arm_only_windows": len(rows_arm)
    }


if __name__ == "__main__":
    results = []
    # 你可以改成只跑某一个文件，例如 files = ["flapping [7lgAK1z-Scs].npy"]
    files = [f for f in os.listdir(processed_dir) if f.endswith(".npy")]

    for f in files:
        print(f"▶️ Comparing modes for {f} ...")
        stats = compare_modes(f)
        results.append(stats)

    df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, "mode_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Comparison saved to {out_path}")
    print(df.head())
