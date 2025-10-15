import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_wrist_height(npy_file, out_dir="analysis/debug_plots"):
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npy_file)
    # ✅ 现在 H 在第 6 / 7 列
    H_left = data[:, 6]
    H_right = data[:, 7]

    plt.figure(figsize=(12, 5))
    plt.plot(H_left, label="H_left (col6)", alpha=0.7)
    plt.plot(H_right, label="H_right (col7)", alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", label="shoulder line (0)")
    plt.title(f"Wrist height trace: {os.path.basename(npy_file)}")
    plt.xlabel("Frame index")
    plt.ylabel("Relative height (normalised, shoulder=0)")
    plt.legend()
    out_path = os.path.join(out_dir, os.path.basename(npy_file).replace(".npy", "_heights.png"))
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_file", type=str, required=True,
                        help="Path to one .npy file in data/processed_angles/")
    parser.add_argument("--out_dir", type=str, default="data/analysis/debug_heights",
                        help="Directory to save output plots")
    args = parser.parse_args()

    plot_wrist_height(args.npy_file, args.out_dir)
