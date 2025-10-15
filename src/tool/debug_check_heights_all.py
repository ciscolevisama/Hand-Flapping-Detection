import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

def plot_wrist_height(npy_file, out_dir):
    """画单个文件的左右手高度曲线"""
    data = np.load(npy_file)
    # 检查列数是否够
    if data.shape[1] < 8:
        print(f"⚠️ {os.path.basename(npy_file)} 列数={data.shape[1]}，跳过（不是新格式）")
        return False

    H_left = data[:, 6]  # ✅ 第6列
    H_right = data[:, 7] # ✅ 第7列

    plt.figure(figsize=(12, 5))
    plt.plot(H_left, label="H_left", alpha=0.7)
    plt.plot(H_right, label="H_right", alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", label="shoulder line (0)")
    plt.title(f"Wrist height trace\n{os.path.basename(npy_file)}")
    plt.xlabel("Frame index")
    plt.ylabel("Relative height (shoulder=0)")
    plt.legend()
    plt.tight_layout()

    out_path = Path(out_dir) / (Path(npy_file).stem + "_heights.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main(args):
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted([p for p in in_dir.glob("*.npy")])
    print(f"🧩 Found {len(npy_files)} .npy files in {in_dir}")

    ok_count = 0
    for file in tqdm(npy_files, desc="Plotting wrist heights"):
        if plot_wrist_height(file, out_dir):
            ok_count += 1

    print(f"✅ Finished. Plots saved to {out_dir} ({ok_count}/{len(npy_files)} succeeded).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/processed_angles",
                        help="输入 .npy 文件所在文件夹")
    parser.add_argument("--out_dir", type=str, default="data/analysis/debug_heights_all",
                        help="输出图片保存文件夹")
    args = parser.parse_args()
    main(args)
