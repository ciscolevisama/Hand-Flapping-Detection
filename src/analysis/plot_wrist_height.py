import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

WINDOW_SIZE = 12  # keep consistent with extract_window_features.py

def plot_wrist_height(npy_path: str, out_dir: str = None):
    npy_path = Path(npy_path)
    if not npy_path.exists():
        print(f"❌ File not found: {npy_path}")
        return

    # load features
    data = np.load(npy_path)
    wr_l = data[:, 6]  # left wrist relative height
    wr_r = data[:, 7]  # right wrist relative height

    frames = np.arange(len(wr_l))

    # plot wrist curves
    plt.figure(figsize=(12, 5))
    plt.plot(frames, wr_l, label="Left wrist (relative)", color="blue", alpha=0.7)
    plt.plot(frames, wr_r, label="Right wrist (relative)", color="red", alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # add vertical lines for window boundaries
    for x in range(0, len(frames), WINDOW_SIZE):
        plt.axvline(x, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    # labels
    plt.xlabel("Frame")
    plt.ylabel("Relative height (shoulder=0, torso=1)")
    plt.title(f"Wrist relative height over time\n{npy_path.stem}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # save or show
    if out_dir:
        out_path = Path(out_dir) / f"{npy_path.stem}_wrist_plot.png"
        plt.savefig(out_path, dpi=150)
        print(f"✅ Plot with window boundaries saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/utils/plot_wrist_height.py data/processed_angles/<video>.npy [out_dir]")
    else:
        npy_file = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else None
        plot_wrist_height(npy_file, out_dir)
