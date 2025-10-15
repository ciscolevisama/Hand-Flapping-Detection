import os
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import importlib.util

# ==== 手动加载 segment_and_label.py，而不是普通 import ====
seg_path = os.path.join("src", "utils", "segment_and_label.py")
spec = importlib.util.spec_from_file_location("seg", seg_path)
seg = importlib.util.module_from_spec(spec)

# 提前注入假 THRESHOLDS，避免 import 报错
seg.THRESHOLDS = {}
seg.E_HIGH = 0.2
seg.E_LOW = -0.2
seg.AMPLITUDE_MIN = 5.0
seg.DELTA_A_THRESHOLD = 20.0
seg.PHASE_THRESHOLD = 45.0
seg.SINGLE_SIDE_RATIO = 0.7
seg.DELTA_A_THRESHOLD_LOW = 10.0

spec.loader.exec_module(seg)


def process_all(processed_dir, E_LOW, E_HIGH, window_size=12, step=6):
    counts = Counter()
    seg.E_LOW = E_LOW
    seg.E_HIGH = E_HIGH

    for file in tqdm(os.listdir(processed_dir)):
        if not file.endswith(".npy"):
            continue
        data = np.load(os.path.join(processed_dir, file))
        for start in range(0, len(data) - window_size + 1, step):
            window = data[start:start+window_size]
            label, conf, uncertain = seg.label_window(window, valid_ratio=1.0)
            counts[label] += 1
    return counts


def main(args):
    results = []
    grid = [(el, eh) for el in args.E_LOW for eh in args.E_HIGH]

    for E_LOW, E_HIGH in grid:
        print(f"\n▶️ Running with E_LOW={E_LOW}, E_HIGH={E_HIGH}")
        counts = process_all(args.processed_dir, E_LOW, E_HIGH)

        total = sum(counts.values())
        row = {"E_LOW": E_LOW, "E_HIGH": E_HIGH, "total": total}
        for cls in [
            "no_flap", "left_only_low", "left_only_high",
            "right_only_low", "right_only_high",
            "both_symmetric_low", "both_symmetric_high",
            "both_asymmetric"
        ]:
            row[cls] = counts.get(cls, 0)
        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "threshold_sweep_results.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n✅ Saved results to {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed_angles",
                        help="Directory with processed .npy files")
    parser.add_argument("--out_dir", type=str, default="data/analysis/threshold_sweep",
                        help="Output directory")
    parser.add_argument("--E_LOW", type=float, nargs="+", default=[-0.2, 0.0, 0.1],
                        help="List of E_LOW values to test")
    parser.add_argument("--E_HIGH", type=float, nargs="+", default=[0.05, 0.1, 0.2],
                        help="List of E_HIGH values to test")
    args = parser.parse_args()
    main(args)
