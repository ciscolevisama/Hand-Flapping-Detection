import os
import numpy as np
import pandas as pd

# Paths
processed_dir = "../../data/processed_angles"
out_dir = "../../data/window_features"
os.makedirs(out_dir, exist_ok=True)

# Parameters
WINDOW_SIZE = 12
STEP = 6
MIN_VALID_RATIO = 0.8
  # require at least 50% valid frames per window


def interpolate_nans(arr):
    """Linearly interpolate NaNs in a 1D array."""
    n = len(arr)
    x = np.arange(n)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return arr  # all NaN, skip
    return np.interp(x, x[mask], arr[mask])


def compute_features(window):
    """
    Compute features for a window (shape = [W, 8])
    Columns: [ang_lwr, ang_rwr, ang_lel, ang_rel, ang_lax, ang_rax, wr_l_rel, wr_r_rel]
    """
    left_wrist = window[:, 0]
    right_wrist = window[:, 1]
    left_elbow = window[:, 2]
    right_elbow = window[:, 3]
    left_armpit = window[:, 4]
    right_armpit = window[:, 5]
    left_rel = window[:, 6]
    right_rel = window[:, 7]

    # Amplitude (peak-to-peak)
    A_left = np.ptp(left_wrist)
    A_right = np.ptp(right_wrist)

    # Frequency (zero-crossing approx) → take max of left/right
    def zero_crossings(sig):
        sig = sig - np.nanmean(sig)
        return len(np.where(np.diff(np.sign(sig)))[0])

    freq_left = zero_crossings(left_wrist)
    freq_right = zero_crossings(right_wrist)
    freq = max(freq_left, freq_right)

    # Left-right amplitude difference
    delta_A = abs(A_left - A_right)

    # Relative height (median)
    H_left = np.nanmedian(left_rel)
    H_right = np.nanmedian(right_rel)

    return {
        "A_left": A_left,
        "A_right": A_right,
        "freq": freq,
        "delta_A": delta_A,
        "H_left": H_left,
        "H_right": H_right,
    }


def process_file(file, window_size=WINDOW_SIZE, step=STEP,
                 min_valid_ratio=MIN_VALID_RATIO,
                 mode="strict", interpolate=True):
    """
    mode:
      - "strict": all 8 features must be valid
      - "arm_only": only arm/wrist + relative height must be valid
    """
    path = os.path.join(processed_dir, file)
    data = np.load(path)
    rows = []

    for start in range(0, len(data) - window_size + 1, step):
        window = data[start:start + window_size]

        # Interpolate NaNs (per column)
        if interpolate:
            window = np.apply_along_axis(interpolate_nans, 0, window)

        # Decide validity mask
        if mode == "strict":
            valid_mask = ~np.isnan(window).any(axis=1)
        elif mode == "arm_only":
            # only check wrist, elbow, armpit, relative heights (exclude torso stuff)
            arm_cols = [0, 1, 2, 3, 4, 5, 6, 7]
            valid_mask = ~np.isnan(window[:, arm_cols]).any(axis=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        valid_count = valid_mask.sum()
        valid_ratio = valid_count / window_size

        if valid_ratio < min_valid_ratio:
            continue  # skip this window

        feats = compute_features(window[valid_mask])
        feats["video"] = file.replace(".npy", "")
        feats["start"] = start
        feats["end"] = start + window_size
        feats["valid_ratio"] = valid_ratio
        rows.append(feats)

    return rows


if __name__ == "__main__":
    all_rows = []
    for file in os.listdir(processed_dir):
        if file.endswith(".npy"):
            print(f"▶️ Processing {file} ...")
            rows = process_file(file, mode="arm_only", interpolate=True)
            if not rows:
                print(f"   ⚠️ Skipped (no valid windows)")
                continue
            df = pd.DataFrame(rows)

            # Save per-video features
            out_path = os.path.join(out_dir, f"{file.replace('.npy','')}_features.csv")
            df.to_csv(out_path, index=False)
            print(f"   ✅ Saved {out_path} ({len(df)} windows)")

            all_rows.extend(rows)

    if all_rows:
        # Save combined features
        df_all = pd.DataFrame(all_rows)
        out_path = os.path.join(out_dir, "all_window_features.csv")
        df_all.to_csv(out_path, index=False)
        print(f"\n✅ All features saved to {out_path}")
    else:
        print("\n❌ No valid windows found in any video.")
