# -*- coding: utf-8 -*-
"""
Window labelling with double-threshold height logic, using upper-quantile H
for high/low decisions and amplitude-trigger inside the fuzzy band.

Features per frame (from extract_angles.py):
    0: ang_lwr, 1: ang_rwr, 2: ang_lel, 3: ang_rel,
    4: ang_lax, 5: ang_rax, 6: wr_l_rel, 7: wr_r_rel

Height semantics:
    shoulder line ≈ 0; above-shoulder is positive; below-shoulder is negative.

Single-hand logic (fuzzy band with amplitude trigger):
    if H_q75 >= E_HIGH  -> *_high, uncertain=False
    elif H_q75 <= E_LOW -> *_low,  uncertain=False
    else (E_LOW < H_q75 < E_HIGH):
        if amplitude >= STRONG_AMP -> *_high, uncertain=True
        else -> *_low, uncertain=True

Two-hand logic (double threshold on q75):
    both >= E_HIGH -> both_symmetric_high / both_asymmetric (by deltaA)
    both <= E_LOW  -> both_symmetric_low  / both_asymmetric (by deltaA)
    one high one low -> both_asymmetric
    otherwise (in fuzzy band) -> both_asymmetric + uncertain=True

Note:
- When imported, this module does NOT read any JSON. Thresholds JSON is only
  read when the script is executed as __main__ (CLI).
"""

import os
import json
import argparse
import hashlib
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# Global thresholds (overridable)
# =========================
THRESHOLDS = {}
AMPLITUDE_MIN = 5.0
DELTA_A_THRESHOLD = 20.0
DELTA_A_THRESHOLD_LOW = 10.0
PHASE_THRESHOLD = 45.0
SINGLE_SIDE_RATIO = 0.7

# Height thresholds (double-threshold scheme)
E_HIGH = 0.20
E_LOW  = 0.00

# Column indices (keep aligned with extract_angles.py)
IDX_ANG_LWR = 0
IDX_ANG_RWR = 1
IDX_H_LEFT  = 6   # wrist height (left, relative; shoulder=0; above positive)
IDX_H_RIGHT = 7   # wrist height (right, relative)

# Quantile for high/low decision within a window
H_DECISION_Q = 80  # use q75 (change to 80 if you want more high-friendly)


# =========================
# Helpers
# =========================
def _nanptp(x: np.ndarray) -> float:
    """Peak-to-peak ignoring NaNs."""
    if x.size == 0 or np.all(np.isnan(x)):
        return 0.0
    return float(np.nanmax(x) - np.nanmin(x))


def _nanpercentile(x: np.ndarray, q: float, fallback="median") -> float:
    """Robust percentile on possibly-NaN data with a sensible fallback."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    try:
        return float(np.nanpercentile(x, q))
    except Exception:
        if fallback == "median":
            return float(np.nanmedian(x))
        return float(np.nanmean(x))


def _strong_amp() -> float:
    """
    Strong amplitude threshold used to decide HIGH inside the fuzzy band.
    Priority: THRESHOLDS['AMPLITUDE_THRESHOLD'] -> fallback: max(AMPLITUDE_MIN*6, 35).
    """
    thr = None
    try:
        if isinstance(THRESHOLDS, dict) and "AMPLITUDE_THRESHOLD" in THRESHOLDS:
            thr = float(THRESHOLDS["AMPLITUDE_THRESHOLD"])
    except Exception:
        thr = None
    if thr is not None and np.isfinite(thr):
        return thr
    return max(AMPLITUDE_MIN * 6.0, 35.0)


def _effective_path(default_rel: str) -> Path:
    """
    Resolve a default path robustly:
    - First try CWD + default_rel
    - If not exists, try project root (…/src/utils/ -> parents[2]) + clean relative
    """
    p = Path(default_rel)
    if p.exists():
        return p
    root = Path(__file__).resolve().parents[2]
    clean_rel = default_rel.replace(".."+os.sep, "").replace("../", "")
    q = root / clean_rel
    return q


# =========================
# Core labelling
# =========================
def label_window(window: np.ndarray, valid_ratio: float = 1.0):
    """
    Label a 12-frame window with confidence and 'uncertain' flag.
    Uses q75 of H for threshold comparisons (more sensitive to raised-hand peaks).
    """
    # Signals
    left  = window[:, IDX_ANG_LWR]
    right = window[:, IDX_ANG_RWR]

    # Heights: use upper quantile for the decision; keep median for confidence shaping if needed
    H_left_q  = _nanpercentile(window[:, IDX_H_LEFT],  H_DECISION_Q)
    H_right_q = _nanpercentile(window[:, IDX_H_RIGHT], H_DECISION_Q)
    H_left_med  = float(np.nanmedian(window[:, IDX_H_LEFT]))
    H_right_med = float(np.nanmedian(window[:, IDX_H_RIGHT]))

    # Amplitudes
    A_left  = _nanptp(left)
    A_right = _nanptp(right)
    L_active = A_left  >= AMPLITUDE_MIN
    R_active = A_right >= AMPLITUDE_MIN

    strong_amp = _strong_amp()

    # Movement strength component for confidence
    max_amp = max(A_left, A_right)
    conf_amp = abs(max_amp - AMPLITUDE_MIN) / (abs(max_amp - AMPLITUDE_MIN) + 5.0)

    # Completely still
    if not L_active and not R_active:
        return "no_flap", round(valid_ratio, 2), False

    # ---------- Single-hand: left active ----------
    if L_active and not R_active:
        if H_left_q >= E_HIGH:
            label, uncertain, ref = "left_only_high", False, E_HIGH
        elif H_left_q <= E_LOW:
            label, uncertain, ref = "left_only_low",  False, E_LOW
        else:
            # Fuzzy band -> amplitude trigger (high-friendly if amplitude is strong)
            if A_left >= strong_amp:
                label, ref = "left_only_high", E_HIGH
            else:
                label, ref = "left_only_low",  E_LOW
            uncertain = True

        # confidence: combine validity, amplitude, and positional distance to the chosen ref
        # (use median height in the distance to reduce sensitivity to spikes)
        conf_pos = abs(H_left_med - ref) / (abs(H_left_med - ref) + 0.5)
        confidence = 0.4*valid_ratio + 0.4*conf_amp + 0.2*conf_pos
        return label, round(confidence, 2), uncertain

    # ---------- Single-hand: right active ----------
    if R_active and not L_active:
        if H_right_q >= E_HIGH:
            label, uncertain, ref = "right_only_high", False, E_HIGH
        elif H_right_q <= E_LOW:
            label, uncertain, ref = "right_only_low",  False, E_LOW
        else:
            if A_right >= strong_amp:
                label, ref = "right_only_high", E_HIGH
            else:
                label, ref = "right_only_low",  E_LOW
            uncertain = True

        conf_pos = abs(H_right_med - ref) / (abs(H_right_med - ref) + 0.5)
        confidence = 0.4*valid_ratio + 0.4*conf_amp + 0.2*conf_pos
        return label, round(confidence, 2), uncertain

    # ---------- Two-hand ----------
    deltaA = abs(A_left - A_right)

    # one-high-one-low (by q75) -> clearly asymmetric
    if (H_left_q >= E_HIGH and H_right_q <= E_LOW) or (H_right_q >= E_HIGH and H_left_q <= E_LOW):
        conf_sym = deltaA / (deltaA + 5.0)
        confidence = 0.3*valid_ratio + 0.4*conf_amp + 0.3*conf_sym
        return "both_asymmetric", round(confidence, 2), False

    # both high (by q75)
    if H_left_q >= E_HIGH and H_right_q >= E_HIGH:
        if deltaA <= DELTA_A_THRESHOLD:
            label = "both_symmetric_high"
            conf_sym = 1 - deltaA / (deltaA + 5.0)
        else:
            label = "both_asymmetric"
            conf_sym = deltaA / (deltaA + 5.0)
        confidence = 0.3*valid_ratio + 0.4*conf_amp + 0.3*conf_sym
        return label, round(confidence, 2), False

    # both low (by q75)
    if H_left_q <= E_LOW and H_right_q <= E_LOW:
        if deltaA <= DELTA_A_THRESHOLD_LOW:
            label = "both_symmetric_low"
            conf_sym = 1 - deltaA / (deltaA + 5.0)
        else:
            label = "both_asymmetric"
            conf_sym = deltaA / (deltaA + 5.0)
        confidence = 0.3*valid_ratio + 0.4*conf_amp + 0.3*conf_sym
        return label, round(confidence, 2), False

    # fuzzy band (both hands inside or mixed around the band) -> asymmetric + uncertain
    conf_sym = deltaA / (deltaA + 5.0)
    confidence = 0.4*valid_ratio + 0.3*conf_amp + 0.3*conf_sym
    return "both_asymmetric", round(confidence, 2), True


def process_file(file: str, processed_dir: str, window_size: int = 12, step: int = 6):
    """
    Sliding-window labelling for a single .npy file.
    Computes a simple valid_ratio (share of frames with finite key columns).
    """
    arr = np.load(os.path.join(processed_dir, file))
    rows = []
    N = len(arr)
    cols_check = [IDX_ANG_LWR, IDX_ANG_RWR, IDX_H_LEFT, IDX_H_RIGHT]

    for start in range(0, N - window_size + 1, step):
        window = arr[start:start+window_size]
        finite_frames = np.isfinite(window[:, cols_check]).all(axis=1)
        valid_ratio = float(finite_frames.mean())
        label, conf, uncertain = label_window(window, valid_ratio=valid_ratio)
        rows.append([file.replace(".npy", ""), start, start+window_size, label, conf, int(uncertain)])

    return rows


# =========================
# CLI entry (only runs when __main__)
# =========================
def _main():
    parser = argparse.ArgumentParser(
        description="Window labelling with upper-quantile height decision and amplitude-trigger in fuzzy band."
    )
    parser.add_argument("--processed_dir", type=str, default=None,
                        help="Dir of processed .npy (default: <repo>/data/processed_angles)")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="Output dir for labels (default: <repo>/data/labels)")
    parser.add_argument("--thr_json", type=str, default=None,
                        help="Threshold JSON (default: <repo>/data/analysis/recommended_thresholds_window.json)")
    parser.add_argument("--branch", type=str, default="calibrated", choices=["raw","corrected","calibrated"],
                        help="Which threshold set to use from JSON")
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--step", type=int, default=6)
    # Optional overrides
    parser.add_argument("--E_HIGH", type=float, default=None)
    parser.add_argument("--E_LOW",  type=float, default=None)
    parser.add_argument("--AMPLITUDE_MIN", type=float, default=None)
    args = parser.parse_args()

    # Resolve default paths robustly
    processed_dir = Path(args.processed_dir) if args.processed_dir else _effective_path("data/processed_angles")
    label_dir     = Path(args.label_dir)     if args.label_dir     else _effective_path("data/labels")
    thr_path      = Path(args.thr_json)      if args.thr_json      else _effective_path("data/analysis/recommended_thresholds_window.json")

    label_dir.mkdir(parents=True, exist_ok=True)
    if not thr_path.exists():
        raise FileNotFoundError(f"❌ Threshold file not found: {thr_path}")

    with open(thr_path, "rb") as f:
        raw_bytes = f.read()
    data = json.loads(raw_bytes)

    global THRESHOLDS, AMPLITUDE_MIN, DELTA_A_THRESHOLD, DELTA_A_THRESHOLD_LOW
    global PHASE_THRESHOLD, SINGLE_SIDE_RATIO, E_HIGH, E_LOW

    if args.branch in data:
        THRESHOLDS = data[args.branch]
        chosen_branch = args.branch
    elif "calibrated" in data:
        THRESHOLDS = data["calibrated"]
        chosen_branch = "calibrated"
    elif "corrected" in data:
        THRESHOLDS = data["corrected"]
        chosen_branch = "corrected"
    else:
        THRESHOLDS = data
        chosen_branch = "raw-or-flat"

    # Apply to globals
    AMPLITUDE_MIN          = THRESHOLDS.get("AMPLITUDE_MIN", AMPLITUDE_MIN)
    DELTA_A_THRESHOLD      = THRESHOLDS.get("DELTA_A_THRESHOLD", DELTA_A_THRESHOLD)
    DELTA_A_THRESHOLD_LOW  = THRESHOLDS.get("DELTA_A_THRESHOLD_LOW", DELTA_A_THRESHOLD_LOW)
    PHASE_THRESHOLD        = THRESHOLDS.get("PHASE_THRESHOLD", PHASE_THRESHOLD)
    SINGLE_SIDE_RATIO      = THRESHOLDS.get("SINGLE_SIDE_RATIO", SINGLE_SIDE_RATIO)
    E_HIGH                 = THRESHOLDS.get("E_HIGH", E_HIGH)
    E_LOW                  = THRESHOLDS.get("E_LOW",  E_LOW)

    # CLI overrides
    if args.E_HIGH is not None: E_HIGH = args.E_HIGH
    if args.E_LOW  is not None: E_LOW  = args.E_LOW
    if args.AMPLITUDE_MIN is not None: AMPLITUDE_MIN = args.AMPLITUDE_MIN

    print(f"[CWD] {os.getcwd()}")
    print(f"[THR_FILE] {thr_path} (md5={hashlib.md5(raw_bytes).hexdigest()})")
    print(f"[THR_BRANCH] {chosen_branch}")
    print("🔎 Effective thresholds:")
    print(f"   AMPLITUDE_MIN={AMPLITUDE_MIN:.3f}, DELTA_A_THRESHOLD={DELTA_A_THRESHOLD:.3f}, "
          f"DELTA_A_THRESHOLD_LOW={DELTA_A_THRESHOLD_LOW:.3f}")
    print(f"   E_HIGH={E_HIGH:.3f}, E_LOW={E_LOW:.3f}")
    print(f"   STRONG_AMP={_strong_amp():.3f} (for fuzzy-band high trigger)")
    print(f"   H decision quantile = q{H_DECISION_Q}")

    all_rows = []
    for file in sorted(os.listdir(processed_dir)):
        if not file.endswith(".npy"):
            continue
        print(f"▶️ Processing {file} ...")
        rows = process_file(file, str(processed_dir), window_size=args.window_size, step=args.step)
        all_rows.extend(rows)

        per_video_csv = label_dir / f"{file.replace('.npy','')}_labels.csv"
        pd.DataFrame(rows, columns=["video","start_frame","end_frame","label","confidence","uncertain"]).to_csv(per_video_csv, index=False)
        print(f"   ✅ Saved {per_video_csv}")

    df = pd.DataFrame(all_rows, columns=["video","start_frame","end_frame","label","confidence","uncertain"])
    agg_csv = label_dir / "weak_labels.csv"
    df.to_csv(agg_csv, index=False)
    print(f"✅ Saved aggregated labels to {agg_csv}")

    meta = {
        "datetime": datetime.datetime.now().isoformat(),
        "thr_file": str(thr_path),
        "thr_md5": hashlib.md5(raw_bytes).hexdigest(),
        "thr_branch": chosen_branch,
        "effective": {
            "AMPLITUDE_MIN": AMPLITUDE_MIN,
            "DELTA_A_THRESHOLD": DELTA_A_THRESHOLD,
            "DELTA_A_THRESHOLD_LOW": DELTA_A_THRESHOLD_LOW,
            "E_HIGH": E_HIGH,
            "E_LOW": E_LOW,
            "STRONG_AMP": _strong_amp(),
            "H_DECISION_Q": H_DECISION_Q
        },
        "columns": {
            "0":"ang_lwr","1":"ang_rwr","2":"ang_lel","3":"ang_rel",
            "4":"ang_lax","5":"ang_rax","6":"wr_l_rel","7":"wr_r_rel"
        }
    }
    with open(label_dir / "labelling_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    _main()
