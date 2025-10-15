import os
import json
import numpy as np
import pandas as pd

IN_CSV = "../../data/window_features/all_window_features.csv"
OUT_JSON = "../../data/analysis/recommended_thresholds_window.json"


def otsu_threshold(x, nbins=128, rng=None):
    """一维 Otsu，返回阈值；若失败返回 None。"""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 100:
        return None
    if rng is None:
        rng = (np.nanpercentile(x, 1), np.nanpercentile(x, 99))
    lo, hi = float(rng[0]), float(rng[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    hist, edges = np.histogram(np.clip(x, lo, hi), bins=nbins, range=(lo, hi))
    w = hist.astype(float)
    p = w / w.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2.0)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    k = np.nanargmax(sigma_b2)
    thr = (edges[k] + edges[k + 1]) / 2.0
    return float(thr)


def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"❌ Not found: {IN_CSV}. Run extract_window_features.py first.")
    df = pd.read_csv(IN_CSV)

    # 读取或默认阈值（用于幅度、对称性等）
    stats = {}
    stats["AMPLITUDE_MIN"] = float(np.nanpercentile(np.r_[df["A_left"], df["A_right"]], 25))
    stats["DELTA_A_THRESHOLD"] = float(np.nanpercentile(df["delta_A"], 75))
    stats["PHASE_THRESHOLD"] = 45.0
    stats["SINGLE_SIDE_RATIO"] = 0.7

    # 活动窗口（避免静止帧干扰）
    active = df[(df["A_left"] >= stats["AMPLITUDE_MIN"]) | (df["A_right"] >= stats["AMPLITUDE_MIN"])].copy()
    if "valid_ratio" in active.columns:
        active = active[active["valid_ratio"] >= 0.8]
    if active.empty:
        raise ValueError("No active windows after filtering; check your features.")

    H_left = active["H_left"].to_numpy()
    H_right = active["H_right"].to_numpy()
    H_max = np.maximum(H_left, H_right)  # 高位检测看更高的那只
    H_min = np.minimum(H_left, H_right)  # 低位检测看更低的那只

    # —— 低位阈值：肩线语义优先。默认 0.0；可用负值分布微调但不跨过 0 —— #
    neg = H_min[H_min < 0]
    if neg.size >= 100:
        e_low = float(np.nanpercentile(neg, 20))  # 负值分布的 20 分位
        e_low = float(np.clip(e_low, -0.3, 0.0))  # 语义边界不越过 0
    else:
        e_low = 0.0

    # —— 高位阈值：在正值分布上估计（语义：明显高于肩） —— #
    pos = H_max[H_max > 0]
    e_high_candidates = []

    # 1) 正值分布的高分位（“明显高于肩”）
    if pos.size >= 100:
        q = float(np.nanpercentile(pos, 80))
        e_high_candidates.append(q)

    # 2) Otsu（若正值较多）
    t_otsu = otsu_threshold(pos, nbins=96, rng=(0.0, max(0.4, np.nanpercentile(pos, 95)))) if pos.size >= 300 else None
    if t_otsu is not None and np.isfinite(t_otsu):
        e_high_candidates.append(t_otsu)

    # 3) 兜底：语义默认
    e_high_candidates.append(0.2)

    # 综合：取候选的“稳健中值”，并限制合理范围
    e_high = float(np.median(e_high_candidates))
    e_high = float(np.clip(e_high, 0.05, 0.4))

    # 其它阈值继承/微调
    out = {}
    # 先尝试读取已有 JSON，保留 raw 与 corrected
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, "r") as f:
            try:
                out = json.load(f)
            except Exception:
                out = {}

    # raw 保留（用于对比）
    raw_branch = out.get("raw", {
        "AMPLITUDE_THRESHOLD": float(np.nanpercentile(np.r_[df["A_left"], df["A_right"]], 75)),
        "AMPLITUDE_MIN": stats["AMPLITUDE_MIN"],
        "FREQ_THRESHOLD": float(np.nanpercentile(df["freq"], 75)) if "freq" in df.columns else 3.0,
        "DELTA_A_THRESHOLD": stats["DELTA_A_THRESHOLD"],
        "PHASE_THRESHOLD": 45.0,
        "SINGLE_SIDE_RATIO": 0.7,
        "E_HIGH": float(np.nanpercentile(np.r_[df["H_left"], df["H_right"]], 75)),
        "E_LOW": float(np.nanpercentile(np.r_[df["H_left"], df["H_right"]], 25)),
    })
    out["raw"] = raw_branch

    # corrected 保留（语义基线）
    corrected = out.get("corrected", {
        "AMPLITUDE_THRESHOLD": raw_branch["AMPLITUDE_THRESHOLD"],
        "AMPLITUDE_MIN": raw_branch["AMPLITUDE_MIN"],
        "FREQ_THRESHOLD": raw_branch.get("FREQ_THRESHOLD", 3.0),
        "DELTA_A_THRESHOLD": 20.0,
        "PHASE_THRESHOLD": 45.0,
        "SINGLE_SIDE_RATIO": 0.7,
        "E_HIGH": 0.2,
        "E_LOW": 0.0
    })
    out["corrected"] = corrected

    # 新增 calibrated（语义+数据驱动）
    calibrated = {
        "AMPLITUDE_THRESHOLD": raw_branch["AMPLITUDE_THRESHOLD"],
        "AMPLITUDE_MIN": stats["AMPLITUDE_MIN"],
        "FREQ_THRESHOLD": raw_branch.get("FREQ_THRESHOLD", 3.0),
        "DELTA_A_THRESHOLD": max(20.0, stats["DELTA_A_THRESHOLD"]),  # 稍稳健
        "PHASE_THRESHOLD": 45.0,
        "SINGLE_SIDE_RATIO": 0.7,
        "E_HIGH": e_high,
        "E_LOW": e_low
    }
    out["calibrated"] = calibrated

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=4)
    print("✅ Saved calibrated thresholds to", OUT_JSON)
    print("📊 Calibrated:", json.dumps(calibrated, indent=2))
    print("ℹ️  Notes: E_HIGH from positive H via quantile/Otsu; E_LOW from negative H (≤0) and clipped to ≤0.")


if __name__ == "__main__":
    main()
