import os
import numpy as np
import pandas as pd
import json

# 路径
processed_dir = "../../data/processed_angles"
label_path = "../../data/labels/weak_labels.csv"
thr_path = "../../data/analysis/recommended_thresholds_window.json"

# 加载阈值
with open(thr_path, "r") as f:
    thr_data = json.load(f)

THRESHOLDS = thr_data["corrected"] if "corrected" in thr_data else thr_data
E_HIGH = THRESHOLDS.get("E_HIGH", 0.2)
E_LOW = THRESHOLDS.get("E_LOW", -0.2)

print(f"阈值: E_HIGH={E_HIGH}, E_LOW={E_LOW}")

# 容差，避免边界掉入“其它”
EPS = 1e-6

def classify(H_left, H_right):
    # 三态判定
    def state(H):
        if H >= E_HIGH - EPS:
            return "high"
        elif H <= E_LOW + EPS:
            return "low"
        else:
            return "mid"

    sL, sR = state(H_left), state(H_right)

    # 分类逻辑
    if (sL == "high" and sR == "low") or (sL == "low" and sR == "high"):
        return "一高一低"
    elif (sL == "high" and sR == "mid") or (sL == "mid" and sR == "high"):
        return "一高一模糊"
    elif (sL == "low" and sR == "mid") or (sL == "mid" and sR == "low"):
        return "一低一模糊"
    elif sL == "mid" and sR == "mid":
        return "双手都模糊"
    else:
        return "其它"  # 理论上不该再出现

# 载入 weak_labels.csv
labels = pd.read_csv(label_path)
uncertain_rows = labels[labels["uncertain"] == 1]
print(f"总 uncertain 窗口: {len(uncertain_rows)}")

# 分类计数
stats = {"一高一低": 0, "一高一模糊": 0, "一低一模糊": 0, "双手都模糊": 0, "其它": 0}

for _, row in uncertain_rows.iterrows():
    video = row["video"] + ".npy"
    start, end = int(row["start_frame"]), int(row["end_frame"])

    # 加载原始窗口
    path = os.path.join(processed_dir, video)
    if not os.path.exists(path):
        continue
    data = np.load(path)
    window = data[start:end]

    H_left = np.median(window[:, 4])
    H_right = np.median(window[:, 5])

    cat = classify(H_left, H_right)
    stats[cat] += 1

# 输出结果
total = len(uncertain_rows)
print("\n=== uncertain 样本来源统计 ===")
rows = []
for k, v in stats.items():
    rate = 100 * v / total if total > 0 else 0
    print(f"{k:8s}: {v:5d} ({rate:.2f}%)")
    rows.append([k, v, rate])

# 保存为 CSV
df_out = pd.DataFrame(rows, columns=["类别", "数量", "占比(%)"])
out_path = "../../data/analysis/uncertain_breakdown.csv"
df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 已保存结果到 {out_path}")
