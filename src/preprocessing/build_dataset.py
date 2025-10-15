import numpy as np
import pandas as pd
import os
from sliding_window import sliding_window
from pathlib import Path

# 标签映射（你也可以放到 data/labels/labels.json）
LABEL_MAP = {
    "no_flap": 0,
    "left_only_low": 1,
    "left_only_high": 2,
    "right_only_low": 3,
    "right_only_high": 4,
    "both_symmetric_low": 5,
    "both_symmetric_high": 6,
    "both_asymmetric": 7
}

def build_dataset(processed_dir, save_dir, label):
    all_X_ml, all_X_dl, all_y = [], [], []

    for file in os.listdir(processed_dir):
        if file.endswith(".npy"):
            path = os.path.join(processed_dir, file)
            features = np.load(path)  # (帧数, 6)

            # 划分窗口
            windows = sliding_window(features, window_size=12, step=1)  # (N, 12, 6)

            # ML: flatten 成 72D 向量
            X_flat = windows.reshape((windows.shape[0], -1))  # (N, 72)

            # DL: 保持原样 (N, 12, 6)
            X_seq = windows

            # 标签
            y = np.full((windows.shape[0],), LABEL_MAP[label])

            # 收集
            all_X_ml.append(X_flat)
            all_X_dl.append(X_seq)
            all_y.append(y)

    # 拼接
    X_ml = np.vstack(all_X_ml)
    X_dl = np.vstack(all_X_dl)
    y = np.concatenate(all_y)

    # 保存目录
    os.makedirs(save_dir, exist_ok=True)

    # ML 数据保存成 CSV
    pd.DataFrame(X_ml).to_csv(os.path.join(save_dir, "X_ml.csv"), index=False)
    pd.DataFrame(y).to_csv(os.path.join(save_dir, "y.csv"), index=False)

    # DL 数据保存成 NPY
    np.save(os.path.join(save_dir, "X_dl.npy"), X_dl)
    np.save(os.path.join(save_dir, "y.npy"), y)

    print(f"✅ Saved ML dataset: {X_ml.shape}, labels: {y.shape}")
    print(f"✅ Saved DL dataset: {X_dl.shape}, labels: {y.shape}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    processed_dir = BASE_DIR / "data" / "processed_angles"
    save_dir = BASE_DIR / "data"
    label = "both_symmetric_low"  # ⚠️ 你要根据视频的动作类别来改
    build_dataset(processed_dir, save_dir, label)
