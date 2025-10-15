import pandas as pd
import os

label_path = "../../data/labels/weak_labels.csv"

if not os.path.exists(label_path):
    raise FileNotFoundError(f"{label_path} not found")

df = pd.read_csv(label_path)

if "confidence" not in df.columns:
    raise KeyError("No 'confidence' column found in weak_labels.csv")

# 统计 NaN
nan_count = df["confidence"].isna().sum()
total = len(df)
nan_rows = df[df["confidence"].isna()]

print(f"总样本数: {total}")
print(f"confidence=NaN 的样本数: {nan_count}")
print(f"占比: {nan_count/total*100:.2f}%")

# 如果有 NaN，展示前几条
if nan_count > 0:
    print("\n前几条 NaN 样本:")
    print(nan_rows.head())
