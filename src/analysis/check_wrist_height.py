import pandas as pd
import matplotlib.pyplot as plt
import os

# 路径
in_path = "../../data/window_features/all_window_features.csv"
out_dir = "../../data/analysis"
os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(in_path):
    raise FileNotFoundError(f"❌ Input file not found: {in_path}. Run extract_window_features.py first.")

# 读取数据
df = pd.read_csv(in_path)

# 检查列是否存在
if "H_left" not in df.columns or "H_right" not in df.columns:
    raise KeyError("❌ H_left / H_right not found in CSV. Check extract_window_features.py!")

# 基本统计
print("📊 Wrist relative height (H_left & H_right):")
print(df[["H_left", "H_right"]].describe())

# 保存直方图
plt.figure(figsize=(10, 5))
plt.hist(df["H_left"], bins=50, alpha=0.5, label="H_left")
plt.hist(df["H_right"], bins=50, alpha=0.5, label="H_right")
plt.axvline(df["H_left"].median(), color="blue", linestyle="--", label="H_left median")
plt.axvline(df["H_right"].median(), color="orange", linestyle="--", label="H_right median")
plt.legend()
plt.title("Distribution of wrist relative heights (normalised)")
plt.xlabel("Relative height (shoulder=0, hip=-1)")
plt.ylabel("Count")

out_path = os.path.join(out_dir, "wrist_height_distribution.png")
plt.savefig(out_path)
plt.close()

print(f"✅ Histogram saved to {out_path}")
