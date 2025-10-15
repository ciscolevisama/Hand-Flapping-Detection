import os
import pandas as pd

# 路径配置
label_file = "../../data/labels/weak_labels.csv"
out_dir = "../../data/labels"
os.makedirs(out_dir, exist_ok=True)

def split_labels(label_file, out_dir):
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"❌ Label file not found: {label_file}")

    df = pd.read_csv(label_file)

    if "uncertain" not in df.columns:
        raise ValueError("❌ 'uncertain' column not found in weak_labels.csv. "
                         "Please run the updated segment_and_label.py first.")

    # Clean 数据（uncertain = 0）
    df_clean = df[df["uncertain"] == 0].copy()
    clean_path = os.path.join(out_dir, "clean_labels.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"✅ Saved clean labels: {len(df_clean)} samples → {clean_path}")

    # Noisy 数据（uncertain = 1）
    df_noisy = df[df["uncertain"] == 1].copy()
    noisy_path = os.path.join(out_dir, "noisy_labels.csv")
    df_noisy.to_csv(noisy_path, index=False)
    print(f"✅ Saved noisy labels: {len(df_noisy)} samples → {noisy_path}")

    # 简要统计
    summary = df.groupby("uncertain")["label"].value_counts().unstack(fill_value=0)
    print("\n📊 Label distribution by uncertainty:")
    print(summary)

if __name__ == "__main__":
    split_labels(label_file, out_dir)
