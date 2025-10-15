import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report

# === 路径配置 ===
model_path = "runs/clean/model_rf.pkl"
feature_file = "data/window_features/all_window_features.csv"
label_csv = "data/labels/weak_labels.csv"

# === 加载模型和数据 ===
print(f"✅ Using model: {model_path}")
print(f"✅ Using features: {feature_file}")
print(f"✅ Using labels: {label_csv}")

rf = joblib.load(model_path)
df = pd.read_csv(feature_file)

# ---- 明确选取用于训练的特征列 ----
feature_cols = ["A_left", "A_right", "freq", "delta_A", "H_left", "H_right", "valid_ratio"]
X = df[feature_cols]
print(f"🧩 Using {len(feature_cols)} feature columns: {feature_cols}")

y = pd.read_csv(label_csv)["label"]

# === 检查长度一致性 ===
if len(X) != len(y):
    print(f"⚠️ Warning: X ({len(X)}) and y ({len(y)}) length mismatch. Truncating to min length.")
    n = min(len(X), len(y))
    X, y = X.iloc[:n], y.iloc[:n]

# === 预测与评估 ===
y_pred = rf.predict(X)

print("\nClassification Report:")
print(classification_report(y, y_pred, digits=3))

# === 导出预测结果 ===
df_out = pd.DataFrame({"true_label": y, "pred_label": y_pred})
os.makedirs("runs_ml", exist_ok=True)
out_path = "runs_ml/preds_rf.csv"
df_out.to_csv(out_path, index=False)

print(f"\n✅ Exported prediction results: {out_path}")
