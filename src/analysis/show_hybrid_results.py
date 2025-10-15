# -*- coding: utf-8 -*-
"""
Compute Hybrid CNN–LSTM model metrics directly from preds_ensemble.csv.
"""

import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
model_dir = "runs_dl/hybrid_len64_clean_finetune"
pred_csv = os.path.join(model_dir, "preds_ensemble.csv")

print("🔍 Loading predictions from:", pred_csv)

if not os.path.exists(pred_csv):
    raise FileNotFoundError("❌ preds_ensemble.csv not found. Run ensemble_folds.py first.")

# === Load predictions ===
df = pd.read_csv(pred_csv)

# expected columns: ['video', 'frame', 'true_label', 'pred_label'] or similar
# Try to infer column names automatically
possible_truth = [c for c in df.columns if "true" in c.lower() or "label" in c.lower()]
possible_pred = [c for c in df.columns if "pred" in c.lower()]

true_col = possible_truth[0]
pred_col = possible_pred[0]
print(f"✅ Using columns: truth={true_col}, pred={pred_col}")

y_true = df[true_col].astype(str)
y_pred = df[pred_col].astype(str)

# === Compute metrics ===
acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
print("\n📊 Overall Metrics:")
print(f"Accuracy: {acc:.3f}")
print(f"F1_macro: {f1_macro:.3f}")

# === Classification report ===
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, digits=3))

# === Confusion matrix ===
labels = sorted(df[true_col].unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Hybrid CNN–LSTM Confusion Matrix")
plt.tight_layout()
plt.show()
