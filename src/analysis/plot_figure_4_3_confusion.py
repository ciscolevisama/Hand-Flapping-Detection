import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------
# 1. Load prediction results
# ------------------------------------------------------
rf_path = "runs_ml/preds_rf.csv"
hyb_path = "runs_dl/hybrid_len64_clean_finetune/preds_ensemble.csv"

df_rf = pd.read_csv(rf_path)
df_hyb = pd.read_csv(hyb_path)

# Make sure column names match
true_col = "true_label"
pred_col = "pred_label"

# ------------------------------------------------------
# 2. Define class order for consistent axis
# ------------------------------------------------------
classes = [
    "no_flap",
    "left_only_low", "left_only_high",
    "right_only_low", "right_only_high",
    "both_symmetric_low", "both_symmetric_high",
    "both_asymmetric"
]

# ------------------------------------------------------
# 3. Compute normalised confusion matrices
# ------------------------------------------------------
cm_rf = confusion_matrix(df_rf[true_col], df_rf[pred_col], labels=classes, normalize="true")
cm_hyb = confusion_matrix(df_hyb[true_col], df_hyb[pred_col], labels=classes, normalize="true")

# ------------------------------------------------------
# 4. Plot side-by-side heatmaps
# ------------------------------------------------------
sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def plot_cm(ax, cm, title):
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plot_cm(axes[0], cm_rf, "(a) Random Forest")
plot_cm(axes[1], cm_hyb, "(b) Hybrid CNN–LSTM")

plt.suptitle("Figure 4.3 – Normalised Confusion Matrices for RF and Hybrid CNN–LSTM Models",
             fontsize=12, weight="bold", y=1.03)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
out_path = "figures/Figure_4_3_ConfusionMatrices.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved {out_path}")
plt.show()
