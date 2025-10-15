import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------
# Read the combined fold-wise results (ML + DL)
# ------------------------------------------------------
df = pd.read_csv("runs/foldwise_f1_all_models.csv")

# Melt the dataframe to long format for Seaborn
fold_cols = [c for c in df.columns if c.startswith("Fold")]
df_long = df.melt(id_vars=["Model"], value_vars=fold_cols,
                  var_name="Fold", value_name="F1")

# Set figure style (academic style)
sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.1)

plt.figure(figsize=(8, 5))
ax = sns.boxplot(
    data=df_long,
    x="Model",
    y="F1",
    palette="pastel",
    width=0.6,
    fliersize=3,
    boxprops={"edgecolor": "black"},
    medianprops={"color": "black"},
)

# Optional: overlay swarm points (individual fold results)
sns.swarmplot(data=df_long, x="Model", y="F1", color="grey", alpha=0.6, size=4)

plt.title("Figure 4.2 – Cross-Validation Consistency of Model Performance", fontsize=12, weight="bold")
plt.ylabel("Macro-F1 Score", fontsize=11)
plt.xlabel("Model", fontsize=11)
plt.ylim(0.4, 1.0)
plt.xticks(rotation=20)
plt.tight_layout()

# ------------------------------------------------------
# Save figure
# ------------------------------------------------------
os.makedirs("figures", exist_ok=True)
out_path = "figures/Figure_4_2_Boxplot.png"
plt.savefig(out_path, dpi=300)
print(f"✅ Saved {out_path}")
plt.show()
