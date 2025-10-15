import pandas as pd
import os

df_ml = pd.read_csv("runs/foldwise_f1_ml_summary.csv")
df_dl = pd.read_csv("runs/foldwise_f1_summary.csv")

df_all = pd.concat([df_ml, df_dl], ignore_index=True)
df_all = df_all[["Model", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean", "Std"]]
df_all = df_all.round(3)

os.makedirs("runs", exist_ok=True)
df_all.to_csv("runs/foldwise_f1_all_models.csv", index=False)

print("\n✅ Combined fold-wise F1 summary for all models:")
print(df_all.to_string(index=False))
print("\n📊 Saved to runs/foldwise_f1_all_models.csv")
