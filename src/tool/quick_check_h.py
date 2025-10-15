# quick_check_h.py（可随手放 src/analysis/ 下）
import pandas as pd
df = pd.read_csv("../../data/window_features/all_window_features.csv")

for col in ["H_left","H_right"]:
    s = df[col].dropna()
    print(col, "min", s.min(), "p5", s.quantile(0.05), "p25", s.quantile(0.25),
          "p50", s.quantile(0.5), "p75", s.quantile(0.75), "p95", s.quantile(0.95), "max", s.max())
