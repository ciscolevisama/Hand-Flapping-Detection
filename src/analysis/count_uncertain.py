import os
import argparse
import pandas as pd

CLASS_NAMES = [
    "no_flap",
    "left_only_low",
    "left_only_high",
    "right_only_low",
    "right_only_high",
    "both_symmetric_low",
    "both_symmetric_high",
    "both_asymmetric"
]


def main(args):
    df = pd.read_csv(args.file)

    stats = []
    for cls in CLASS_NAMES:
        sub = df[df["label"] == cls]
        total = len(sub)
        unc = int(sub["uncertain"].sum()) if total > 0 else 0
        ratio = unc / total if total > 0 else 0
        stats.append({
            "class": cls,
            "total": total,
            "uncertain_count": unc,
            "uncertain_ratio": round(ratio, 3)
        })

    out_df = pd.DataFrame(stats)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "uncertain_stats.csv")
    out_df.to_csv(out_path, index=False)

    print(f"✅ Saved to {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True,
                        help="Path to weak_labels.csv")
    parser.add_argument("--out_dir", type=str, default="analysis_out",
                        help="Directory to save results")
    args = parser.parse_args()
    main(args)
