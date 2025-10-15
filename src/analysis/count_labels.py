import os
import argparse
import pandas as pd
import json
from collections import Counter

# taxonomy 保持一致
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

def count_from_csv(path, label_col="label"):
    df = pd.read_csv(path)
    return Counter(df[label_col])

def count_from_jsonl(path, label_key="label"):
    counts = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            label = data.get(label_key)
            if label is not None:
                counts[label] += 1
    return counts

def main(args):
    # 自动识别文件类型
    if args.file.endswith(".csv"):
        counts = count_from_csv(args.file, label_col=args.label_col)
    elif args.file.endswith(".json") or args.file.endswith(".jsonl"):
        counts = count_from_jsonl(args.file, label_key=args.label_col)
    else:
        raise ValueError("Unsupported file format. Use .csv or .jsonl")

    # 填补缺失类别
    counts = {cls: counts.get(cls, 0) for cls in CLASS_NAMES}

    # 保存为 CSV
    out_path = os.path.join(args.out_dir, "label_counts.csv")
    pd.DataFrame(list(counts.items()), columns=["class", "count"]).to_csv(out_path, index=False)

    print("✅ Saved label counts to", out_path)
    for cls, n in counts.items():
        print(f"{cls:20s} : {n}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Input label file (.csv or .jsonl)")
    parser.add_argument("--out_dir", type=str, default="analysis_out", help="Output directory")
    parser.add_argument("--label_col", type=str, default="label", help="Label column/key name")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
