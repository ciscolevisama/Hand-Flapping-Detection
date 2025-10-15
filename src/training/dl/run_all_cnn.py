"""
run_all_dl_modes.py
--------------------------------------------------
自动跑 3 种数据模式 (full / clean_only / conf_weighted)
并汇总 f1_macro、acc、highF1 对比表和图表。

运行示例：
    python src/training/run_all_dl_modes.py
--------------------------------------------------
"""

import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def run_and_capture(cmd):
    """运行命令并捕获控制台输出文本"""
    print(f"\n🚀 Running: {cmd}\n" + "=" * 80)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout


def parse_metrics(output):
    """
    从训练日志文本中提取最终 f1_macro / highF1 / acc。
    匹配形如：
      [Fold x] final f1_macro=0.4347 highF1=0.1697 acc=0.8957
    """
    pattern = re.compile(
        r"final f1_macro=([\d\.]+)\s+highF1=([\d\.]+)\s+acc=([\d\.]+)"
    )
    f1s, highs, accs = [], [], []
    for f1, h, a in pattern.findall(output):
        f1s.append(float(f1))
        highs.append(float(h))
        accs.append(float(a))
    if f1s:
        return sum(f1s) / len(f1s), sum(highs) / len(highs), sum(accs) / len(accs)
    else:
        return None, None, None


def main():
    # ✅ 三种模式的配置
    modes = ["full", "clean_only", "conf_weighted"]

    # 训练脚本位置（你现有的 train_cnn.py）
    base_cmd = (
        "python src/training/dl/train_cnn.py "
        "--seq_len 64 --epochs 40 --lr 5e-4 --cv 5 "
        "--balancing both --sampler_alpha 0.7 --cw_beta 0.5 --cw_clip 3 "
        "--loss focal --gamma 2.0 "
    )

    results = []

    for mode in modes:
        outdir = f"runs_dl/cnn_len64_{mode}_auto"
        cmd = base_cmd + f"--mode {mode} --outdir {outdir}"
        log = run_and_capture(cmd)
        f1, highF1, acc = parse_metrics(log)
        print(f"✅ {mode}: f1_macro={f1:.4f}, highF1={highF1:.4f}, acc={acc:.4f}")
        results.append((mode, f1, highF1, acc))

    # 汇总为 DataFrame
    df = pd.DataFrame(results, columns=["mode", "f1_macro", "highF1", "acc"])
    out_csv = Path("runs_dl") / "dl_modes_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n📊 Summary saved to {out_csv}")

    # 绘图
    plt.figure(figsize=(8, 5))
    x = range(len(df))
    plt.bar(x, df["f1_macro"], width=0.25, label="F1_macro")
    plt.bar([i + 0.25 for i in x], df["highF1"], width=0.25, label="High F1")
    plt.bar([i + 0.5 for i in x], df["acc"], width=0.25, label="Accuracy")

    plt.xticks([i + 0.25 for i in x], df["mode"])
    plt.ylabel("Score")
    plt.title("CNN Performance under Different Data Modes")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_png = Path("runs_dl") / "dl_modes_summary.png"
    plt.savefig(out_png, dpi=300)
    print(f"🖼️ Figure saved to {out_png}")

    print("\n✅ All experiments finished.")
    print(df)


if __name__ == "__main__":
    main()
