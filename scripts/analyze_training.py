"""训练日志分析工具

从训练日志和 GPU 监控数据中提取关键信息，生成可读报告。

用法:
    python scripts/analyze_training.py                          # 分析默认路径
    python scripts/analyze_training.py --log outputs/verl_grpo/train.log
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


def parse_memory_logs(log_file: Path) -> list[dict]:
    """从训练日志中提取 [MEM] 显存变化点。"""
    pattern = re.compile(
        r"(?P<tag>Before|After)\s+(?P<event>.+?),\s+"
        r"memory allocated \(GB\):\s*(?P<alloc>[\d.]+),\s+"
        r"memory reserved \(GB\):\s*(?P<reserved>[\d.]+),\s+"
        r"device memory used/total \(GB\):\s*(?P<used>[\d.]+)/(?P<total>[\d.]+)"
    )
    entries = []
    for line in log_file.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            entries.append({
                "tag": m.group("tag"),
                "event": m.group("event").strip(),
                "alloc_gb": float(m.group("alloc")),
                "reserved_gb": float(m.group("reserved")),
                "used_gb": float(m.group("used")),
                "total_gb": float(m.group("total")),
            })
    return entries


def parse_gpu_csv(csv_file: Path):
    """从 GPU 监控 CSV 提取统计。"""
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        return df
    except Exception:
        return None


def analyze_memory_flow(entries: list[dict]):
    """分析显存变化流，提取关键阶段。"""
    if not entries:
        print("  (无显存日志数据)")
        return

    print("\n" + "=" * 70)
    print("  显存变化流（DEBUG log_gpu_memory_usage 提取）")
    print("=" * 70)

    prev_alloc = 0.0
    for e in entries:
        delta = e["alloc_gb"] - prev_alloc
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        bar_len = int(e["alloc_gb"] / e["total_gb"] * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(
            f"  [{e['tag']:6s}] {e['event'][:50]:<50s} "
            f"alloc={e['alloc_gb']:6.1f}GB ({delta_str:>6s}) "
            f"|{bar}| {e['used_gb']:.1f}/{e['total_gb']:.0f}GB"
        )
        prev_alloc = e["alloc_gb"]

    peak = max(e["alloc_gb"] for e in entries)
    total = entries[0]["total_gb"]
    print(f"\n  峰值显存: {peak:.1f} GB / {total:.0f} GB ({peak/total*100:.1f}%)")


def analyze_gpu_utilization(df):
    """分析 GPU 利用率。"""
    if df is None or len(df) == 0:
        print("  (无 GPU 监控数据)")
        return

    print("\n" + "=" * 70)
    print("  GPU 利用率统计（nvidia-smi 采样）")
    print("=" * 70)

    for gpu_id in sorted(df["gpu_id"].unique()):
        g = df[df["gpu_id"] == gpu_id]
        print(f"\n  GPU {gpu_id}:")
        print(f"    显存:   avg={g['mem_used_mb'].mean():.0f}MB, "
              f"peak={g['mem_used_mb'].max():.0f}MB / {g['mem_total_mb'].iloc[0]:.0f}MB "
              f"({g['mem_used_mb'].max()/g['mem_total_mb'].iloc[0]*100:.1f}%)")
        print(f"    利用率: avg={g['gpu_util%'].mean():.1f}%, "
              f"peak={g['gpu_util%'].max():.1f}%")
        print(f"    功耗:   avg={g['power_w'].mean():.0f}W, "
              f"peak={g['power_w'].max():.0f}W")
        print(f"    温度:   avg={g['temp_c'].mean():.0f}°C, "
              f"peak={g['temp_c'].max():.0f}°C")


def analyze_rollout_data(rollout_dir: Path):
    """分析 rollout 数据，查看模型生成质量变化。"""
    import json

    jsonl_files = sorted(rollout_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("  (无 rollout 数据)")
        return

    print("\n" + "=" * 70)
    print("  Rollout 数据（模型生成质量变化）")
    print("=" * 70)

    for f in jsonl_files:
        records = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if not records:
            continue
        scores = [r.get("score", 0) for r in records]
        step = records[0].get("step", "?")
        avg_score = sum(scores) / len(scores) if scores else 0
        pos_ratio = sum(1 for s in scores if s > 0) / len(scores) * 100 if scores else 0
        perfect = sum(1 for s in scores if s >= 1.0) / len(scores) * 100 if scores else 0
        print(f"  Step {step:>4s}: n={len(records):>4d}, "
              f"avg_reward={avg_score:.3f}, "
              f"positive={pos_ratio:.1f}%, "
              f"perfect={perfect:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="训练日志分析工具")
    parser.add_argument("--log", type=str, default="outputs/verl_grpo/train.log")
    parser.add_argument("--gpu_csv", type=str, default="outputs/verl_grpo/gpu_monitor.csv")
    parser.add_argument("--rollout_dir", type=str, default="outputs/verl_grpo/rollout_data")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║          GRPO Training Analysis Report          ║")
    print("╚══════════════════════════════════════════════════╝")

    log_file = Path(args.log)
    if log_file.exists():
        entries = parse_memory_logs(log_file)
        analyze_memory_flow(entries)
    else:
        print(f"\n  日志文件不存在: {log_file}")

    gpu_csv = Path(args.gpu_csv)
    if gpu_csv.exists():
        df = parse_gpu_csv(gpu_csv)
        analyze_gpu_utilization(df)
    else:
        print(f"\n  GPU 监控文件不存在: {gpu_csv}")

    rollout_dir = Path(args.rollout_dir)
    if rollout_dir.exists():
        analyze_rollout_data(rollout_dir)
    else:
        print(f"\n  Rollout 目录不存在: {rollout_dir}")


if __name__ == "__main__":
    main()
