"""Generate aggregate report figures from the frozen summary only."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "artifacts/experiments/c10_frozen_test_seed42_candidates5000/test_summary.json"
OUTPUT = ROOT / "docs/figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    if args.suffix and not args.suffix.replace("_", "").isalnum():
        raise ValueError("suffix may contain only letters, numbers and underscores")
    suffix = f"_{args.suffix}" if args.suffix else ""
    methods = ["popularity", "user_pearson_cf", "content", "hybrid"]
    labels = ["Phổ biến", "Pearson CF", "Nội dung", "Kết hợp"]
    colors = ["#83918c", "#c56b4e", "#2b7a6b", "#173f5f"]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    x = range(len(methods))
    width = 0.24
    for offset, metric, name in (
        (-width, "precision_at_10", "Precision@10"),
        (0, "recall_at_10", "Recall@10"),
        (width, "ndcg_at_10", "NDCG@10"),
    ):
        values = [data["methods"][method][metric] for method in methods]
        ax.bar([value + offset for value in x], values, width, label=name)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, max(0.15, max(data["methods"][method]["recall_at_10"] for method in methods) * 1.15))
    ax.set_ylabel("Trung bình trên người dùng")
    ax.set_title("Chỉ số lần test thứ hai trên 738 người dùng" if suffix else "Chỉ số kiểm tra cố định trên 738 người dùng đủ điều kiện")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"test_metrics{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    values = [data["methods"][method]["latency_ms"]["mean"] for method in methods]
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Mili giây trung bình mỗi người dùng")
    ax.set_title("Thời gian gợi ý ngoại tuyến trong lần test thứ hai" if suffix else "Thời gian trả gợi ý trên tập kiểm tra cố định")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, fmt="%.3f ms", padding=4)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"test_latency{suffix}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
