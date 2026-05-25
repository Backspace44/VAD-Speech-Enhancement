"""Plotting helpers for speech enhancement evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.eval.eval_core import METRIC_NAMES


def plot_results(stats: Dict, improvements: Dict, output_dir: Path):
    """Generate PNG plots comparing methods."""
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in stats.keys() if m != "Noisy"]
    metrics = METRIC_NAMES

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        all_methods = ["Noisy"] + methods
        values = []
        errors = []

        for method in all_methods:
            if metric in stats[method]:
                values.append(stats[method][metric]["mean"])
                errors.append(stats[method][metric]["std"])
            else:
                values.append(0)
                errors.append(0)

        x_pos = np.arange(len(all_methods))
        bars = ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7)
        bars[0].set_color("red")
        bars[0].set_alpha(0.3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_methods, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} Comparison")
        ax.grid(True, alpha=0.3)

    fig.delaxes(axes[5])
    plt.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    print(" Saved: plots/metrics_comparison.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    metric_labels = ["PESQ", "STOI", "SNR", "SegSNR", "LSD"]
    x = np.arange(len(metric_labels))
    width = 0.25

    for i, method in enumerate(methods):
        values = []
        for metric in metrics:
            if metric in improvements[method]:
                values.append(improvements[method][metric])
            else:
                values.append(0)
        ax.bar(x + i * width, values, width, label=method.replace("_", " "), alpha=0.8)

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Improvement over Noisy")
    ax.set_title("Improvement over Noisy Baseline")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(plots_dir / "improvements.png", dpi=150, bbox_inches="tight")
    print(" Saved: plots/improvements.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("tight")
    ax.axis("off")

    table_data = [["Method"] + [m.upper() for m in metrics]]
    for method in ["Noisy"] + methods:
        row = [method.replace("_", " ")]
        for metric in metrics:
            if metric in stats[method]:
                val = stats[method][metric]["mean"]
                row.append(f"{val:.3f}")
            else:
                row.append("N/A")
        table_data.append(row)

    table_data.append(["---"] * (len(metrics) + 1))
    table_data.append(["Improvement", "", "", "", "", ""])

    for method in methods:
        row = [method.replace("_", " ")]
        for metric in metrics:
            if metric in improvements[method]:
                val = improvements[method][metric]
                row.append(f"+{val:.3f}" if val >= 0 else f"{val:.3f}")
            else:
                row.append("N/A")
        table_data.append(row)

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(metrics) + 1):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title("Enhancement Methods - Summary Table", fontsize=14, weight="bold", pad=20)
    plt.savefig(plots_dir / "summary_table.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/summary_table.png")
    plt.close()
