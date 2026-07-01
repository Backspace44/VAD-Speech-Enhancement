"""Reporting helpers for speech enhancement evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.eval.eval_core import METRIC_NAMES
from src.experiments import save_json


def stats_to_dataframe(stats: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """Flatten nested statistics into a dataframe."""
    rows = []
    for method, metrics in stats.items():
        for metric_name, metric_stats in metrics.items():
            rows.append(
                {
                    "method": method,
                    "metric": metric_name,
                    **metric_stats,
                }
            )
    return pd.DataFrame(rows)


def improvements_to_dataframe(improvements: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Flatten metric improvements into a dataframe."""
    rows = []
    for method, metrics in improvements.items():
        for metric_name, delta in metrics.items():
            rows.append(
                {
                    "method": method,
                    "metric": metric_name,
                    "improvement": delta,
                }
            )
    return pd.DataFrame(rows)


def save_results(stats: Dict, improvements: Dict, output_dir: Path):
    """Save results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "evaluation_stats.json", stats)
    save_json(output_dir / "improvements.json", improvements)
    stats_to_dataframe(stats).to_csv(output_dir / "evaluation_stats.csv", index=False)
    improvements_to_dataframe(improvements).to_csv(output_dir / "improvements.csv", index=False)
    print(f"\nResults saved to {output_dir}")


def print_summary_clean(stats: Dict, improvements: Dict):
    """Print summary to console using ASCII-safe formatting."""
    print("\n" + "=" * 70)
    print(" EVALUATION SUMMARY")
    print("=" * 70)

    print("\nNoisy Baseline:")
    for metric in METRIC_NAMES:
        if metric in stats.get("Noisy", {}):
            val = stats["Noisy"][metric]["mean"]
            std = stats["Noisy"][metric]["std"]
            print(f"  {metric.upper():8s}: {val:7.3f} (+/-{std:.3f})")

    for method in stats.keys():
        if method == "Noisy":
            continue

        print(f"\n {method.replace('_', ' ')}:")
        for metric in METRIC_NAMES:
            if metric in stats[method]:
                val = stats[method][metric]["mean"]
                std = stats[method][metric]["std"]
                imp = improvements.get(method, {}).get(metric, 0)
                sign = "+" if imp >= 0 else ""
                print(f"  {metric.upper():8s}: {val:7.3f} (+/-{std:.3f})  [{sign}{imp:.3f}]")

    print("\n" + "=" * 70)
