"""Helpers for formatting and presenting speech enhancement metrics."""

from __future__ import annotations

from typing import Iterable


DISPLAY_NAME_MAP = {
    "pesq": "PESQ",
    "stoi": "STOI",
    "snr": "SNR",
    "segsnr": "SegSNR",
    "lsd": "LSD",
}


def build_demo_metric_rows(results_by_method: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    """Convert evaluation metrics into the row format used by demo charts."""
    rows: list[dict[str, float | str]] = []
    for method, metrics in results_by_method.items():
        rows.append(
            {
                "Method": method,
                "PESQ": float(metrics.get("pesq", 0.0)),
                "STOI": float(metrics.get("stoi", 0.0)),
                "SNR": float(metrics.get("snr", 0.0)),
            }
        )
    return rows


def format_metric_lines(
    metrics: dict[str, float],
    keys: Iterable[str] = ("pesq", "stoi", "snr"),
) -> list[str]:
    """Return human-readable metric lines for console output."""
    lines: list[str] = []
    for key in keys:
        value = float(metrics.get(key, 0.0))
        label = DISPLAY_NAME_MAP.get(key, key.upper())
        suffix = " dB" if key in {"snr", "segsnr"} else ""
        lines.append(f"  {label}: {value:6.3f}{suffix}")
    return lines
