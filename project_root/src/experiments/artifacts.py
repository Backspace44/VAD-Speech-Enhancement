"""Common experiment artifact helpers used by training and evaluation scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def save_json(path: Path, payload: dict | list) -> Path:
    """Persist JSON data with a stable UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def save_csv_rows(path: Path, rows: list[dict]) -> Path:
    """Persist tabular rows to CSV using the first row as schema."""
    if not rows:
        raise ValueError(f"Cannot save empty CSV row set to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_csv_row(path: Path, row: dict) -> Path:
    """Append a single row to a CSV file, creating the header when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def save_training_history(exp_dir: Path, history: list[dict]) -> Path:
    """Persist per-epoch training history for an experiment."""
    return save_json(exp_dir / "training_history.json", history)


def save_experiment_summary(exp_dir: Path, summary: dict) -> Path:
    """Persist a compact experiment summary for easy comparison."""
    return save_json(exp_dir / "summary.json", summary)


def build_experiment_index_row(summary: dict, summary_path: Path) -> dict:
    """Flatten experiment metadata into the global experiment index format."""
    return {
        "timestamp": summary["timestamp"],
        "experiment_name": summary["experiment_name"],
        "model_variant": summary["model_variant"],
        "optimizer": summary["optimizer"],
        "scheduler": summary["scheduler"],
        "loss": summary["loss"],
        "dataset": summary["dataset"],
        "epochs_completed": summary["num_epochs_completed"],
        "best_epoch": summary["best_epoch"],
        "best_val_loss": summary["best_val_loss"],
        "final_train_loss": summary["final_train_loss"],
        "final_val_loss": summary["final_val_loss"],
        "checkpoint_dir": summary["checkpoint_dir"],
        "best_checkpoint": summary["best_checkpoint"],
        "summary_file": str(summary_path),
    }


def load_experiment_summary(checkpoints_dir: Path, experiment_name: str) -> dict:
    """Load a saved experiment summary and annotate its source path."""
    summary_path = checkpoints_dir / experiment_name / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary file for experiment '{experiment_name}': {summary_path}"
        )
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["summary_path"] = str(summary_path)
    return summary


def classify_experiment_summary(summary: dict) -> str:
    """Assign a benchmark category using the current project conventions."""
    dataset = summary.get("dataset", "unknown")
    model_variant = summary.get("model_variant", "unknown")

    if dataset == "librispeech" and "tiny_fast" in model_variant:
        return "fast_benchmark"
    if dataset == "librispeech":
        return "quality_librispeech"
    if dataset == "voicebank" and "balanced_res" in model_variant:
        return "quality_voicebank"
    if dataset == "voicebank":
        return "baseline_voicebank"
    return "other"


def build_benchmark_row(summary: dict) -> dict:
    """Flatten one experiment summary into the benchmark report shape."""
    return {
        "experiment_name": summary["experiment_name"],
        "category": classify_experiment_summary(summary),
        "dataset": summary["dataset"],
        "model_variant": summary["model_variant"],
        "optimizer": summary["optimizer"],
        "scheduler": summary["scheduler"],
        "loss": summary["loss"],
        "num_epochs_completed": summary["num_epochs_completed"],
        "final_train_loss": summary["final_train_loss"],
        "final_val_loss": summary["final_val_loss"],
        "best_val_loss": summary["best_val_loss"],
        "best_epoch": summary["best_epoch"],
        "best_checkpoint": summary["best_checkpoint"],
        "summary_path": summary["summary_path"],
    }


def pick_best_rows_by_key(rows: list[dict], key_field: str, metric_field: str) -> dict[str, dict]:
    """Select the best row per key using ascending metric order."""
    ordered = sorted(rows, key=lambda row: (row[key_field], float(row[metric_field])))
    best_rows: dict[str, dict] = {}
    for row in ordered:
        key = row[key_field]
        if key not in best_rows:
            best_rows[key] = row
    return best_rows
