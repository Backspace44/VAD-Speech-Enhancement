"""Checkpoint and experiment artifact helpers for training."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from src import config
from src.experiments import (
    append_csv_row,
    build_experiment_index_row,
    save_experiment_summary,
    save_training_history,
)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    batch_idx,
    train_loss,
    exp_dir,
    name: str = "checkpoint",
    extra_state: dict | None = None,
):
    """Save a periodic or interrupt-resume checkpoint."""
    checkpoint_path = exp_dir / f"{name}.pth"
    checkpoint_dict = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "model_config": deepcopy(config.MODEL_CONFIG),
        "model_variant": config.ACTIVE_MODEL_VARIANT,
    }
    if scheduler is not None:
        checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state:
        checkpoint_dict.update(extra_state)

    torch.save(checkpoint_dict, checkpoint_path)
    return checkpoint_path


def save_best_model_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    avg_train_loss: float,
    avg_val_loss: float,
    exp_dir: Path,
    cli_args: dict,
) -> Path:
    """Save the current best validation checkpoint."""
    checkpoint_path = exp_dir / "masknet_best.pth"
    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "config": cli_args,
        "model_config": deepcopy(config.MODEL_CONFIG),
        "model_variant": config.ACTIVE_MODEL_VARIANT,
    }
    if scheduler is not None:
        checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint_dict, checkpoint_path)
    return checkpoint_path


def finalize_experiment(
    exp_dir: Path,
    history: list[dict],
    *,
    best_epoch: int | None,
    best_val_loss: float,
    use_amp: bool,
) -> tuple[Path, Path]:
    """Write training history, summary, and experiment index row."""
    history_path = save_training_history(exp_dir, history)

    summary = {
        "experiment_name": exp_dir.name,
        "model_variant": config.ACTIVE_MODEL_VARIANT,
        "optimizer": config.OPTIMIZER_CONFIG["type"],
        "scheduler": config.TRAINING_CONFIG.get("scheduler_type"),
        "loss": config.LOSS_CONFIG["type"],
        "dataset": config.DATASET_CONFIG["dataset"],
        "num_epochs_configured": int(config.TRAINING_CONFIG["num_epochs"]),
        "num_epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss) if history else None,
        "final_train_loss": float(history[-1]["train_loss"]) if history else None,
        "final_val_loss": float(history[-1]["val_loss"]) if history else None,
        "use_mixed_precision": bool(use_amp),
        "checkpoint_dir": str(exp_dir),
        "best_checkpoint": str(exp_dir / "masknet_best.pth"),
        "history_file": str(history_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = save_experiment_summary(exp_dir, summary)

    append_csv_row(
        config.RESULTS_DIR / "experiment_index.csv",
        build_experiment_index_row(summary, summary_path),
    )
    return history_path, summary_path
