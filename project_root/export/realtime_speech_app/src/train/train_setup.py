"""Setup helpers for training configuration, loss, optimizer, and scheduler."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from src import config


def apply_args_to_config(args):
    """Apply command line arguments to configuration."""
    if args.experiment_preset:
        config.apply_preset(args.experiment_preset, config_type="experiment")

    if args.preset:
        config.apply_preset(args.preset, config_type="training")

    if args.model_variant:
        config.apply_preset(args.model_variant, config_type="model")

    if args.optimizer_preset:
        config.apply_preset(args.optimizer_preset, config_type="optimizer")

    if args.loss_preset:
        config.apply_preset(args.loss_preset, config_type="loss")

    if args.base_channels is not None:
        config.MODEL_CONFIG["base_channels"] = args.base_channels
    if args.use_lstm is not None:
        config.MODEL_CONFIG["use_lstm"] = args.use_lstm
    if args.use_attention is not None:
        config.MODEL_CONFIG["use_attention"] = args.use_attention
    if args.use_residual is not None:
        config.MODEL_CONFIG["use_residual"] = args.use_residual
    if args.use_depthwise is not None:
        config.MODEL_CONFIG["use_depthwise"] = args.use_depthwise

    if args.batch_size is not None:
        config.TRAINING_CONFIG["batch_size"] = args.batch_size
    if args.epochs is not None:
        config.TRAINING_CONFIG["num_epochs"] = args.epochs
    if args.learning_rate is not None:
        config.OPTIMIZER_CONFIG["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        config.OPTIMIZER_CONFIG["weight_decay"] = args.weight_decay
    if args.optimizer is not None:
        config.OPTIMIZER_CONFIG["type"] = args.optimizer
    if args.loss is not None:
        config.LOSS_CONFIG["type"] = args.loss
    if args.num_workers is not None:
        config.DATASET_CONFIG["num_workers"] = args.num_workers
    if args.max_length_sec is not None:
        config.DATASET_CONFIG["max_length_samples"] = int(args.max_length_sec * config.SAMPLE_RATE)
    if args.max_length_samples is not None:
        config.DATASET_CONFIG["max_length_samples"] = args.max_length_samples
    if args.no_augmentation:
        config.TRAINING_CONFIG["use_augmentation"] = False
    if args.save_interval is not None:
        config.TRAINING_CONFIG["save_every_n_epochs"] = args.save_interval

    if args.device is not None:
        if args.device == "auto":
            config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            config.DEVICE = torch.device(args.device)

    if args.seed is not None:
        config.RANDOM_SEED = args.seed
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.no_mixed_precision:
        config.TRAINING_CONFIG["use_mixed_precision"] = False
    if args.no_tensorboard:
        config.TRAINING_CONFIG["use_tensorboard"] = False
    if args.log_interval is not None:
        config.TRAINING_CONFIG["log_interval"] = args.log_interval
    if args.dataset is not None:
        config.DATASET_CONFIG["dataset"] = args.dataset
    if args.use_vad_labels:
        config.DATASET_CONFIG["use_vad_labels"] = True
    if args.vad_label_set is not None:
        config.DATASET_CONFIG["vad_label_set"] = args.vad_label_set
    if args.vad_soft_mask_floor is not None:
        config.DATASET_CONFIG["vad_soft_mask_floor"] = args.vad_soft_mask_floor

    if args.data_dir is not None:
        config.DATA_ROOT = Path(args.data_dir)
        config.VOICEBANK_ROOT = config.DATA_ROOT / "voicebank_demand"
        config.LIBRISPEECH_ROOT = config.DATA_ROOT / "librispeech_demand"
        config.LIBRISPEECH_VAD_ROOT = config.LIBRISPEECH_ROOT / "vad_labels"

    if args.checkpoint_dir is not None:
        config.CHECKPOINTS_DIR = Path(args.checkpoint_dir)
    if args.log_dir is not None:
        config.LOGS_DIR = Path(args.log_dir)


def load_config_file(config_file: str):
    """Load a saved configuration snapshot before applying CLI overrides."""
    with open(config_file, "r", encoding="utf-8") as f:
        loaded_config = json.load(f)

    model_variant = loaded_config.get("model_variant")
    if model_variant:
        config.apply_preset(model_variant, config_type="model")

    config.MODEL_CONFIG.update(loaded_config.get("model", {}))
    config.OPTIMIZER_CONFIG.update(loaded_config.get("optimizer", {}))
    config.TRAINING_CONFIG.update(loaded_config.get("training", {}))
    config.DATASET_CONFIG.update(loaded_config.get("dataset", {}))
    config.LOSS_CONFIG.update(loaded_config.get("loss", {}))


def validate_dataset_paths(dataset_name: str):
    """Fail fast when the selected dataset layout is incomplete."""
    if dataset_name != "librispeech":
        raise ValueError(
            f"Training and validation use only LibriSpeech-DEMAND. "
            f"'{dataset_name}' is reserved for benchmarking/evaluation."
        )

    if dataset_name == "librispeech":
        required_paths = [
            config.LIBRISPEECH_ROOT / "clean" / "train" / "train-clean-100",
            config.LIBRISPEECH_ROOT / "noise" / "train",
            config.LIBRISPEECH_ROOT / "clean" / "val" / "dev-clean",
            config.LIBRISPEECH_ROOT / "noise" / "val",
        ]

    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        lines = "\n".join(f" - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing dataset directories for '{dataset_name}':\n{lines}")


def resolve_dataset_dirs(dataset_name: str, logger):
    """Resolve train/val data directories and optional LibriSpeech VAD roots."""
    if dataset_name != "librispeech":
        raise ValueError(
            f"Training and validation use only LibriSpeech-DEMAND. "
            f"'{dataset_name}' is reserved for benchmarking/evaluation."
        )

    train_vad_dir = None
    val_vad_dir = None

    if dataset_name == "librispeech":
        train_clean_dir = config.LIBRISPEECH_ROOT / "clean" / "train" / "train-clean-100"
        train_noise_dir = config.LIBRISPEECH_ROOT / "noise" / "train"
        val_clean_dir = config.LIBRISPEECH_ROOT / "clean" / "val" / "dev-clean"
        val_noise_dir = config.LIBRISPEECH_ROOT / "noise" / "val"
        if config.DATASET_CONFIG.get("use_vad_labels", False):
            vad_label_set = config.DATASET_CONFIG.get("vad_label_set", "adaptive_v1")
            train_vad_dir = config.LIBRISPEECH_VAD_ROOT / vad_label_set / "train"
            val_vad_dir = config.LIBRISPEECH_VAD_ROOT / vad_label_set / "val"
            logger.info(f"Using VAD labels from: {train_vad_dir}")

    return {
        "train_clean_dir": train_clean_dir,
        "train_noise_dir": train_noise_dir,
        "val_clean_dir": val_clean_dir,
        "val_noise_dir": val_noise_dir,
        "train_vad_dir": train_vad_dir,
        "val_vad_dir": val_vad_dir,
    }


def clip_gradients(model: nn.Module):
    """Clip gradients if enabled in config and return the pre-clip norm."""
    clip_value = config.TRAINING_CONFIG.get("gradient_clip_value")
    norm_type = config.TRAINING_CONFIG.get("gradient_clip_norm_type", 2.0)

    if clip_value is None or clip_value <= 0:
        return None
    return torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value, norm_type=norm_type)


def build_optimizer(model: nn.Module):
    """Build optimizer from active config."""
    optimizer_type = config.OPTIMIZER_CONFIG["type"]
    lr = config.OPTIMIZER_CONFIG["learning_rate"]

    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(config.OPTIMIZER_CONFIG["beta1"], config.OPTIMIZER_CONFIG["beta2"]),
            eps=config.OPTIMIZER_CONFIG["eps"],
            weight_decay=config.OPTIMIZER_CONFIG["weight_decay"],
        )
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(config.OPTIMIZER_CONFIG["beta1"], config.OPTIMIZER_CONFIG["beta2"]),
            eps=config.OPTIMIZER_CONFIG["eps"],
            weight_decay=config.OPTIMIZER_CONFIG["weight_decay"],
        )
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.OPTIMIZER_CONFIG["momentum"],
            weight_decay=config.OPTIMIZER_CONFIG["weight_decay"],
            nesterov=config.OPTIMIZER_CONFIG["nesterov"],
        )
    if optimizer_type == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=config.OPTIMIZER_CONFIG.get("alpha", 0.99),
            momentum=config.OPTIMIZER_CONFIG.get("momentum", 0.0),
            eps=config.OPTIMIZER_CONFIG["eps"],
            weight_decay=config.OPTIMIZER_CONFIG["weight_decay"],
        )
    raise ValueError(f"Unknown optimizer: {optimizer_type}")


def build_loss_function():
    """Build loss criterion from active config."""
    loss_type = config.LOSS_CONFIG["type"]

    if loss_type == "mse":
        return loss_type, nn.MSELoss()
    if loss_type == "mae":
        return loss_type, nn.L1Loss()
    if loss_type == "huber":
        return loss_type, nn.SmoothL1Loss(beta=config.LOSS_CONFIG.get("huber_delta", 1.0))
    if loss_type == "combined":
        alpha = config.LOSS_CONFIG.get("alpha", 0.5)
        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        return loss_type, lambda pred, target: alpha * mse_loss(pred, target) + (1 - alpha) * mae_loss(pred, target)
    if loss_type == "weighted_combined":
        alpha = config.LOSS_CONFIG.get("alpha", 0.5)
        speech_weight = config.LOSS_CONFIG.get("speech_weight", 2.0)
        speech_threshold = config.LOSS_CONFIG.get("speech_threshold", 0.6)

        def criterion(pred, target):
            weights = torch.ones_like(target)
            weights = torch.where(target >= speech_threshold, weights * speech_weight, weights)
            mse_term = ((pred - target) ** 2) * weights
            mae_term = torch.abs(pred - target) * weights
            return alpha * mse_term.mean() + (1 - alpha) * mae_term.mean()

        return loss_type, criterion
    raise ValueError(f"Unknown loss: {loss_type}")


def build_scheduler(optimizer, steps_per_epoch: int):
    """Build LR scheduler from active config."""
    scheduler_type = config.TRAINING_CONFIG.get("scheduler_type", "ReduceLROnPlateau")

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.TRAINING_CONFIG.get("scheduler_factor", 0.5),
            patience=config.TRAINING_CONFIG.get("scheduler_patience", 5),
            verbose=True,
            min_lr=config.TRAINING_CONFIG.get("scheduler_min_lr", 1e-6),
        )
        description = (
            "ReduceLROnPlateau "
            f"(factor={config.TRAINING_CONFIG.get('scheduler_factor', 0.5)}, "
            f"patience={config.TRAINING_CONFIG.get('scheduler_patience', 5)})"
        )
    elif scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.TRAINING_CONFIG.get("step_size", 20),
            gamma=config.TRAINING_CONFIG.get("gamma", 0.5),
        )
        description = (
            "StepLR "
            f"(step_size={config.TRAINING_CONFIG.get('step_size', 20)}, "
            f"gamma={config.TRAINING_CONFIG.get('gamma', 0.5)})"
        )
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.TRAINING_CONFIG.get("T_max", config.TRAINING_CONFIG["num_epochs"]),
            eta_min=config.TRAINING_CONFIG.get("eta_min", 1e-6),
        )
        description = (
            "CosineAnnealingLR "
            f"(T_max={config.TRAINING_CONFIG.get('T_max', config.TRAINING_CONFIG['num_epochs'])}, "
            f"eta_min={config.TRAINING_CONFIG.get('eta_min', 1e-6)})"
        )
    elif scheduler_type == "OneCycleLR":
        if steps_per_epoch <= 0:
            raise ValueError("OneCycleLR requires at least one training batch per epoch")
        total_steps = config.TRAINING_CONFIG["num_epochs"] * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.TRAINING_CONFIG.get("max_lr", config.OPTIMIZER_CONFIG["learning_rate"]),
            total_steps=total_steps + 1,
            pct_start=config.TRAINING_CONFIG.get("pct_start", 0.3),
            anneal_strategy=config.TRAINING_CONFIG.get("anneal_strategy", "cos"),
        )
        description = (
            "OneCycleLR "
            f"(max_lr={config.TRAINING_CONFIG.get('max_lr', config.OPTIMIZER_CONFIG['learning_rate'])}, "
            f"pct_start={config.TRAINING_CONFIG.get('pct_start', 0.3)}, "
            f"anneal_strategy={config.TRAINING_CONFIG.get('anneal_strategy', 'cos')}, "
            f"steps_per_epoch={steps_per_epoch}, "
            f"total_steps={total_steps + 1})"
        )
    else:
        scheduler = None
        description = "None"
    return scheduler_type, scheduler, description


def step_scheduler_batch(scheduler, scheduler_type: str):
    """Step schedulers that update on each optimizer step."""
    if scheduler is not None and scheduler_type == "OneCycleLR":
        try:
            scheduler.step()
        except ValueError as exc:
            if "Tried to step" not in str(exc):
                raise
