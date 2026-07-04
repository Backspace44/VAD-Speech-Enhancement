from copy import deepcopy
from pathlib import Path
import torch
from datetime import datetime






ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
VOICEBANK_ROOT = DATA_ROOT / "voicebank_demand"
LIBRISPEECH_ROOT = DATA_ROOT / "librispeech_demand"
LIBRISPEECH_VAD_ROOT = LIBRISPEECH_ROOT / "vad_labels"


CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"
PLOTS_DIR = RESULTS_DIR / "plots"
REALTIME_RECORDINGS_DIR = RESULTS_DIR / "realtime_recordings"


def get_experiment_dir(experiment_name: str = None):
    """Get experiment directory with timestamp."""
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = CHECKPOINTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


TRAINING_LOG_FILE = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
EVAL_LOG_FILE = LOGS_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


for directory in [CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, TENSORBOARD_DIR, PLOTS_DIR, REALTIME_RECORDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)






SAMPLE_RATE = 16000
FRAME_LEN_MS = 25.0
HOP_LEN_MS = 10.0


FRAME_LEN = int(SAMPLE_RATE * FRAME_LEN_MS / 1000.0)
HOP_LEN = int(SAMPLE_RATE * HOP_LEN_MS / 1000.0)


N_FFT = 512
N_FREQ_BINS = N_FFT // 2 + 1


MIN_AUDIO_LENGTH_SEC = 1.0
MAX_AUDIO_LENGTH_SEC = 10.0
MIN_AUDIO_SAMPLES = int(MIN_AUDIO_LENGTH_SEC * SAMPLE_RATE)
MAX_AUDIO_SAMPLES = int(MAX_AUDIO_LENGTH_SEC * SAMPLE_RATE)


SNR_LEVELS = [0, 5, 10, 15, 20]






DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False





DATASET_CONFIG = {

    "dataset": "librispeech",
    

    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    
    # VAD labels
    "use_vad_labels": False,  # apply VAD mask to IRM
    "vad_label_set": "adaptive_v1",
    "vad_soft_mask_floor": 0.15,
    "on_the_fly_snr_range": [0.0, 20.0],

    "num_workers": 0,  # Windows-safe default
    "pin_memory": True,
    "persistent_workers": False,  # requires num_workers > 0
    

    "pad_mode": "constant",
    "truncate_long_audio": True,
    "max_length_samples": MAX_AUDIO_SAMPLES,
    

    "cache_spectrograms": False,
    "cache_audio": False,
}






MODEL_CONFIG = {
    "in_channels": 1,
    "base_channels": 16,  # fast default
    "use_lstm": True,
    "use_attention": False,  # faster
    "use_residual": False,
    "use_depthwise": False,
    "output_channels": 1,
    "output_activation": "sigmoid",
    "output_scale": 1.0,
    "mask_type": "magnitude",
    "complex_mask_clip": 5.0,
}


MODEL_VARIANTS = {
    "tiny": {
        "base_channels": 16,
        "use_lstm": False,
        "use_attention": False,
        "use_residual": False,
        "description": "Lightweight model (~200K params) for fast inference",
        "expected_params": 200_000,
    },

    "tiny_fast": {
        "base_channels": 16,
        "use_lstm": False,
        "use_attention": False,
        "use_residual": False,
        "use_depthwise": True,
        "description": "Depthwise-separable tiny model for faster long-sequence training",
        "expected_params": 650_000,
    },
    
    "small": {
        "base_channels": 24,
        "use_lstm": True,
        "use_attention": False,
        "use_residual": False,
        "use_depthwise": False,
        "description": "Small model (~800K params) with LSTM",
        "expected_params": 800_000,
    },
    
    "balanced": {
        "base_channels": 32,
        "use_lstm": True,
        "use_attention": True,
        "use_residual": False,
        "use_depthwise": False,
        "description": "Balanced baseline model (~2.5M params)",
        "expected_params": 2_500_000,
    },

    "balanced_res": {
        "base_channels": 32,
        "use_lstm": True,
        "use_attention": True,
        "use_residual": True,
        "use_depthwise": False,
        "description": "Balanced residual model (~2.9M params) - STAGE 2 RECOMMENDED",
        "expected_params": 2_900_000,
    },

    "complex_balanced_res": {
        "in_channels": 2,
        "base_channels": 32,
        "use_lstm": True,
        "use_attention": True,
        "use_residual": True,
        "use_depthwise": False,
        "output_channels": 2,
        "output_activation": "tanh",
        "output_scale": 5.0,
        "mask_type": "complex",
        "complex_mask_clip": 5.0,
        "description": "Complex MaskNet with residual U-Net blocks for real/imaginary CRM estimation",
        "expected_params": 2_950_000,
    },
    
    "large": {
        "base_channels": 48,
        "use_lstm": True,
        "use_attention": True,
        "use_residual": False,
        "use_depthwise": False,
        "description": "Large model (~5M params) for best quality",
        "expected_params": 5_000_000,
    },
    
    "xlarge": {
        "base_channels": 64,
        "use_lstm": True,
        "use_attention": True,
        "use_residual": False,
        "use_depthwise": False,
        "description": "Extra large model (~10M params) for research",
        "expected_params": 10_000_000,
    },
}


ACTIVE_MODEL_VARIANT = "tiny"  # fast default


def set_active_model_variant(variant_name: str):
    """Apply a model variant and track it as the active one."""
    global ACTIVE_MODEL_VARIANT

    if variant_name not in MODEL_VARIANTS:
        available = ", ".join(MODEL_VARIANTS.keys())
        raise ValueError(f"Unknown model variant: {variant_name}. Available: {available}")

    ACTIVE_MODEL_VARIANT = variant_name
    variant = MODEL_VARIANTS[variant_name]
    MODEL_CONFIG.update({
        "in_channels": variant.get("in_channels", 1),
        "base_channels": variant["base_channels"],
        "use_lstm": variant["use_lstm"],
        "use_attention": variant["use_attention"],
        "use_residual": variant.get("use_residual", False),
        "use_depthwise": variant.get("use_depthwise", False),
        "output_channels": variant.get("output_channels", 1),
        "output_activation": variant.get("output_activation", "sigmoid"),
        "output_scale": variant.get("output_scale", 1.0),
        "mask_type": variant.get("mask_type", "magnitude"),
        "complex_mask_clip": variant.get("complex_mask_clip", 5.0),
    })


set_active_model_variant(ACTIVE_MODEL_VARIANT)





OPTIMIZER_CONFIG = {
    "type": "adam",
    

    "learning_rate": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 1e-5,
    "amsgrad": False,
    

    "momentum": 0.9,
    "nesterov": True,
    

    "alpha": 0.99,
}


OPTIMIZER_PRESETS = {
    "default": {
        "type": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
    },
    
    "aggressive": {
        "type": "adam",
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "description": "Higher LR for faster convergence (may be unstable)",
    },
    
    "conservative": {
        "type": "adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "description": "Lower LR for stable training",
    },
    
    "adamw": {
        "type": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "description": "AdamW optimizer with decoupled weight decay",
    },
    
    "sgd": {
        "type": "sgd",
        "learning_rate": 1e-2,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 1e-4,
        "description": "SGD with momentum (requires careful tuning)",
    },

    "rmsprop": {
        "type": "rmsprop",
        "learning_rate": 5e-4,
        "alpha": 0.99,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "description": "RMSprop optimizer for smoother adaptive updates",
    },
}






TRAINING_CONFIG = {

    "batch_size": 8,  # GTX 1050
    "num_epochs": 1,  # demo run
    "num_workers": 0,  # Windows-safe default
    "val_ratio": 0.1,
    "use_augmentation": True,
    
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,

    "warmup_epochs": 2,
    "warmup_start_lr": 1e-6,
    

    "scheduler_type": "ReduceLROnPlateau",
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,
    

    "step_size": 10,
    "gamma": 0.5,
    

    "T_max": 50,
    "eta_min": 1e-6,
    

    "max_lr": 1e-2,
    "pct_start": 0.3,
    "anneal_strategy": "cos",
    

    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-4,
    "early_stopping_mode": "min",
    

    "gradient_clip_value": 5.0,
    "gradient_clip_norm_type": 2.0,
    

    "use_mixed_precision": torch.cuda.is_available(),
    "amp_opt_level": "O1",
    

    "save_every_n_epochs": 5,
    "save_best_only": False,
    "save_last": True,
    "max_checkpoints_to_keep": 5,
    

    "log_interval": 10,
    "use_tensorboard": True,
    "tensorboard_log_histograms": True,
    "tensorboard_log_interval": 100,
    

    "val_interval": 1,
    "val_batch_size": None,
    

    "resume_from_checkpoint": None,
    "load_optimizer_state": True,
    "load_scheduler_state": True,
}


TRAINING_PRESETS = {
    "quick_test": {
        "batch_size": 2,
        "num_epochs": 5,
        "save_every_n_epochs": 1,
        "early_stopping_patience": 3,
        "description": "Quick test run (5 epochs)",
    },
    
    "development": {
        "batch_size": 4,
        "num_epochs": 50,
        "save_every_n_epochs": 5,
        "early_stopping_patience": 10,
        "description": "Development training (50 epochs)",
    },
    
    "production": {
        "batch_size": 8,
        "num_epochs": 100,
        "save_every_n_epochs": 5,
        "early_stopping_patience": 15,
        "use_mixed_precision": True,
        "description": "Production training (100 epochs, full optimization)",
    },
    
    "long": {
        "batch_size": 8,
        "num_epochs": 200,
        "save_every_n_epochs": 10,
        "early_stopping_patience": 25,
        "use_mixed_precision": True,
        "description": "Long training run (200 epochs)",
    },
    
    "debug": {
        "batch_size": 2,
        "num_epochs": 2,
        "save_every_n_epochs": 1,
        "log_interval": 1,
        "description": "Debug mode (2 epochs, verbose logging)",
    },

    "onecycle_test": {
        "batch_size": 4,
        "num_epochs": 3,
        "scheduler_type": "OneCycleLR",
        "max_lr": 3e-3,
        "pct_start": 0.2,
        "anneal_strategy": "cos",
        "early_stopping_patience": 3,
        "description": "Short OneCycleLR run for stage-2 optimizer/scheduler experiments",
    },

    "stage2_recommended": {
        "batch_size": 4,
        "num_epochs": 10,
        "scheduler_type": "OneCycleLR",
        "max_lr": 3e-3,
        "pct_start": 0.2,
        "anneal_strategy": "cos",
        "early_stopping_patience": 5,
        "gradient_clip_value": 5.0,
        "description": "Recommended stage-2 training setup for residual MaskNet experiments",
    },
}


EXPERIMENT_PRESETS = {
    "stage2_recommended": {
        "model": "balanced_res",
        "optimizer": "adamw",
        "training": "stage2_recommended",
        "loss": "mse",
        "description": "Balanced residual MaskNet + AdamW + OneCycleLR",
    },
    "librispeech_soft_vad_recommended": {
        "model": "balanced_res",
        "optimizer": "adamw",
        "training": "onecycle_test",
        "loss": "mse",
        "dataset": {
            "dataset": "librispeech",
            "use_vad_labels": True,
            "vad_label_set": "adaptive_soft_v1",
            "vad_soft_mask_floor": 0.15,
        },
        "description": "Recommended LibriSpeech setup with soft VAD gating + balanced_res + AdamW + OneCycleLR",
    },
    "librispeech_complex_masknet_recommended": {
        "model": "complex_balanced_res",
        "optimizer": "adamw",
        "training": "stage2_recommended",
        "loss": "mse",
        "dataset": {
            "dataset": "librispeech",
            "use_vad_labels": True,
            "vad_label_set": "adaptive_soft_v1",
            "vad_soft_mask_floor": 0.15,
        },
        "description": "Complex MaskNet on LibriSpeech + soft VAD gating + AdamW + OneCycleLR",
    },
    "librispeech_fast_benchmark": {
        "model": "tiny_fast",
        "optimizer": "adamw",
        "training": "onecycle_test",
        "loss": "mse",
        "dataset": {
            "dataset": "librispeech",
            "use_vad_labels": True,
            "vad_label_set": "adaptive_soft_v1",
            "vad_soft_mask_floor": 0.15,
            "max_length_samples": SAMPLE_RATE,
        },
        "description": "Fast LibriSpeech benchmark: tiny_fast + adaptive_soft_v1 + 1s crops",
    },
}





LOSS_CONFIG = {
    "type": "mse",
    "alpha": 0.5,
    "speech_weight": 2.0,
    "speech_threshold": 0.6,
    

    "huber_delta": 1.0,
    

    "sisdr_eps": 1e-8,
    

    "spectral_alpha": 0.5,
    "use_log_spectral": True,
}


LOSS_PRESETS = {
    "mse": {
        "type": "mse",
        "description": "Mean Squared Error - standard choice",
    },
    
    "mae": {
        "type": "mae",
        "description": "Mean Absolute Error - robust to outliers",
    },
    
    "huber": {
        "type": "huber",
        "huber_delta": 1.0,
        "description": "Huber loss - combines MSE and MAE benefits",
    },
    
    "combined": {
        "type": "combined",
        "alpha": 0.5,
        "description": "Weighted combination of MSE and MAE",
    },

    "weighted_combined": {
        "type": "weighted_combined",
        "alpha": 0.5,
        "speech_weight": 2.0,
        "speech_threshold": 0.6,
        "description": "Combined loss with extra weight on speech-dominant mask regions",
    },

    "weighted_combined_tuned": {
        "type": "weighted_combined",
        "alpha": 0.7,
        "speech_weight": 1.2,
        "speech_threshold": 0.9,
        "description": "Best short-run weighted loss variant from stage-2 tuning",
    },
    
    "aggressive_combined": {
        "type": "combined",
        "alpha": 0.7,
        "description": "MSE-heavy combined loss for faster convergence",
    },
}


AUGMENTATION_CONFIG = {
    "enabled": False,  # fast training
    

    "random_gain": {
        "prob": 0.5,
        "min_gain_db": -6,
        "max_gain_db": 6,
    },
    
    "add_noise": {
        "prob": 0.3,
        "snr_db_range": [5, 20],
        "noise_type": "white",
    },
    
    "time_stretch": {
        "prob": 0.3,
        "rate_range": [0.9, 1.1],
    },
    
    "pitch_shift": {
        "prob": 0.2,
        "semitone_range": [-2, 2],
    },
    
    "apply_reverb": {
        "prob": 0.3,
        "room_size": [0.1, 0.5],
        "damping": [0.3, 0.7],
        "wet_dry_mix": 0.3,
    },
    
    "random_eq": {
        "prob": 0.4,
        "bands": 3,
        "gain_range_db": [-6, 6],
    },
    
    "dynamic_range_compression": {
        "prob": 0.3,
        "threshold_db": -20,
        "ratio": 4,
        "attack": 0.005,
        "release": 0.1,
    },
    
    "add_clipping": {
        "prob": 0.2,
        "threshold_range": [0.7, 0.95],
    },
}


SPECAUGMENT_CONFIG = {
    "enabled": False,  # faster training
    "prob": 0.5,
    "freq_mask_param": 15,
    "time_mask_param": 25,
    "num_freq_masks": 1,
    "num_time_masks": 1,
}


MIXUP_CONFIG = {
    "enabled": True,
    "alpha": 0.2,
    "prob": 0.3,
}


PREPROCESSING_CONFIG = {

    "audio_preprocessing": {
        "enabled": True,
        "normalize": True,
        "normalization_method": "peak",
        "target_level": -3.0,
        "remove_dc": True,
        "preemphasis": False,
        "preemphasis_coef": 0.97,
        "apply_agc": False,
        "trim_silence": True,
        "silence_threshold": 0.01,
    },
    

    "spectrogram_normalization": {
        "enabled": True,
        "method": "per_channel",
        "stats_type": "mean_std",
        "epsilon": 1e-8,
        "clip_percentile": None,
    },
    

    "log_scale": {
        "enabled": True,
        "scale": "db",
        "ref": 1.0,
        "amin": 1e-10,
        "top_db": 80.0,
    },
    

    "compute_stats": True,
    "save_stats": True,
    "stats_file": "normalization_stats.npz",
}





EVAL_CONFIG = {
    "compute_stoi": True,
    "compute_pesq": True,
    "save_examples": True,
    "max_examples": 10,
    

    "enable_training_metrics": True,
    "training_metrics_every_n_epochs": 5,
    "num_training_eval_samples": 10,
    

    "metrics_weights": {
        "val_pesq": 0.4,
        "val_stoi": 0.3,
        "val_snr": 0.2,
        "val_composite": 0.1,
    },
    

    "stratified_eval": True,
    "statistical_tests": True,
    "save_audio_samples": True,
    

    "eval_batch_size": 1,
    "save_spectrograms": True,
}





INFERENCE_CONFIG = {

    "checkpoint_path": None,
    "load_best": True,
    

    "input_path": None,
    "output_path": None,
    "save_spectrograms": False,
    

    "batch_size": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_mixed_precision": False,
    

    "normalize_input": True,
    "denormalize_output": True,
    "apply_preprocessing": True,
    

    "output_sample_rate": 16000,
    "output_bit_depth": 16,
    "output_format": "wav",
    

    "compute_metrics": True,
    "visualize": False,
}





EXPERIMENT_CONFIG = {

    "experiment_name": None,
    "description": "",
    "tags": [],
    

    "track_code_version": True,
    "track_system_info": True,
    "track_hyperparameters": True,
    

    "use_tensorboard": True,
    "use_wandb": False,
    "use_mlflow": False,
    

    "wandb_project": "speech-enhancement",
    "wandb_entity": None,
    "wandb_tags": [],
    

    "mlflow_tracking_uri": None,
    "mlflow_experiment_name": "speech-enhancement",
    

    "save_config": True,
    "save_model_architecture": True,
    "save_training_curves": True,
}





DISTRIBUTED_CONFIG = {

    "enabled": False,
    

    "backend": "nccl",
    "init_method": "env://",
    "world_size": NUM_GPUS,
    "rank": 0,
    

    "use_dataparallel": False,
    "use_distributeddataparallel": False,
    

    "sync_batchnorm": True,
    "find_unused_parameters": False,
}





def get_active_config():
    """Get the currently active configuration as a dictionary."""
    return {
        "paths": {
            "root": ".",
            "data_root": "data",
            "checkpoints": "checkpoints",
            "results": "results",
            "logs": "logs",
            "tensorboard": "logs/tensorboard",
        },
        "device": str(DEVICE),
        "num_gpus": NUM_GPUS,
        "model": deepcopy(MODEL_CONFIG),
        "model_variant": ACTIVE_MODEL_VARIANT,
        "optimizer": deepcopy(OPTIMIZER_CONFIG),
        "training": deepcopy(TRAINING_CONFIG),
        "dataset": deepcopy(DATASET_CONFIG),
        "loss": deepcopy(LOSS_CONFIG),
        "augmentation": deepcopy(AUGMENTATION_CONFIG),
        "preprocessing": deepcopy(PREPROCESSING_CONFIG),
        "evaluation": deepcopy(EVAL_CONFIG),
        "inference": deepcopy(INFERENCE_CONFIG),
        "experiment": deepcopy(EXPERIMENT_CONFIG),
    }


def apply_preset(preset_name: str, config_type: str = "training"):
    """Apply a named preset."""
    if config_type == "training":
        if preset_name in TRAINING_PRESETS:
            TRAINING_CONFIG.update(TRAINING_PRESETS[preset_name])
            print(f"Applied training preset: {preset_name}")
            if "description" in TRAINING_PRESETS[preset_name]:
                print(f"  {TRAINING_PRESETS[preset_name]['description']}")
        else:
            available = ", ".join(TRAINING_PRESETS.keys())
            print(f"Unknown preset: {preset_name}. Available: {available}")
    
    elif config_type == "model":
        if preset_name in MODEL_VARIANTS:
            set_active_model_variant(preset_name)
            variant = MODEL_VARIANTS[preset_name]
            print(f"Applied model variant: {preset_name}")
            print(f"  {variant['description']}")
            print(f"  Expected params: ~{variant['expected_params']:,}")
        else:
            available = ", ".join(MODEL_VARIANTS.keys())
            print(f"Unknown variant: {preset_name}. Available: {available}")
    
    elif config_type == "optimizer":
        if preset_name in OPTIMIZER_PRESETS:
            OPTIMIZER_CONFIG.update(OPTIMIZER_PRESETS[preset_name])
            print(f"Applied optimizer preset: {preset_name}")
            if "description" in OPTIMIZER_PRESETS[preset_name]:
                print(f"  {OPTIMIZER_PRESETS[preset_name]['description']}")
        else:
            available = ", ".join(OPTIMIZER_PRESETS.keys())
            print(f"Unknown preset: {preset_name}. Available: {available}")
    
    elif config_type == "loss":
        if preset_name in LOSS_PRESETS:
            LOSS_CONFIG.update(LOSS_PRESETS[preset_name])
            print(f"Applied loss preset: {preset_name}")
            if "description" in LOSS_PRESETS[preset_name]:
                print(f"  {LOSS_PRESETS[preset_name]['description']}")
        else:
            available = ", ".join(LOSS_PRESETS.keys())
            print(f"Unknown preset: {preset_name}. Available: {available}")

    elif config_type == "experiment":
        if preset_name in EXPERIMENT_PRESETS:
            preset = EXPERIMENT_PRESETS[preset_name]
            if "model" in preset:
                apply_preset(preset["model"], config_type="model")
            if "optimizer" in preset:
                apply_preset(preset["optimizer"], config_type="optimizer")
            if "training" in preset:
                apply_preset(preset["training"], config_type="training")
            if "loss" in preset:
                apply_preset(preset["loss"], config_type="loss")
            if "dataset" in preset:
                DATASET_CONFIG.update(preset["dataset"])

            print(f"Applied experiment preset: {preset_name}")
            if "description" in preset:
                print(f"  {preset['description']}")
        else:
            available = ", ".join(EXPERIMENT_PRESETS.keys())
            print(f"Unknown preset: {preset_name}. Available: {available}")


def print_config_summary():
    """Print a summary of the active configuration."""
    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print(f"\nPaths:")
    print(f"  Root: {ROOT}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    print(f"  TensorBoard: {TENSORBOARD_DIR}")
    
    print(f"\nSystem:")
    print(f"  Device: {DEVICE}")
    print(f"  GPUs: {NUM_GPUS}")
    print(f"  Mixed Precision: {TRAINING_CONFIG['use_mixed_precision']}")
    
    print(f"\nModel:")
    print(f"  Variant: {ACTIVE_MODEL_VARIANT}")
    if ACTIVE_MODEL_VARIANT in MODEL_VARIANTS:
        variant = MODEL_VARIANTS[ACTIVE_MODEL_VARIANT]
        print(f"  Description: {variant['description']}")
        print(f"  Input Channels: {MODEL_CONFIG['in_channels']}")
        print(f"  Output Channels: {MODEL_CONFIG['output_channels']}")
        print(f"  Base Channels: {MODEL_CONFIG['base_channels']}")
        print(f"  LSTM: {MODEL_CONFIG['use_lstm']}")
        print(f"  Attention: {MODEL_CONFIG['use_attention']}")
        print(f"  Residual Blocks: {MODEL_CONFIG.get('use_residual', False)}")
        print(f"  Depthwise Blocks: {MODEL_CONFIG.get('use_depthwise', False)}")
        print(f"  Mask Type: {MODEL_CONFIG.get('mask_type', 'magnitude')}")
        print(f"  Output Activation: {MODEL_CONFIG.get('output_activation', 'sigmoid')}")
        print(f"  Expected Params: ~{variant['expected_params']:,}")
    
    print(f"\nTraining:")
    print(f"  Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  Optimizer: {OPTIMIZER_CONFIG['type'].upper()}")
    print(f"  Learning Rate: {OPTIMIZER_CONFIG['learning_rate']}")
    print(f"  Scheduler: {TRAINING_CONFIG['scheduler_type']}")
    print(f"  Early Stopping: {TRAINING_CONFIG['early_stopping_patience']} epochs")
    
    print(f"\nData:")
    print(f"  Dataset: {DATASET_CONFIG['dataset']}")
    print(f"  Workers: {DATASET_CONFIG['num_workers']}")
    print(f"  Max Length Samples: {DATASET_CONFIG['max_length_samples']}")
    print(f"  Use VAD Labels: {DATASET_CONFIG.get('use_vad_labels', False)}")
    if DATASET_CONFIG.get('use_vad_labels', False):
        print(f"  VAD Label Set: {DATASET_CONFIG.get('vad_label_set', 'N/A')}")
        print(f"  VAD Soft Mask Floor: {DATASET_CONFIG.get('vad_soft_mask_floor', 0.0)}")
    print(f"  Augmentation: {AUGMENTATION_CONFIG['enabled']}")
    print(f"  Preprocessing: {PREPROCESSING_CONFIG['audio_preprocessing']['enabled']}")
    
    print(f"\nLoss:")
    print(f"  Type: {LOSS_CONFIG['type'].upper()}")
    if LOSS_CONFIG['type'] == 'combined':
        print(f"  Alpha: {LOSS_CONFIG['alpha']}")
    
    print(f"\nEvaluation:")
    print(f"  PESQ: {EVAL_CONFIG['compute_pesq']}")
    print(f"  STOI: {EVAL_CONFIG['compute_stoi']}")
    print(f"  Training Metrics: {EVAL_CONFIG['enable_training_metrics']}")
    
    print("=" * 70)




