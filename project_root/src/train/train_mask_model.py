"""
Training script for speech enhancement model.
Usage: python train_mask_model.py --help
"""

import argparse
import sys
import json
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler

from src import config
from src.models.mask_model import MaskNet
from src.train.checkpointing import finalize_experiment
from src.train.train_loop import run_training
from src.train.train_setup import (
    apply_args_to_config,
    build_loss_function,
    build_optimizer,
    build_scheduler,
    load_config_file,
    resolve_dataset_dirs,
    validate_dataset_paths,
)
from src.utils.logger import setup_logger, TrainingLogger
from src.data_prep.dataset import create_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train speech enhancement model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model-variant',
        type=str,
        default=None,
        choices=sorted(config.MODEL_VARIANTS.keys()),
        help='Model architecture variant'
    )
    model_group.add_argument(
        '--base-channels',
        type=int,
        default=None,
        help='Base channels (overrides variant)'
    )
    model_group.add_argument(
        '--use-lstm',
        action='store_true',
        default=None,
        help='Enable LSTM layers'
    )
    model_group.add_argument(
        '--use-attention',
        action='store_true',
        default=None,
        help='Enable attention mechanisms'
    )
    model_group.add_argument(
        '--use-residual',
        action='store_true',
        default=None,
        help='Enable residual blocks in the model'
    )
    model_group.add_argument(
        '--use-depthwise',
        action='store_true',
        default=None,
        help='Enable depthwise-separable convolution blocks for faster throughput'
    )

    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument(
        '--preset',
        type=str,
        default=None,
        choices=['quick_test', 'development', 'production', 'long', 'debug', 'onecycle_test', 'stage2_recommended'],
        help='Training preset configuration'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size'
    )
    train_group.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    train_group.add_argument(
        '--learning-rate', '--lr',
        type=float,
        default=None,
        help='Initial learning rate'
    )
    train_group.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help='Weight decay (L2 regularization)'
    )

    optim_group = parser.add_argument_group('Optimizer Configuration')
    optim_group.add_argument(
        '--optimizer',
        type=str,
        default=None,
        choices=['adam', 'adamw', 'sgd', 'rmsprop'],
        help='Optimizer type'
    )
    optim_group.add_argument(
        '--optimizer-preset',
        type=str,
        default=None,
        choices=['default', 'aggressive', 'conservative', 'adamw', 'sgd', 'rmsprop'],
        help='Optimizer preset configuration'
    )

    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument(
        '--loss',
        type=str,
        default=None,
        choices=['mse', 'mae', 'huber', 'combined', 'weighted_combined'],
        help='Loss function type'
    )
    loss_group.add_argument(
        '--loss-preset',
        type=str,
        default=None,
        choices=['mse', 'mae', 'huber', 'combined', 'weighted_combined', 'weighted_combined_tuned', 'aggressive_combined'],
        help='Loss preset configuration'
    )

    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to dataset directory'
    )
    data_group.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['voicebank', 'librispeech'],
        help='Dataset to use'
    )
    data_group.add_argument(
        '--use-vad-labels',
        action='store_true',
        help='Use frame-level VAD labels when training on LibriSpeech'
    )
    data_group.add_argument(
        '--vad-label-set',
        type=str,
        default=None,
        help='Versioned VAD label set to use for LibriSpeech (for example: adaptive_v1)'
    )
    data_group.add_argument(
        '--vad-soft-mask-floor',
        type=float,
        default=None,
        help='Minimum mask scaling used when VAD labels are soft scores (0 disables the floor)'
    )
    data_group.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of data loading workers'
    )
    data_group.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    data_group.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Limit number of training samples (useful for quick tests)'
    )
    data_group.add_argument(
        '--max-val-samples',
        type=int,
        default=None,
        help='Limit number of validation samples'
    )
    data_group.add_argument(
        '--max-length-sec',
        type=float,
        default=None,
        help='Maximum audio crop length in seconds for train/val samples'
    )
    data_group.add_argument(
        '--max-length-samples',
        type=int,
        default=None,
        help='Maximum audio crop length in samples for train/val samples'
    )

    checkpoint_group = parser.add_argument_group('Checkpoint Configuration')
    checkpoint_group.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    checkpoint_group.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )
    checkpoint_group.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name for checkpoint directory'
    )
    checkpoint_group.add_argument(
        '--save-interval',
        type=int,
        default=None,
        help='Save checkpoint every N epochs'
    )

    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'auto'],
        help='Device to use for training'
    )
    system_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    system_group.add_argument(
        '--no-mixed-precision',
        action='store_true',
        help='Disable mixed precision training'
    )

    log_group = parser.add_argument_group('Logging Configuration')
    log_group.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Directory for log files'
    )
    log_group.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    log_group.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Disable TensorBoard logging'
    )
    log_group.add_argument(
        '--log-interval',
        type=int,
        default=None,
        help='Log training metrics every N batches'
    )

    parser.add_argument(
        '--experiment-preset',
        type=str,
        default=None,
        choices=sorted(config.EXPERIMENT_PRESETS.keys()),
        help='Apply a combined model/optimizer/training/loss experiment preset'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Maximum number of batches to train (useful for short test runs)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without training'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default=None,
        help='Load configuration from JSON file'
    )
    parser.add_argument(
        '--save-config',
        type=str,
        default=None,
        help='Save configuration to JSON file and exit'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    if args.config_file:
        load_config_file(args.config_file)

    apply_args_to_config(args)

    if args.save_config:
        config_dict = config.get_active_config()
        with open(args.save_config, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Configuration saved to {args.save_config}")
        return

    if args.dry_run:
        print("\n" + "=" * 70)
        print("SPEECH ENHANCEMENT TRAINING (DRY RUN)")
        print("=" * 70)
        config.print_config_summary()
        print("\nDry run - exiting without creating experiment directories")
        return

    if args.experiment_name:
        exp_dir = config.get_experiment_dir(args.experiment_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"train_{config.ACTIVE_MODEL_VARIANT}_{timestamp}"
        exp_dir = config.get_experiment_dir(exp_name)

    logger = setup_logger(
        name="training",
        log_dir=args.log_dir or config.LOGS_DIR,
        log_file=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=args.log_level
    )
    train_logger = TrainingLogger(logger)

    logger.info("\n" + "=" * 70)
    logger.info("SPEECH ENHANCEMENT TRAINING")
    logger.info("=" * 70)
    config.print_config_summary()

    config_dict = config.get_active_config()
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Configuration saved to {config_path}")

    logger.info("\nInitializing model...")
    model = MaskNet(**config.MODEL_CONFIG).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    logger.info("\nInitializing optimizer...")
    optimizer_type = config.OPTIMIZER_CONFIG['type']
    lr = config.OPTIMIZER_CONFIG['learning_rate']
    optimizer = build_optimizer(model)
    logger.info(f"Optimizer: {optimizer_type.upper()}, LR: {lr}")

    loss_type, criterion = build_loss_function()
    logger.info(f"Loss function: {loss_type.upper()}")

    start_epoch = 0
    resume_scheduler_state = None
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_scheduler_state = checkpoint.get('scheduler_state_dict')
        start_epoch = checkpoint['epoch']
        if 'batch_idx' in checkpoint:
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, batch {checkpoint['batch_idx']}")
            logger.info("NOTE: Training will restart from beginning of this epoch (batch-level resume not implemented)")
        else:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    writer = None
    if config.TRAINING_CONFIG['use_tensorboard'] and not args.no_tensorboard:
        tensorboard_dir = config.TENSORBOARD_DIR / exp_dir.name
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard logging to: {tensorboard_dir}")

    train_logger.log_training_start({
        'model_variant': config.ACTIVE_MODEL_VARIANT,
        'batch_size': config.TRAINING_CONFIG['batch_size'],
        'num_epochs': config.TRAINING_CONFIG['num_epochs'],
        'learning_rate': config.OPTIMIZER_CONFIG['learning_rate'],
        'device': str(config.DEVICE)
    })

    logger.info("\nLoading dataset...")
    dataset_name = config.DATASET_CONFIG['dataset']
    validate_dataset_paths(dataset_name)
    dataset_dirs = resolve_dataset_dirs(dataset_name, logger)

    train_loader, val_loader = create_dataloaders(
        train_clean_dir=dataset_dirs['train_clean_dir'],
        train_noisy_dir=dataset_dirs['train_noisy_dir'],
        val_clean_dir=dataset_dirs['val_clean_dir'],
        val_noisy_dir=dataset_dirs['val_noisy_dir'],
        batch_size=config.TRAINING_CONFIG['batch_size'],
        num_workers=config.DATASET_CONFIG['num_workers'],
        use_augmentation=config.TRAINING_CONFIG.get('use_augmentation', True),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        train_vad_dir=dataset_dirs['train_vad_dir'],
        val_vad_dir=dataset_dirs['val_vad_dir'],
        use_vad_labels=config.DATASET_CONFIG.get('use_vad_labels', False),
        vad_soft_mask_floor=config.DATASET_CONFIG.get('vad_soft_mask_floor', 0.0),
        complex_mask_clip=config.MODEL_CONFIG.get('complex_mask_clip', 5.0),
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    logger.info("\nInitializing learning rate scheduler...")
    scheduler_type, scheduler, scheduler_description = build_scheduler(optimizer, len(train_loader))
    if resume_scheduler_state is not None and scheduler is not None:
        scheduler.load_state_dict(resume_scheduler_state)
        logger.info("Loaded scheduler state")
    logger.info(f"Scheduler: {scheduler_description}")

    use_amp = config.TRAINING_CONFIG.get('use_mixed_precision', False) and torch.cuda.is_available()
    scaler = GradScaler('cuda') if use_amp else None

    training_state = run_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_type=scheduler_type,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch,
        logger=logger,
        writer=writer,
        scaler=scaler,
        args=args,
        exp_dir=exp_dir,
    )

    history_path, summary_path = finalize_experiment(
        exp_dir,
        training_state['history'],
        best_epoch=training_state['best_epoch'],
        best_val_loss=training_state['best_val_loss'],
        use_amp=use_amp,
    )

    logger.info(f"Training history saved to {history_path}")
    logger.info(f"Experiment summary saved to {summary_path}")
    logger.info(f"Experiment index updated: {config.RESULTS_DIR / 'experiment_index.csv'}")
    logger.info("Training completed")

    if writer:
        writer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Checking for saved checkpoints...")

        try:
            exp_dirs = sorted(config.CHECKPOINTS_DIR.glob('*'), key=lambda x: x.stat().st_mtime, reverse=True)
            if exp_dirs:
                latest_exp = exp_dirs[0]
                print(f"Latest experiment: {latest_exp.name}")

                checkpoints = list(latest_exp.glob('checkpoint_*.pth'))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    print(f" Latest periodic checkpoint: {latest_checkpoint.name}")
                    print(f"\nTo resume training, run:")
                    print(f"  python train_mask_model.py --resume {latest_checkpoint}")
                else:
                    print("No periodic checkpoints found (training may not have reached 500 batches)")
                    best_checkpoint = latest_exp / 'masknet_best.pth'
                    if best_checkpoint.exists():
                        print(f" Best checkpoint available: {best_checkpoint}")
            else:
                print("No experiment directory found")
        except Exception as e:
            print(f"Could not locate checkpoint: {e}")

        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
