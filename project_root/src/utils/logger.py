"""
Structured logging system for speech enhancement project.
Provides consistent logging across all modules with file and console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorama
from colorama import Fore, Style


colorama.init(autoreset=True)


def build_log_file_name(name: str) -> str:
    """Create a timestamped log filename for a script logger."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{name}_{timestamp}.log"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):

        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        

        result = super().format(record)
        

        record.levelname = levelname
        
        return result


def setup_logger(
    name: str = "speech_enhancement",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    level: str = "INFO",
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
    format_string: Optional[str] = None,
    use_color: bool = True,
    console_only: bool = False
) -> logging.Logger:
    """
    Setup a structured logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (default: logs/)
        log_file: Log file name (default: auto-generated with timestamp)
        level: Default logging level
        console_level: Console-specific level (overrides level)
        file_level: File-specific level (overrides level)
        format_string: Custom format string
        use_color: Use colored output for console
        console_only: If True, only console handler is used (no file logging)
        
    Returns:
        Configured logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    

    logger.handlers.clear()
    

    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    

    file_format = '%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s'
    

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level or level))
    
    if use_color:
        console_formatter = ColoredFormatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    else:
        console_formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    

    if not console_only and (log_dir is not None or log_file is not None):

        if log_dir is None:
            from src import config
            log_dir = config.LOGS_DIR
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        

        if log_file is None:
            log_file = build_log_file_name(name)
        
        log_path = log_dir / log_file
        

        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level or level))
        
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")
    
    return logger


def setup_script_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create a standard logger for CLI, demo, and evaluation scripts."""
    try:
        return setup_logger(
            name=name,
            log_dir=log_dir,
            log_file=log_file or build_log_file_name(name),
            level=level,
        )
    except PermissionError:
        return setup_logger(
            name=name,
            level=level,
            console_only=True,
        )


def get_logger(name: str = "speech_enhancement") -> logging.Logger:
    """
    Get an existing logger or create a basic one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


class TrainingLogger:
    """Specialized logger for training progress."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.epoch_start_time = None
        self.batch_start_time = None
    
    def log_training_start(self, config: dict):
        """Log training configuration and start."""
        self.logger.info("=" * 70)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 70)
        self.logger.info(f"Model variant: {config.get('model_variant', 'N/A')}")
        self.logger.info(f"Batch size: {config.get('batch_size', 'N/A')}")
        self.logger.info(f"Epochs: {config.get('num_epochs', 'N/A')}")
        self.logger.info(f"Learning rate: {config.get('learning_rate', 'N/A')}")
        self.logger.info(f"Device: {config.get('device', 'N/A')}")
        self.logger.info("=" * 70)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch_start_time = datetime.now()
        self.logger.info(f"\nEpoch {epoch}/{total_epochs} started")
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """Log epoch end with metrics."""
        if self.epoch_start_time:
            duration = (datetime.now() - self.epoch_start_time).total_seconds()
            self.logger.info(f"Epoch {epoch} completed in {duration:.2f}s")
        
        self.logger.info(f"Epoch {epoch} metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_batch(self, epoch: int, batch: int, total_batches: int, loss: float):
        """Log batch progress."""
        if batch % 10 == 0:
            self.logger.debug(
                f"Epoch {epoch} - Batch {batch}/{total_batches} - Loss: {loss:.4f}"
            )
    
    def log_checkpoint_saved(self, path: str, is_best: bool = False):
        """Log checkpoint saving."""
        if is_best:
            self.logger.info(f"Best model saved: {path}")
        else:
            self.logger.info(f"Checkpoint saved: {path}")
    
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping."""
        self.logger.warning(
            f"Early stopping triggered at epoch {epoch} "
            f"(patience: {patience})"
        )
    
    def log_training_end(self, total_time: float):
        """Log training completion."""
        self.logger.info("=" * 70)
        self.logger.info(f"TRAINING COMPLETED in {total_time:.2f}s ({total_time/60:.2f} min)")
        self.logger.info("=" * 70)


class EvaluationLogger:
    """Specialized logger for evaluation."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_evaluation_start(self, config: dict):
        """Log evaluation start."""
        self.logger.info("=" * 70)
        self.logger.info("EVALUATION STARTED")
        self.logger.info("=" * 70)
        self.logger.info(f"Model: {config.get('checkpoint_path', 'N/A')}")
        self.logger.info(f"Dataset: {config.get('dataset', 'N/A')}")
        self.logger.info("=" * 70)
    
    def log_metrics(self, metrics: dict, prefix: str = ""):
        """Log evaluation metrics."""
        self.logger.info(f"\n{prefix}Evaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        self.logger.info(f"    {sub_key}: {sub_value:.4f}")
                    else:
                        self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_sample_processed(self, sample_name: str, metrics: dict):
        """Log individual sample processing."""
        self.logger.debug(f"Processed: {sample_name}")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.debug(f"  {key}: {value:.4f}")
    
    def log_evaluation_end(self, total_samples: int, total_time: float):
        """Log evaluation completion."""
        self.logger.info("=" * 70)
        self.logger.info(
            f"EVALUATION COMPLETED: {total_samples} samples in {total_time:.2f}s "
            f"({total_samples/total_time:.2f} samples/s)"
        )
        self.logger.info("=" * 70)



if __name__ == "__main__":

    logger = setup_logger("test", log_file="test.log")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    

    train_logger = TrainingLogger(logger)
    train_logger.log_training_start({
        'model_variant': 'balanced',
        'batch_size': 4,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'device': 'cuda'
    })
    

    eval_logger = EvaluationLogger(logger)
    eval_logger.log_metrics({
        'PESQ': 2.85,
        'STOI': 0.92,
        'SNR': 15.3
    })
