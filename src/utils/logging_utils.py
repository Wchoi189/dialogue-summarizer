"""
Logging utilities with enhanced Korean text support and structured logging.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from icecream import ic
from omegaconf import DictConfig
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table


def setup_logging(
    cfg: Optional[DictConfig] = None,
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    use_rich: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging with Korean text support.
    
    Args:
        cfg: Configuration containing logging settings
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        use_rich: Whether to use Rich formatting
        
    Returns:
        Configured logger
    """
    # Extract logging config if provided
    if cfg and "logging" in cfg:
        log_cfg = cfg.logging
        log_level = log_cfg.get("level", log_level)
        log_file = log_cfg.get("file", log_file)
        use_rich = log_cfg.get("use_rich", use_rich)
    
    # Clear existing handlers
    logging.getLogger().handlers = []
    
    # Setup formatters
    if use_rich:
        # Rich handler for console output
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True
        )
        console_handler.setLevel(getattr(logging, log_level.upper()))
    else:
        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        ic(f"File logging enabled: {log_file}")
    
    # Configure icecream
    ic.configureOutput(prefix="🧊 ", includeContext=True)
    
    ic(f"Logging configured: level={log_level}, rich={use_rich}")
    return root_logger


def log_config_summary(cfg: DictConfig, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a summary of the configuration.
    
    Args:
        cfg: Configuration to summarize
        logger: Logger instance (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    console = Console()
    
    # Create configuration summary table
    table = Table(title="Configuration Summary")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="magenta")
    table.add_column("Value", style="green")
    
    def add_config_rows(config_dict: Any, section: str = ""):
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if isinstance(value, dict) or isinstance(value, list):
                    add_config_rows(value, f"{section}.{key}" if section else key)
                else:
                    table.add_row(section, key, str(value))
        elif isinstance(config_dict, list):
            for idx, item in enumerate(config_dict):
                add_config_rows(item, f"{section}[{idx}]")
        else:
            table.add_row(section, "", str(config_dict))
    
    # Convert config to dict and add rows
    from omegaconf import OmegaConf
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    add_config_rows(config_dict)
    
    console.print(table)
    logger.info("Configuration summary displayed")


def log_training_progress(
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log training progress with structured format.
    
    Args:
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss (optional)
        metrics: Additional metrics dictionary
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Basic progress message
    msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}"
    
    if val_loss is not None:
        msg += f" | Val Loss: {val_loss:.4f}"
    
    if metrics:
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        msg += f" | {metric_str}"
    
    logger.info(msg)
    ic(f"Training progress: {msg}")


def log_model_info(
    model: Any,
    total_params: Optional[int] = None,
    trainable_params: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log model information and parameters.
    
    Args:
        model: Model instance
        total_params: Total parameter count
        trainable_params: Trainable parameter count
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    model_name = model.__class__.__name__
    logger.info(f"Model: {model_name}")
    
    if total_params is not None:
        logger.info(f"Total parameters: {total_params:,}")
        ic(f"Total parameters: {total_params:,}")
    
    if trainable_params is not None:
        logger.info(f"Trainable parameters: {trainable_params:,}")
        ic(f"Trainable parameters: {trainable_params:,}")
        
        if total_params is not None:
            trainable_ratio = trainable_params / total_params * 100
            logger.info(f"Trainable ratio: {trainable_ratio:.2f}%")


def log_data_info(
    train_size: int,
    val_size: int,
    test_size: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log dataset size information.
    
    Args:
        train_size: Training set size
        val_size: Validation set size
        test_size: Test set size (optional)
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Dataset sizes - Train: {train_size:,}, Val: {val_size:,}")
    ic(f"Dataset sizes - Train: {train_size:,}, Val: {val_size:,}")
    
    if test_size is not None:
        logger.info(f"Test size: {test_size:,}")
        ic(f"Test size: {test_size:,}")


def log_evaluation_results(
    results: Dict[str, float],
    dataset_name: str = "validation",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log evaluation results in a structured format.
    
    Args:
        results: Dictionary of metric names and values
        dataset_name: Name of the dataset being evaluated
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"=== {dataset_name.title()} Results ===")
    
    for metric_name, value in results.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    ic(f"{dataset_name} results: {results}")


class ExperimentLogger:
    """Enhanced logger for experiment tracking."""
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path],
        cfg: Optional[DictConfig] = None
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory for log outputs
            cfg: Configuration for logging settings
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger with file output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logging(
            cfg=cfg,
            log_file=log_file
        )
        
        self.start_time = datetime.now()
        self.logger.info(f"Experiment started: {experiment_name}")
        ic(f"Experiment logger initialized: {experiment_name}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.logger.info("=== Hyperparameters ===")
        for param, value in hyperparams.items():
            self.logger.info(f"{param}: {value}")
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log epoch results."""
        log_training_progress(
            epoch=epoch,
            train_loss=train_metrics.get("loss", 0.0),
            val_loss=val_metrics.get("loss") if val_metrics else None,
            metrics={**train_metrics, **(val_metrics or {})},
            logger=self.logger
        )
    
    def log_final_results(self, results: Dict[str, float]) -> None:
        """Log final experiment results."""
        duration = datetime.now() - self.start_time
        
        self.logger.info("=== Final Results ===")
        log_evaluation_results(results, "final", self.logger)
        self.logger.info(f"Experiment duration: {duration}")
        ic(f"Experiment completed in: {duration}")
    
    # def get_config_summary(self, cfg: DictConfig) -> Dict[str, Any]:
    #     """Get a summary of key configuration parameters."""
    #     summary = {
    #         "model_name": cfg.get("model", {}).get("name", "unknown"),
    #         "dataset_path": cfg.get("dataset", {}).get("data_path", "unknown"),
            
    #         # ✅ CHANGE THIS LINE
    #         "batch_size": cfg.get("dataset", {}).get("batch_size", "unknown"),
            
    #         "learning_rate": cfg.get("training", {}).get("optimizer", {}).get("lr", "unknown"),
    #         "max_epochs": cfg.get("training", {}).get("max_epochs", "unknown"),
    #     }
    #     return summary
    
    def get_config_summary(self, cfg: DictConfig) -> Dict[str, Any]:
        """Get a summary of key configuration parameters."""
        summary = {
            "model_name": cfg.get("model", {}).get("name", "unknown"),
            "dataset_path": cfg.get("dataset", {}).get("data_path", "unknown"),
            
            # ✅ FIX: Update the path to correctly find batch_size
            "batch_size": cfg.get("dataset", {}).get("batch_size", "unknown"),
            
            "learning_rate": cfg.get("training", {}).get("optimizer", {}).get("lr", "unknown"),
            "max_epochs": cfg.get("training", {}).get("max_epochs", "unknown"),
        }
        return summary    