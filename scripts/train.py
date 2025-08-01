#!/usr/bin/env python3
"""
Training script for dialogue summarization using Click CLI.
Enhanced with comprehensive logging and experiment tracking.
"""
# FILE: scripts/train.py

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Third-Party Imports
import click
import pytorch_lightning as pl
import torch
from icecream import ic
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger

# --- Configuration (Should be inside a main function) ---
# Suppress informational messages from the transformers library
from transformers import logging
logging.set_verbosity_error()

# --- Local Application Imports ---
# Add project source to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.datamodule import DialogueDataModule
from models.kobart_model import KoBARTSummarizationModel
from utils.config_utils import ConfigManager
from utils.logging_utils import ExperimentLogger, setup_logging
from utils.wandb_utils import WandBManager, WandBMetricsCallback

def setup_pytorch_optimizations(cfg):
    """
    Setup PyTorch optimizations from dedicated pytorch config.
    
    Args:
        cfg: Complete configuration including pytorch section
    """
    if "pytorch" not in cfg:
        ic("No pytorch config found, using defaults")
        return
    
    pytorch_cfg = cfg.pytorch
    ic(f"Applying PyTorch optimizations: {pytorch_cfg}")
    
    # 1. Setup Dynamo settings
    if "dynamo" in pytorch_cfg:
        dynamo_cfg = pytorch_cfg.dynamo
        
        # Cache size limit
        if "cache_size_limit" in dynamo_cfg:
            cache_limit = dynamo_cfg.cache_size_limit
            torch._dynamo.config.cache_size_limit = cache_limit
            ic(f"✓ Set dynamo cache_size_limit: {cache_limit}")
        
        # Error suppression
        if dynamo_cfg.get("suppress_errors", False):
            torch._dynamo.config.suppress_errors = True
            ic("✓ Enabled dynamo error suppression")
        
        # Verbose mode
        if dynamo_cfg.get("verbose", False):
            torch._dynamo.config.verbose = True
            ic("✓ Enabled dynamo verbose mode")
    
    # 2. Setup compilation settings (stored for later use in model)
    if "compile" in pytorch_cfg:
        compile_cfg = pytorch_cfg.compile
        ic(f"✓ Model compilation config loaded: enabled={compile_cfg.get('enabled', False)}")
    
    # 3. Setup performance settings
    if "float32_matmul_precision" in pytorch_cfg:
        precision = pytorch_cfg.float32_matmul_precision
        torch.set_float32_matmul_precision(precision)
        ic(f"✓ Set float32_matmul_precision: {precision}")
    
    # 4. Setup CUDNN settings
    if pytorch_cfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True
        ic("✓ Enabled CUDNN benchmark")
    
    if pytorch_cfg.get("cudnn_deterministic", False):
        torch.backends.cudnn.deterministic = True
        ic("✓ Enabled CUDNN deterministic mode")
    
    # 5. Memory management settings (stored for later use)
    if "empty_cache_steps" in pytorch_cfg:
        ic(f"✓ GPU cache clearing every {pytorch_cfg.empty_cache_steps} steps")

class DialogueTrainer:
    """Main trainer class for dialogue summarization."""
    
    def __init__(self):
        """Initialize trainer."""
        self.config_manager = ConfigManager()
        self.cfg = None
        self.wandb_manager = None
        self.experiment_logger = None
    
    def train(
        self,
        config_name: str = "config",
        config_path: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        resume_from: Optional[str] = None,
        fast_dev_run: bool = False,
        max_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> str:
        """
        Train dialogue summarization model.
        
        Args:
            config_name: Name of config file (without .yaml)
            config_path: Custom path to config directory
            overrides: List of config overrides
            resume_from: Path to checkpoint to resume from
            fast_dev_run: Run one batch for debugging
            max_epochs: Override max epochs
            batch_size: Override batch size
            learning_rate: Override learning rate
            
        Returns:
            Path to best model checkpoint
        """
        ic(f"Starting training with config: {config_name}")
        
        # Set config directory if provided
        if config_path:
            self.config_manager = ConfigManager(config_path)
        
        # Build overrides from arguments
        overrides = overrides or []
        if fast_dev_run:
            overrides.append("training.fast_dev_run=true")
        if max_epochs is not None:
            overrides.append(f"training.max_epochs={max_epochs}")
        if batch_size is not None:
            overrides.append(f"dataset.batch_size={batch_size}")
        if learning_rate is not None:
            overrides.append(f"training.optimizer.lr={learning_rate}")
        
        # Load configuration
        self.cfg = self.config_manager.load_config(
            config_name=config_name,
            overrides=overrides
        )

        # ✅ CRITICAL: Setup PyTorch optimizations BEFORE any model/training code
        setup_pytorch_optimizations(self.cfg)    

        # Validate configuration
        self.config_manager.validate_config(self.cfg)
        
        # Setup logging
        setup_logging(self.cfg)
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
        # Create output directories
        output_dir = self._create_output_directories()
        
        # Setup data
        ic("Setting up data module...")
        datamodule = DialogueDataModule(self.cfg)
        
        # Setup model
        ic("Setting up model...")
        model = KoBARTSummarizationModel(self.cfg)

        # Apply memory management if configured
        if hasattr(self.cfg, 'pytorch') and 'empty_cache_steps' in self.cfg.pytorch:
            self.empty_cache_steps = self.cfg.pytorch.empty_cache_steps
        else:
            self.empty_cache_steps = 10  # default

        # Ensure model is in training mode
        model.train()
        
        # Log model info
        model_info = model.get_model_summary()
        ic(f"Model info: {model_info}")
        
        # Setup trainer
        trainer = self._setup_trainer(output_dir, resume_from, fast_dev_run)
        assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
        
        # Train model
        ic("Starting training...")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=resume_from
        )
        # Log final evaluation
        ic("Starting final evaluation on the best model...")
        
        # Capture the results from the test run
        test_results = trainer.test(
            model=model,
            dataloaders=datamodule.val_dataloader(), # Use the validation dataloader
            ckpt_path="best"
        )

        # DYNAMICALLY UPDATE THE RUN NAME
        if self.wandb_manager and test_results:
            # The result is a list of dictionaries, so we take the first one
            final_metrics = test_results[0]
            # Extract the final overall ROUGE F1-score
            rouge_f_score = final_metrics.get("eval/rouge_f")
            if rouge_f_score is not None:
                ic(f"Updating WandB run name with final ROUGE score: {rouge_f_score:.4f}")
                self.wandb_manager.update_run_name_with_submission_score(rouge_f_score)

        # Get best model path
        best_model_path = trainer.checkpoint_callback.best_model_path
        ic(f"Training completed. Best model: {best_model_path}")
        
        # Log final results
        assert self.experiment_logger is not None
        if hasattr(model, "validation_metrics"):
            self.experiment_logger.log_final_results(model.validation_metrics)
        
        # Cleanup
        if self.wandb_manager:
            self.wandb_manager.finish()
        
        return best_model_path
    
    def validate(
        self,
        config_name: str = "config",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        overrides: Optional[List[str]] = None
    ) -> dict:
        """
        Validate model on validation set.
        
        Args:
            config_name: Name of config file
            checkpoint_path: Path to model checkpoint
            config_path: Custom config directory path
            overrides: Config overrides
            
        Returns:
            Validation metrics
        """
        ic(f"Starting validation with checkpoint: {checkpoint_path}")
        
        # Setup config
        if config_path:
            self.config_manager = ConfigManager(config_path)
        
        overrides = overrides or []
        
        self.cfg = self.config_manager.load_config(
            config_name=config_name,
            overrides=overrides
        )
        
        # Setup logging
        setup_logging(self.cfg)
        
        # Setup data
        datamodule = DialogueDataModule(self.cfg)
        
        # Load model from checkpoint
        if checkpoint_path:
            model = KoBARTSummarizationModel.load_from_checkpoint(
                checkpoint_path,
                cfg=self.cfg
            )
        else:
            model = KoBARTSummarizationModel(self.cfg)
        
        # Setup trainer for validation
        trainer = pl.Trainer(
            accelerator=self.cfg.training.accelerator,
            devices=self.cfg.training.devices,
            precision=self.cfg.training.precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        
        # Run validation
        results = trainer.validate(model, datamodule=datamodule)
        
        ic(f"Validation results: {results}")
        return dict(results[0]) if results else {}
    
    def _setup_experiment_tracking(self) -> None:
        """Setup experiment tracking with WandB."""
        assert self.cfg is not None
        # Setup WandB if configured
        if "wandb" in self.cfg and not self.cfg.wandb.get("offline", False):
            self.wandb_manager = WandBManager(self.cfg)
        
        # Setup experiment logger
        output_dir = Path(self.cfg.output_dir)
        self.experiment_logger = ExperimentLogger(
            experiment_name=self.cfg.experiment_name,
            output_dir=output_dir,
            cfg=self.cfg
        )
        
        # Log configuration
        config_summary = self.config_manager.get_config_summary(self.cfg)
        self.experiment_logger.log_hyperparameters(config_summary)
        
        ic("Experiment tracking setup complete")
    
    def _create_output_directories(self) -> Path:
        """Create output directories."""
        assert self.cfg is not None, "Configuration must be loaded before creating output directories"
        output_dir = Path(self.cfg.output_dir)
        
        # Create subdirectories
        directories = [
            output_dir / "models",
            output_dir / "logs", 
            output_dir / "predictions",
            output_dir / "configs"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = output_dir / "configs" / "config.yaml"
        if self.cfg is not None:
            self.config_manager.save_config(self.cfg, config_path)
        
        ic(f"Output directories created: {output_dir}")
        return output_dir
    
    def _setup_trainer(
        self,
        output_dir: Path,
        resume_from: Optional[str] = None,
        fast_dev_run: bool = False
    ) -> pl.Trainer:
        """Setup PyTorch Lightning trainer."""
        assert self.cfg is not None, "Configuration must be loaded before setting up trainer"
        training_cfg = self.cfg.training
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir / "models",
            filename="best-{epoch:02d}-{val/rouge_f:.4f}", 
            monitor="val/rouge_f",
            mode=training_cfg.mode,
            save_top_k=training_cfg.save_top_k,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if training_cfg.early_stopping.enabled:
            early_stop_callback = EarlyStopping(
                monitor=training_cfg.early_stopping.monitor,
                patience=training_cfg.early_stopping.patience,
                mode=training_cfg.early_stopping.mode,
                min_delta=training_cfg.early_stopping.min_delta,
                verbose=training_cfg.early_stopping.verbose,
            )
            callbacks.append(early_stop_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        
        # Progress bar disabled to avoid hanging issues
        # Uncomment the next two lines if you want to enable progress bar
        # progress_bar = TQDMProgressBar()
        # callbacks.append(progress_bar)
        
        # WandB callback
        if self.wandb_manager:
            wandb_callback = WandBMetricsCallback(self.wandb_manager)
            callbacks.append(wandb_callback)
        
        # Setup loggers
        loggers = []
        
        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=output_dir / "logs",
            name="tensorboard",
            version=None
        )
        loggers.append(tb_logger)
        
        # WandB logger
        if self.wandb_manager:
            wandb_logger = self.wandb_manager.setup_wandb(
                job_type="training",
                tags=["training", "kobart"]
            )
            loggers.append(wandb_logger)
        
        # Create trainer
        trainer = pl.Trainer(
            # Hardware
            accelerator=training_cfg.accelerator,
            devices=training_cfg.devices,
            precision=training_cfg.precision,
            
            # Training
            max_epochs=training_cfg.max_epochs,
            max_steps=training_cfg.max_steps,
            accumulate_grad_batches=training_cfg.accumulate_grad_batches,
            gradient_clip_val=training_cfg.gradient_clip_val,
            gradient_clip_algorithm=training_cfg.gradient_clip_algorithm,
            
            # Validation
            val_check_interval=training_cfg.val_check_interval,
            check_val_every_n_epoch=training_cfg.check_val_every_n_epoch,
            
            # Logging
            log_every_n_steps=training_cfg.get('log_every_n_steps', 50),
            logger=loggers,
            
            # Callbacks
            callbacks=callbacks,
            
            # Progress bar enabled for better feedback
            enable_progress_bar=True,
            
            # Debugging
            fast_dev_run=fast_dev_run or training_cfg.fast_dev_run,
            overfit_batches=training_cfg.overfit_batches,
            limit_train_batches=training_cfg.limit_train_batches,
            limit_val_batches=training_cfg.limit_val_batches,
            
            # Reproducibility - use get() with defaults for missing keys
            deterministic=training_cfg.get("deterministic", False),
            benchmark=training_cfg.get("benchmark", True),
            
            # Profiler - use get() with default for missing key
            profiler=training_cfg.get("profiler", None),
        )
        
        ic("Trainer setup complete")
        return trainer


@click.group()
def cli():
    """Dialogue Summarization Training CLI."""
    pass


@cli.command()
@click.option('--config-name', default='config', help='Configuration name')
@click.option('--config-path', type=click.Path(), help='Custom config directory')
@click.option('--resume-from', type=click.Path(), help='Checkpoint to resume from')
@click.option('--fast-dev-run', is_flag=True, help='Run one batch for debugging')
@click.option('--max-epochs', type=int, help='Override max epochs')
@click.option('--batch-size', type=int, help='Override batch size')
@click.option('--learning-rate', type=float, help='Override learning rate')
@click.option('--override', 'overrides', multiple=True, help='Config overrides (key=value)')
def train(config_name, config_path, resume_from, fast_dev_run, max_epochs, batch_size, learning_rate, overrides):
    """Train dialogue summarization model."""
    trainer = DialogueTrainer()
    
    best_model_path = trainer.train(
        config_name=config_name,
        config_path=config_path,
        overrides=list(overrides),
        resume_from=resume_from,
        fast_dev_run=fast_dev_run,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    click.echo(f"Training complete. Best model: {best_model_path}")


@cli.command()
@click.option('--config-name', default='config', help='Configuration name')
@click.option('--checkpoint-path', type=click.Path(exists=True), help='Model checkpoint')
@click.option('--config-path', type=click.Path(), help='Custom config directory')
@click.option('--override', 'overrides', multiple=True, help='Config overrides')
def validate(config_name, checkpoint_path, config_path, overrides):
    """Validate model on validation set."""
    trainer = DialogueTrainer()
    
    results = trainer.validate(
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        overrides=list(overrides)
    )
    
    click.echo("Validation Results:")
    for key, value in results.items():
        click.echo(f"  {key}: {value}")


def main():
    """Main entry point using Click CLI."""
    # Set environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = "0"

    # ADD THIS LINE to optimize for Tensor Cores
    torch.set_float32_matmul_precision('medium')    

    # Enable faster PyTorch operations
    torch.backends.cudnn.benchmark = True
    
    cli()


if __name__ == "__main__":
    main()