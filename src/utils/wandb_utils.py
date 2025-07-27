"""
Weights & Biases utilities for experiment tracking.
Enhanced integration with PyTorch Lightning and Hydra configuration.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class WandBManager:
    """Enhanced WandB manager for experiment tracking."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize WandB manager with configuration.
        
        Args:
            cfg: Hydra configuration containing wandb settings
        """
        self.cfg = cfg
        self.wandb_cfg = cfg.get("wandb", {})
        self.run = None
        self.logger = None
    
    def setup_wandb(
        self, 
        job_type: str = "training",
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> WandbLogger:
        """
        Setup WandB logging for PyTorch Lightning.
        
        Args:
            job_type: Type of job (training, evaluation, inference)
            tags: Additional tags for the run
            notes: Notes for the run
            
        Returns:
            Configured WandbLogger instance
        """
        if not self._validate_wandb_config():
            logger.warning("WandB configuration invalid, using offline mode")
            os.environ["WANDB_MODE"] = "offline"
        
        # Prepare run configuration
        run_config = self._prepare_run_config()
        
        # Setup tags
        run_tags = [job_type]
        if tags:
            run_tags.extend(tags)
        if "tags" in self.wandb_cfg:
            run_tags.extend(self.wandb_cfg.tags)
        
        # Initialize WandB logger
        self.logger = WandbLogger(
            project=self.wandb_cfg.get("project", "dialogue-summarization"),
            entity=self.wandb_cfg.get("entity"),
            name=self._generate_run_name(),
            job_type=job_type,
            tags=run_tags,
            notes=notes or self.wandb_cfg.get("notes"),
            config=run_config,
            save_dir=self.cfg.get("output_dir", "outputs"),
            offline=self.wandb_cfg.get("offline", False),
            log_model=self.wandb_cfg.get("log_model", True),
        )
        
        # Set additional environment variables
        if self.wandb_cfg.get("log_model", True):
            os.environ["WANDB_LOG_MODEL"] = "true"
        
        if self.wandb_cfg.get("watch", False):
            os.environ["WANDB_WATCH"] = "all"
        else:
            os.environ["WANDB_WATCH"] = "false"
        
        logger.info(f"WandB logger initialized: {self.logger.experiment.name}")
        return self.logger
    
    def _validate_wandb_config(self) -> bool:
        """Validate WandB configuration."""
        required_fields = ["project"]
        
        for field in required_fields:
            if field not in self.wandb_cfg:
                logger.error(f"Missing required WandB config field: {field}")
                return False
        
        # Check API key if not in offline mode
        if not self.wandb_cfg.get("offline", False):
            api_key = os.getenv("WANDB_API_KEY") or self.wandb_cfg.get("api_key")
            if not api_key:
                logger.warning("WandB API key not found, will use offline mode")
                return False
        
        return True
    
    def _prepare_run_config(self) -> Dict[str, Any]:
        """Prepare configuration for WandB run."""
        # Convert Hydra config to plain dict for WandB
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        
        # Add system information
        config_dict["system"] = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "working_directory": str(Path.cwd()),
        }
        
        # Add model summary information
        if "model" in config_dict:
            config_dict["model_summary"] = {
                "architecture": config_dict["model"].get("name", "unknown"),
                "parameters": config_dict["model"].get("parameters", {}),
            }
        
        return config_dict
    
    def _generate_run_name(self) -> str:
        """Generate a descriptive run name similar to peer's format."""
        if "name" in self.wandb_cfg and self.wandb_cfg.name:
            return self.wandb_cfg.name
        
        # Get configuration values
        model_name = self.cfg.get("model", {}).get("name", "kobart")
        
        # Get batch size from training config (not dataset config)
        training_cfg = self.cfg.get("training", {})
        batch_size = training_cfg.get("batch_size", 32)
        
        # Get training parameters
        learning_rate = training_cfg.get("optimizer", {}).get("lr", "unknown")
        
        # Get wandb entity (shortened for run name)
        wandb_entity = self.wandb_cfg.get("entity", "user")
        # Use a shorter version of the entity for the run name
        if wandb_entity == "boot_camp_13_2nd_group_2nd":
            wandb_username = "wb2x"  # Use your shorter identifier
        else:
            wandb_username = wandb_entity.split("_")[0] if "_" in wandb_entity else wandb_entity
        
        # Get augmentation or preprocessing strategy
        preprocessing_cfg = self.cfg.get("preprocessing", {})
        augmentation_strategy = preprocessing_cfg.get("strategy", "std")  # shortened to "std"
        
        # Get additional training info
        early_stopping = training_cfg.get("early_stopping", {})
        early_stopping_patience = early_stopping.get("patience", "") if early_stopping else ""
        early_stopping_str = f"es{early_stopping_patience}" if early_stopping_patience else ""
        
        # Format run name: username_(submission)_kobart-std-b32-lr001-es20_Rsubmission
        # Clean model name for consistency
        clean_model_name = model_name.replace("_", "-")
        
        # Generate run name components
        components = [clean_model_name, augmentation_strategy, f"b{batch_size}"]
        
        # Add learning rate if available
        if learning_rate != "unknown":
            # Convert scientific notation to simple format: 1.0e-5 -> 1e5
            if isinstance(learning_rate, (int, float)):
                if learning_rate >= 0.001:
                    lr_str = f"{learning_rate:.3f}".replace(".", "")[:4]  # e.g., 0.001 -> 001
                else:
                    # For very small learning rates like 1e-5
                    lr_str = f"{learning_rate:.0e}".replace("e-0", "e").replace("e-", "e")  # 1e-5 -> 1e5
            else:
                lr_str = str(learning_rate).replace(".", "")[:4]
            components.append(f"lr{lr_str}")
        
        # Add early stopping info
        if early_stopping_str:
            components.append(early_stopping_str)
        
        # Combine components with hyphens
        model_details = "-".join(components)
        
        # Final format: username_(submission)_model-details_Rsubmission
        run_name = f"{wandb_username}_(submission)_{model_details}_Rsubmission"
        
        return run_name
    
    def update_run_name_with_submission_score(self, rouge_score: float) -> None:
        """
        Update the run name with actual submission score.
        This should only be called manually when you get submission results.
        
        Args:
            rouge_score: The ROUGE score from submission results
        """
        if not self.logger or not self.logger.experiment:
            logger.warning("WandB not initialized, cannot update run name")
            return
        
        current_name = self.logger.experiment.name
        
        # Replace the _Rsubmission suffix with actual score
        if current_name.endswith("_Rsubmission"):
            # Format score to 4 decimal places
            formatted_score = f"{rouge_score:.4f}"
            
            # Replace _Rsubmission with _R{actual_score}
            new_name = current_name.replace("_Rsubmission", f"_R{formatted_score}")
            
            # Update the run name
            try:
                self.logger.experiment.name = new_name
                logger.info(f"Updated run name with submission score: {current_name} -> {new_name}")
            except Exception as e:
                logger.warning(f"Failed to update run name: {e}")
        else:
            logger.warning("Run name does not end with '_Rsubmission', cannot update")
    
    def update_run_name_with_metrics(self, metrics: Dict[str, float]) -> None:
        """
        DEPRECATED: This method is no longer used for automatic updates.
        Use update_run_name_with_submission_score() manually when you get submission results.
        
        Args:
            metrics: Dictionary containing performance metrics (e.g., rouge scores)
        """
        logger.info("Automatic run name updates are disabled. Use update_run_name_with_submission_score() for manual updates.")
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, update_name: bool = False) -> None:
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
            update_name: Whether to update run name with best metrics
        """
        if self.logger and self.logger.experiment:
            if step is not None:
                self.logger.log_metrics(metrics, step=step)
            else:
                self.logger.log_metrics(metrics)
            
            # Update run name with metrics if requested
            if update_name:
                self.update_run_name_with_metrics(metrics)
    
    def log_artifacts(
        self, 
        file_paths: Union[str, Path, List[Union[str, Path]]], 
        artifact_type: str = "model",
        artifact_name: Optional[str] = None
    ) -> None:
        """
        Log artifacts (models, configs, etc.) to WandB.
        
        Args:
            file_paths: Path(s) to files to log
            artifact_type: Type of artifact (model, dataset, config, etc.)
            artifact_name: Name for the artifact (auto-generated if None)
        """
        if not self.logger or not self.logger.experiment:
            logger.warning("WandB not initialized, cannot log artifacts")
            return
        
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        
        # Generate artifact name if not provided
        if artifact_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifact_name = f"{artifact_type}_{timestamp}"
        
        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=f"{artifact_type.title()} artifact from dialogue summarization"
        )
        
        # Add files to artifact
        for file_path in file_paths:
            file_path = Path(file_path)
            if file_path.exists():
                artifact.add_file(str(file_path))
                logger.info(f"Added file to artifact: {file_path}")
            else:
                logger.warning(f"File not found, skipping: {file_path}")
        
        # Log artifact
        self.logger.experiment.log_artifact(artifact)
        logger.info(f"Artifact logged: {artifact_name}")
    
    def log_model_checkpoint(
        self, 
        checkpoint_path: Union[str, Path],
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log model checkpoint as WandB artifact.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_name: Name for the model artifact
            metadata: Additional metadata for the model
        """
        checkpoint_path = Path(checkpoint_path)
        
        if model_name is None:
            model_name = f"model_{checkpoint_path.stem}"
        
        # Create model artifact
        model_artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description="Dialogue summarization model checkpoint",
            metadata=metadata or {}
        )
        
        model_artifact.add_file(str(checkpoint_path))
        
        if self.logger and self.logger.experiment:
            self.logger.experiment.log_artifact(model_artifact)
            logger.info(f"Model checkpoint logged: {model_name}")
    
    def log_predictions(
        self, 
        predictions_file: Union[str, Path],
        ground_truth_file: Optional[Union[str, Path]] = None,
        dataset_split: str = "test"
    ) -> None:
        """
        Log prediction results to WandB.
        
        Args:
            predictions_file: Path to predictions CSV file
            ground_truth_file: Path to ground truth file (optional)
            dataset_split: Dataset split name (train, dev, test)
        """
        import pandas as pd
        
        predictions_df = pd.read_csv(predictions_file)
        
        # Create predictions table
        predictions_table = wandb.Table(dataframe=predictions_df)
        
        if self.logger and self.logger.experiment:
            self.logger.experiment.log({
                f"{dataset_split}_predictions": predictions_table
            })
        
        # Log as artifact
        self.log_artifacts(
            predictions_file,
            artifact_type="predictions", 
            artifact_name=f"{dataset_split}_predictions"
        )
        
        logger.info(f"Predictions logged for {dataset_split} split")
    
    def finish(self) -> None:
        """Finish WandB run."""
        if self.logger and self.logger.experiment:
            self.logger.experiment.finish()
            logger.info("WandB run finished")


class WandBMetricsCallback(Callback):
    """PyTorch Lightning callback for logging custom metrics to WandB."""
    
    def __init__(self, wandb_manager: WandBManager):
        """
        Initialize callback with WandB manager.
        
        Args:
            wandb_manager: WandB manager instance
        """
        super().__init__()
        self.wandb_manager = wandb_manager
    
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Log validation metrics at epoch end."""
        if hasattr(pl_module, "validation_metrics"):
            metrics = pl_module.validation_metrics
            self.wandb_manager.log_metrics(
                metrics, 
                step=trainer.current_epoch,
                update_name=False  # Never auto-update run name
            )
    
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Log training metrics at epoch end.""" 
        if hasattr(pl_module, "training_metrics"):
            metrics = pl_module.training_metrics
            self.wandb_manager.log_metrics(
                metrics,
                step=trainer.current_epoch
            )


def setup_wandb_environment(cfg: DictConfig) -> None:
    """
    Setup WandB environment variables from configuration.
    
    Args:
        cfg: Configuration containing WandB settings
    """
    wandb_cfg = cfg.get("wandb", {})
    
    # Set API key if provided in config
    if "api_key" in wandb_cfg:
        os.environ["WANDB_API_KEY"] = wandb_cfg.api_key
    
    # Set mode (online/offline)
    if wandb_cfg.get("offline", False):
        os.environ["WANDB_MODE"] = "offline"
    
    # Set additional environment variables
    env_vars = {
        "WANDB_PROJECT": wandb_cfg.get("project"),
        "WANDB_ENTITY": wandb_cfg.get("entity"),
        "WANDB_JOB_TYPE": wandb_cfg.get("job_type", "training"),
    }
    
    for var, value in env_vars.items():
        if value:
            os.environ[var] = str(value)
    
    logger.info("WandB environment configured")


def get_wandb_run_url() -> Optional[str]:
    """
    Get the URL of the current WandB run.
    
    Returns:
        URL string if run is active, None otherwise
    """
    if wandb.run:
        return wandb.run.get_url()
    return None


def download_wandb_artifact(
    artifact_path: str,
    download_dir: Union[str, Path],
    project: Optional[str] = None,
    entity: Optional[str] = None
) -> Path:
    """
    Download a WandB artifact.
    
    Args:
        artifact_path: Path to artifact (format: "project/artifact:version")
        download_dir: Directory to download to
        project: WandB project name (if not in artifact_path)
        entity: WandB entity name (if not in artifact_path)
        
    Returns:
        Path to downloaded artifact directory
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize WandB if not already done
    if not wandb.run:
        wandb.init(project=project, entity=entity, job_type="download")
    
    # Download artifact
    artifact = wandb.use_artifact(artifact_path)
    artifact_dir = artifact.download(root=str(download_dir))
    
    logger.info(f"Artifact downloaded to: {artifact_dir}")
    return Path(artifact_dir)