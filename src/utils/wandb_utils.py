# FILE: src/utils/wandb_utils.py
"""
Weights & Biases utilities for experiment tracking.
Enhanced integration with PyTorch Lightning and Hydra configuration.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import wandb
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger(__name__)

class WandBManager:
    """Enhanced WandB manager for experiment tracking."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.wandb_cfg = cfg.get("wandb", {})
        self.logger: Optional[WandbLogger] = None

    def setup_wandb(self) -> WandbLogger:
        """Initializes and returns the WandbLogger for the Trainer."""
        if not self._validate_wandb_config():
            ic("WandB configuration invalid or API key not found. Running in offline mode.")
            os.environ["WANDB_MODE"] = "offline"
            self.wandb_cfg["offline"] = True

        run_config = OmegaConf.to_container(self.cfg, resolve=True)
        run_name = self._generate_run_name()

        self.logger = WandbLogger(
            project=self.wandb_cfg.get("project"),
            entity=self.wandb_cfg.get("entity"),
            name=run_name,
            config=run_config,
            tags=self.wandb_cfg.get("tags"),
            notes=self.wandb_cfg.get("notes"),
            log_model=self.wandb_cfg.get("log_model", False), # Default to False to save space
            offline=self.wandb_cfg.get("offline", False),
            save_dir=self.cfg.get("output_dir", "outputs")
        )
        ic(f"WandB logger initialized for run: {run_name}")
        return self.logger

    def _validate_wandb_config(self) -> bool:
        """Validates that project and API key (for online mode) are present."""
        if "project" not in self.wandb_cfg:
            logger.error("Missing 'project' in wandb config.")
            return False
        if not self.wandb_cfg.get("offline", False):
            return "WANDB_API_KEY" in os.environ
        return True

    def _generate_run_name(self) -> str:
        """Generates a descriptive run name from the configuration."""
        # Allow manual override from the config file
        if self.wandb_cfg.get("name"):
            return self.wandb_cfg.name

        # --- 1. Gather Components from Config ---
        
        # User prefix from the wandb entity
        entity = self.wandb_cfg.get("entity", "user")
        user_prefix = "wb2x" if entity == "boot_camp_13_2nd_group_2nd" else entity

        # Model name
        model_name = self.cfg.model.get('name', 'model').replace("_", "-")

        # Preprocessing strategy (defaults to 'std')
        strategy = self.cfg.get("preprocessing", {}).get("strategy", "std")
        
        # Batch size (now correctly sourced from the dataset config)
        batch_size = self.cfg.dataset.get('batch_size', 'N/A')

        # Learning rate
        lr_float = self.cfg.training.optimizer.get('lr', 0)
        lr_str = f"{lr_float:.0e}".replace("e-0", "e") # Formats to 5e-6

        # Early stopping patience
        patience = self.cfg.training.early_stopping.get('patience')
        es_str = f"es{patience}" if patience else ""

        # --- 2. Assemble the Run Name ---
        
        # Create the detailed middle part of the name
        details_parts = [model_name, strategy, f"b{batch_size}", f"lr{lr_str}", es_str]
        model_details = "-".join(filter(None, details_parts)) # Filter removes empty strings

        # Final desired format
        run_name = f"{user_prefix}_(submission)_{model_details}_Rsubmission"
        
        return run_name

    def update_run_name_with_submission_score(self, rouge_score: float):
        """Updates the name of the finished run with its final ROUGE score."""
        if not (self.logger and self.logger.experiment):
            ic("WandB logger not initialized, cannot update run name.")
            return
        
        try:
            current_name = self.logger.experiment.name
            new_name = f"{current_name}-R{rouge_score:.4f}"
            self.logger.experiment.name = new_name
            ic(f"Updated WandB run name: {current_name} -> {new_name}")
        except Exception as e:
            ic(f"Failed to update WandB run name: {e}")

    def finish(self):
        """Finishes the WandB run."""
        if wandb.run:
            wandb.finish()
            ic("WandB run finished.")