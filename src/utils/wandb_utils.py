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
        
        # 📝 NEW: Get the experiment name
        experiment_name = self.cfg.get('experiment_name', 'no-exp')
        
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

        # # --- 2. Assemble the Run Name ---
        
        # # Create the detailed middle part of the name
        # details_parts = [model_name, strategy, f"b{batch_size}", f"lr{lr_str}", es_str]
        # model_details = "-".join(filter(None, details_parts)) # Filter removes empty strings

        # # Final desired format - use placeholder that can be replaced later
        # run_name = f"{user_prefix}_(submission)_{model_details}_PLACEHOLDER"
        # --- 2. Assemble the Run Name ---
        
        # 📝 FIX: Include the experiment name in the details list
        details_parts = [experiment_name, model_name, strategy, f"b{batch_size}", f"lr{lr_str}", es_str]
        model_details = "-".join(filter(None, details_parts)) # Filter removes empty strings

        # Final desired format - use placeholder that can be replaced later
        run_name = f"{user_prefix}_(submission)_{model_details}_PLACEHOLDER"

        return run_name
    
    def finalize_run_name_with_score(self, rouge_score: float):
        """Replace placeholder with actual score - call this at the end of training."""
        if not (self.logger and self.logger.experiment):
            return
        
        try:
            current_name = self.logger.experiment.name
            final_name = current_name.replace("_PLACEHOLDER", f"_R{rouge_score:.4f}")
            
            # Update all the places where the name should appear
            experiment = self.logger.experiment
            
            # 1. Summary for UI display
            experiment.summary.update({
                "final_run_name": final_name,
                "final_rouge_score": rouge_score,
                "run_completed": True
            })
            
            # 2. Tags
            score_tag = f"R{rouge_score:.4f}"
            current_tags = list(experiment.tags) if experiment.tags else []
            if score_tag not in current_tags:
                current_tags.append(score_tag)
                experiment.tags = current_tags
            
            # 3. Config update
            experiment.config.update({
                "final_run_name": final_name,
                "final_score": rouge_score
            })
            
            # 4. Log as metric
            experiment.log({
                "final_submission_score": rouge_score,
                "run_name_finalized": 1
            })
            experiment.name = final_name  # Update the run name in WandB
            ic(f"✅ Finalized run name: {current_name} -> {final_name}")
            ic(f"   Score: R{rouge_score:.4f}")
            
        except Exception as e:
            ic(f"❌ Failed to finalize run name: {e}")

    def update_run_name_with_submission_score(self, rouge_score: float):
        """Updates the run with final ROUGE score using tags and summary."""
        if not (self.logger and self.logger.experiment):
            ic("WandB logger not initialized, cannot update run info.")
            return
        
        try:
            experiment = self.logger.experiment
            
            # 1. Update run summary with final score
            experiment.summary["final_rouge_f"] = rouge_score
            
            # 2. Add final score as a tag (handle tuple/list conversion)
            score_tag = f"R{rouge_score:.4f}"
            current_tags = experiment.tags
            
            if current_tags is None:
                current_tags = []
            elif isinstance(current_tags, tuple):
                current_tags = list(current_tags)
            elif not isinstance(current_tags, list):
                current_tags = [current_tags] if current_tags else []
            
            if score_tag not in current_tags:
                current_tags.append(score_tag)
                experiment.tags = current_tags
            
            # 3. Log the final score as a metric
            experiment.log({"final_submission_score": rouge_score})
            
            # 4. Update display name in summary (this shows in UI)
            current_name = experiment.name
            display_name = f"{current_name}_R{rouge_score:.4f}"
            experiment.summary["display_name"] = display_name
            experiment.name = display_name  # Update the run name
            # 5. Try to update the actual run name (this may or may not work)
            try:
                import wandb
                if wandb.run and wandb.run.id == experiment.id:
                    # Create a new name that replaces the placeholder
                    new_name = current_name.replace("_Rsubmission", f"_R{rouge_score:.4f}")
                    wandb.run.name = new_name
                    ic(f"✅ Successfully updated run name: {current_name} -> {new_name}")
                else:
                 ic("⚠️ Could not update run name directly, but summary and tags updated")
            except Exception as name_error:
                ic(f"⚠️ Run name update failed (expected): {name_error}")
            
            # 6. Add a config entry that shows the final name
            experiment.config.update({"final_run_name": display_name})
            
            ic(f"✅ Updated WandB run with final score: R{rouge_score:.4f}")
            ic(f"   - Tags: {current_tags}")
            ic(f"   - Display name: {display_name}")
            
        except Exception as e:
            ic(f"❌ Failed to update WandB run with score: {e}")
        
    def finish(self):
        """Finishes the WandB run."""
        if wandb.run:
            wandb.finish()
            ic("WandB run finished.")

    def debug_wandb_state(self):
        """Debug current WandB state."""
        ic("=== WandB Debug Info ===")
        ic(f"Logger initialized: {self.logger is not None}")
        
        if self.logger:
            ic(f"Logger type: {type(self.logger)}")
            ic(f"Experiment available: {hasattr(self.logger, 'experiment')}")
            
            if hasattr(self.logger, 'experiment') and self.logger.experiment:
                ic(f"Run name: {self.logger.experiment.name}")
                ic(f"Run ID: {self.logger.experiment.id}")
                ic(f"Project: {self.logger.experiment.project}")
        
        # Check wandb module state
        import wandb
        ic(f"WandB run active: {wandb.run is not None}")
        if wandb.run:
            ic(f"Active run name: {wandb.run.name}")
            ic(f"Active run ID: {wandb.run.id}")        