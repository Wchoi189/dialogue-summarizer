# FILE: src/models/base_model.py
"""
Base Lightning module for dialogue summarization models.
Provides common functionality for training, validation, and testing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from icecream import ic
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from postprocessing.postprocessing import apply_post_processing
from utils.config_utils import ConfigManager
from evaluation.metrics import calculate_rouge_scores 
import re  # ✅ FIX: Import the 're' module

class BaseSummarizationModel(pl.LightningModule, ABC):
    """Base class for dialogue summarization models."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.training_cfg = cfg.training
        self.save_hyperparameters(cfg)

        # State Variables & Helpers
        self.validation_step_outputs: List[Dict] = []
        self.test_step_outputs: List[Dict] = []
        self._logged_postprocessing_stages = set()
        self.config_manager = ConfigManager()

        # Core Components (to be initialized by subclass)
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.generation_config = self._setup_generation_config()
        ic("BaseSummarizationModel initialized")

    @abstractmethod
    def _setup_model(self) -> None:
        """Setup the actual model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _setup_tokenizer(self) -> None:
        """Setup tokenizer. Must be implemented by subclasses."""
        pass

    def _calculate_metrics(self, predictions: list, targets: list) -> Dict[str, float]:
        """Calculates averaged ROUGE scores for a batch of predictions and targets."""
        
        # This function is now simplified to always return an averaged dictionary.
        rouge_scores = calculate_rouge_scores(
            predictions=predictions,
            references=targets,
            average=True  # Always request the averaged result
        )
        
        # The `isinstance` check is no longer needed because the return type is guaranteed.
        return cast(Dict[str, float], rouge_scores) # ✅ FIX: Use a type cast here to resolve Pylance error
    
    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]: # ✅ FIX: Specify the return type
        """Forward pass through the model."""
        assert self.model is not None, "Model not initialized."
        return cast(Dict[str, torch.Tensor], self.model(**kwargs)) # ✅ FIX: Use a type cast here to resolve Pylance error

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        model_inputs = {k: v for k, v in batch.items() if k != "sample_ids"}
        outputs = self.forward(**model_inputs)
        loss = outputs["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Periodically clear GPU memory based on config
        empty_cache_steps = self.cfg.pytorch.get("empty_cache_steps", 50)
        if (self.global_step + 1) % empty_cache_steps == 0:
            self.clear_gpu_memory()
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a single validation step."""
        output = self._shared_eval_step(batch, "val")
        self.validation_step_outputs.append(output)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a single test step."""
        output = self._shared_eval_step(batch, "test")
        self.test_step_outputs.append(output)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> List[str]:
        """Performs a single prediction step."""
        assert self.tokenizer is not None and self.model is not None, "Components must be initialized"
        generated_tokens = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **self.generation_config
        )
        raw_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        predictions = [self._apply_post_processing(text, stage="inference") for text in raw_preds]
        return predictions

    def on_validation_epoch_end(self):
        """Computes and logs metrics at the end of the validation epoch."""
        self._on_eval_epoch_end(self.validation_step_outputs, stage="val")

    def on_test_epoch_end(self):
        """Computes and logs metrics at the end of the test epoch."""
        self._on_eval_epoch_end(self.test_step_outputs, stage="test")

    def configure_optimizers(self) -> Any:
        """Configures the optimizer and learning rate scheduler."""
        optimizer_cfg = self.training_cfg.optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=optimizer_cfg.lr,
            weight_decay=optimizer_cfg.weight_decay
        )
        
        scheduler_cfg = self.training_cfg.lr_scheduler
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * scheduler_cfg.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=int(total_steps)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Gets a summary of the model's parameters."""
        if not self.model:
            return {"error": "Model not initialized"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming fp32
        }
    # ------------------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------------------
    
    def _shared_eval_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, Any]:
        """Performs a single evaluation step for either validation or testing."""
        assert self.tokenizer is not None and self.model is not None, "Components must be initialized"
        
        model_inputs = {k: v for k, v in batch.items() if k != "sample_ids"}
        loss = None
        targets = []
        
        # Conditionally handle loss and targets if labels exist
        if "labels" in batch:
            outputs = self.forward(**model_inputs)
            loss = outputs["loss"]
            self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            
            labels = batch["labels"]
            labels[labels == -100] = self.tokenizer.pad_token_id
            raw_targets = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
            # targets = [self._apply_post_processing(text, stage=stage) for text in raw_targets]
            # ✅ FIX: Apply simple cleaning, but DO NOT apply post-processing (including reverse swap)
            # We want to log the names as seen by the model, not revert them.
            targets = [self._clean_only_post_processing(text) for text in raw_targets]
        # Generate predictions
        generated_tokens = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **self.generation_config
        )
        raw_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        
        # ✅ FIX: Apply full post-processing ONLY to the model's predictions
        predictions = [self._apply_post_processing(text, stage=stage) for text in raw_preds]
        
        # Decode inputs for logging purposes
        raw_inputs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
        # inputs = [self._apply_post_processing(text, stage=stage) for text in raw_inputs]
        # ✅ FIX: Apply simple cleaning, but DO NOT apply post-processing (including reverse swap)
        inputs = [self._clean_only_post_processing(text) for text in raw_inputs]
        return {"loss": loss, "predictions": predictions, "targets": targets, "inputs": inputs}

    # ✅ NEW: Add a new helper function for basic cleaning without the full pipeline
    def _clean_only_post_processing(self, text: str) -> str:
        """Applies minimal post-processing for logging purposes (removes special tokens only)."""
        post_cfg = self.cfg.get("postprocessing", {})
        remove_tokens = post_cfg.get("remove_tokens", [])
        
        for token in remove_tokens:
            text = text.replace(token, "")
        
        # Also clean up extra spaces
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _on_eval_epoch_end(self, outputs: List[Dict], stage: str):
        """Centralized logic for processing epoch-end outputs."""
        if not outputs:
            return

        all_predictions = [p for out in outputs for p in out["predictions"]]
        all_targets = [t for out in outputs for t in out["targets"]]
        all_inputs = [i for out in outputs for i in out["inputs"]]

        
        if all_targets:
            rouge_scores = self._calculate_metrics(all_predictions, all_targets)
            
            # Filter for scalar metrics before logging
            scalar_rouge_scores = {k: v for k, v in rouge_scores.items() if not isinstance(v, list)}
            self.log_dict({f"{stage}/{k}": v for k, v in scalar_rouge_scores.items()})

        if stage == "val":
            self._log_wandb_validation_table(all_inputs, all_targets, all_predictions)
        
        outputs.clear()
        self.clear_gpu_memory()


    def _log_wandb_validation_table(self, inputs: List[str], targets: List[str], predictions: List[str]):
        """Logs a sample of predictions and metrics to a wandb.Table."""
        # Ensure we have a WandbLogger and the experiment is running
        if not isinstance(self.logger, WandbLogger) or not self.logger.experiment:
            return

        try:
            table = wandb.Table(columns=["Epoch", "Input", "Ground Truth", "Prediction", "ROUGE-1", "ROUGE-2", "ROUGE-L"])
            for i in range(min(len(predictions), 5)):
                # Calculate ROUGE scores for each individual sample
                sample_rouge = self._calculate_metrics(
                    predictions=[predictions[i]],
                    targets=[targets[i]],
        
            )
    
            
                table.add_data(
                    self.current_epoch,
                    inputs[i],
                    targets[i],
                    predictions[i],
                    f"{sample_rouge.get('rouge1_f', 0.0):.4f}",
                    f"{sample_rouge.get('rouge2_f', 0.0):.4f}",
                    f"{sample_rouge.get('rougeL_f', 0.0):.4f}"
                )
            
            # Use the PyTorch Lightning logger to correctly log the table
            self.logger.experiment.log({"validation_samples": table})

        except Exception as e:
            ic(f"WandB table logging failed: {e}")

    def _setup_generation_config(self) -> Dict[str, Any]:
        """Creates the generation config dictionary from the main config."""
        # Use self.cfg.generation as it's a top-level key
        gen_cfg = self.cfg.get("generation", {})
        return {
            "max_length": gen_cfg.get("max_length", 50),
            "min_length": gen_cfg.get("min_length", 10),
            "num_beams": gen_cfg.get("num_beams", 4),
            "repetition_penalty": gen_cfg.get("repetition_penalty", 1.5),
            "length_penalty": gen_cfg.get("length_penalty", 0.8),
            "no_repeat_ngram_size": gen_cfg.get("no_repeat_ngram_size", 3),
            "early_stopping": gen_cfg.get("early_stopping", True)
        }
    
    def _apply_post_processing(self, text: str, stage: str) -> str:
        """Applies all post-processing steps from the config."""
        post_cfg = self.cfg.get("postprocessing", {})
        
        # ✅ Call the single, centralized function
        return apply_post_processing(text, post_cfg)
    
    def clear_gpu_memory(self):
        """Clears GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()