"""
Base Lightning module for dialogue summarization models.
Provides common functionality for training, validation, and testing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from icecream import ic
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import get_cosine_schedule_with_warmup
from evaluation.metrics import RougeCalculator
import wandb
from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger(__name__)


class BaseSummarizationModel(pl.LightningModule, ABC):
    """Base class for dialogue summarization models."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize base model.
        """
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.training_cfg = cfg.training
        
        # Save hyperparameters
        self.save_hyperparameters(cfg)
        
        # MODIFIED: Provide a clear type hint for self.model
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer = None

        # Ensure model and tokenizer are set up if subclass implements setup methods
        if hasattr(self, "_setup_model"):
            self._setup_model()
        if hasattr(self, "_setup_tokenizer"):
            self._setup_tokenizer()
        
        # Metrics storage
        self.training_metrics = {}
        self.validation_metrics = {}
        
        # Store validation outputs for epoch end processing (PyTorch Lightning v2.0+)
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Generation config
        self.generation_config = self._setup_generation_config()
       
        self.rouge_calculator = RougeCalculator()
        ic(f"BaseSummarizationModel initialized")
    
    def clear_gpu_memory(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "cached": 0.0, "total": 0.0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            "allocated": allocated,
            "cached": cached,
            "total": total,
            "free": total - allocated
        }
    
    def log_gpu_memory(self, step_name: str = "") -> None:
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            memory = self.get_gpu_memory_usage()
            ic(f"{step_name} GPU Memory - Allocated: {memory['allocated']:.2f}GB, "
               f"Cached: {memory['cached']:.2f}GB, Free: {memory['free']:.2f}GB")
    
    @abstractmethod
    def _setup_model(self) -> None:
        """Setup the actual model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _setup_tokenizer(self) -> None:
        """Setup tokenizer. Must be implemented by subclasses."""
        pass
    
    def _setup_generation_config(self) -> Dict[str, Any]:
        """Setup generation configuration."""
        gen_cfg = self.training_cfg.get("generation", {})
        
        num_beams = gen_cfg.get("num_beams", 4)
        early_stopping = gen_cfg.get("early_stopping", True)
        
        # Fix beam search configuration: if num_beams=1, disable early_stopping
        if num_beams == 1:
            early_stopping = False
        
        return {
            "max_length": gen_cfg.get("max_length", 50),
            "min_length": gen_cfg.get("min_length", 1),
            "num_beams": num_beams,
            "no_repeat_ngram_size": gen_cfg.get("no_repeat_ngram_size", 2),
            "early_stopping": early_stopping,
            "do_sample": False,
            "length_penalty": gen_cfg.get("length_penalty", 1.0),
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input
            decoder_input_ids: Decoder input token IDs
            decoder_attention_mask: Attention mask for decoder
            labels: Target labels for training
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized. Make sure the subclass sets up 'self.model' in _setup_model().")
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        # # Log memory before step
        # if batch_idx % 50 == 0:  # Log every 50 steps
        #     self.log_gpu_memory(f"Training step {batch_idx} start")
        # Log only on the first step of the entire training run
        if self.global_step == 0:
             self.log_gpu_memory(f"Training step {batch_idx} start")
        outputs = self.forward(**{k: v for k, v in batch.items() if k != "sample_ids"})
        loss = outputs["loss"]
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store metrics
        self.training_metrics["loss"] = loss.item()
        
        # Clear GPU memory periodically
        if batch_idx % 10 == 0:  # Clear every 10 steps
            self.clear_gpu_memory()
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with outputs
        """
        
        if batch_idx == 0:
            self.log_gpu_memory(f"Validation step {batch_idx} start")
        
        # Forward pass for loss calculation
        outputs = self.forward(**{k: v for k, v in batch.items() if k != "sample_ids"})
        loss = outputs["loss"]
        
        # Clear memory after forward pass
        self.clear_gpu_memory()
        
        # Generate predictions for evaluation
        predictions = self.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        # Clear memory after generation
        self.clear_gpu_memory()
        
        # Decode predictions and targets
        pred_texts = self._decode_predictions(predictions)
        target_texts = self._decode_targets(batch["labels"])
        input_texts = self._decode_inputs(batch["input_ids"])
        
        # DEBUG: Log sample predictions to help debug ROUGE issues
        if batch_idx == 0:  # Only log first batch to avoid spam
            ic(f"Input shape: {batch['input_ids'].shape}")
            ic(f"Predictions shape: {predictions.shape}")
            ic(f"Sample input: {input_texts[0][:200]}...")
            ic(f"Sample target: {target_texts[0][:200]}...")
            ic(f"Sample prediction: {pred_texts[0][:200]}...")
            ic(f"Prediction length: {len(pred_texts[0])}")
            ic(f"Target length: {len(target_texts[0])}")
        
        step_output = {
            "loss": loss.detach(),
            "predictions": pred_texts,
            "targets": target_texts,
            "inputs": input_texts,
            "sample_ids": batch["sample_ids"]
        }
        
        # Store outputs for epoch end processing (PyTorch Lightning v2.0+)
        self.validation_step_outputs.append(step_output)
        
        return step_output
    
    def on_validation_epoch_end(self) -> None:
        """
        Process validation epoch end and log a table of predictions to WandB.
        """
        outputs = self.validation_step_outputs
        if not outputs:
            return

        
        assert self.model is not None  # Ensure model is initialized
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        all_predictions, all_targets, all_inputs = [], [], []
        for output in outputs:
            all_predictions.extend(output["predictions"])
            all_targets.extend(output["targets"])
            all_inputs.extend(output["inputs"])

        # Calculate overall ROUGE scores
        rouge_scores = self.rouge_calculator.calculate_rouge(
            predictions=all_predictions,
            references=all_targets,
            average=True
        )
        
        # DEBUG: Log sample texts to debug ROUGE issues
        ic(f"Total predictions: {len(all_predictions)}")
        ic(f"Total targets: {len(all_targets)}")
        if all_predictions and all_targets:
            ic(f"First prediction: '{all_predictions[0]}'")
            ic(f"First target: '{all_targets[0]}'")
            ic(f"Prediction empty?: {all_predictions[0] == ''}")
            ic(f"Target empty?: {all_targets[0] == ''}")
        
        # Log metrics to WandB
        self.log("val/loss", avg_loss, prog_bar=True)
        self.log_dict({f"val/{k}": v for k, v in rouge_scores.items()}, sync_dist=True)  # type: ignore

        # ✅ CREATE AND LOG A WANDB TABLE WITH SAMPLE PREDICTIONS
        try:
            if wandb.run is not None:
                # Create a table with columns: Epoch, Input, Ground Truth, Prediction, ROUGE-1
                table = wandb.Table(columns=["Epoch", "Input", "Ground Truth", "Prediction", "ROUGE-1"])
                
                # Take a sample of 5 from the validation set to log
                for i in range(min(len(all_predictions), 5)):
                    input_text = all_inputs[i][:100] + "..." if len(all_inputs[i]) > 100 else all_inputs[i]
                    target_text = all_targets[i]
                    pred_text = all_predictions[i]
                    
                    # Calculate ROUGE for this single sample
                    sample_rouge = self.rouge_calculator.calculate_rouge([pred_text], [target_text])
                    rouge1_f = sample_rouge.get('rouge1_f', 0.0)

                    table.add_data(self.current_epoch, input_text, target_text, pred_text, f"{rouge1_f:.4f}")
                
                wandb.log({"validation_samples": table})
                ic("WandB table logged successfully")
        except Exception as e:
            ic(f"WandB logging failed (this is OK): {e}")
        # Store metrics and clear outputs
        self.validation_metrics = {"loss": avg_loss.item(), **rouge_scores}
        ic(f"Validation metrics: {self.validation_metrics}")
        self.validation_step_outputs.clear()
        self.clear_gpu_memory()
        self.log_gpu_memory("Validation epoch end")
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Test step.
        """
        # Forward pass for loss calculation
        outputs = self.forward(**{k: v for k, v in batch.items() if k != "sample_ids"})
        loss = outputs.loss
        
        # Generate predictions for evaluation
        predictions = self.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        # Decode predictions and targets
        pred_texts = self._decode_predictions(predictions)
        target_texts = self._decode_targets(batch["labels"])
        
        step_output = {
            "loss": loss.detach(),
            "predictions": pred_texts,
            "targets": target_texts,
            "sample_ids": batch["sample_ids"]
        }
        
        self.test_step_outputs.append(step_output)
        return step_output
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        Prediction step (same as test step).
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Dictionary with outputs
        """
        return self.test_step(batch, batch_idx)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        assert self.model is not None  # Ensure model is initialized
        assert self.tokenizer is not None  # Ensure tokenizer is initialized
        
        # DEBUG: Log generation parameters occasionally
        if hasattr(self, '_debug_generation_logged') and not self._debug_generation_logged:
            ic(f"Generation config: {self.generation_config}")
            ic(f"Input shape: {input_ids.shape}")
            ic(f"Max input length: {input_ids.shape[1]}")
            self._debug_generation_logged = True
        elif not hasattr(self, '_debug_generation_logged'):
            self._debug_generation_logged = False
        
        # Merge generation config with kwargs
        gen_kwargs = {**self.generation_config, **kwargs}
        
        # Set pad token if not specified
        if "pad_token_id" not in gen_kwargs and hasattr(self.tokenizer, "pad_token_id"):
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        return outputs
    
    def _decode_predictions(self, predictions: torch.Tensor) -> List[str]:
        """
        Decode prediction token IDs to text.
        
        Args:
            predictions: Generated token IDs
            
        Returns:
            List of decoded text strings
        """
        assert self.tokenizer is not None  # Ensure tokenizer is initialized
        decoded = []
        for pred in predictions:
            text = self.tokenizer.decode(
                pred,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded.append(text.strip())
        
        return decoded
    
    def _decode_targets(self, labels: torch.Tensor) -> List[str]:
        """
        Decode target labels to text.
        
        Args:
            labels: Target token IDs
            
        Returns:
            List of decoded text strings
        """
        assert self.tokenizer is not None  # Ensure tokenizer is initialized
        # Replace -100 with pad token for decoding
        labels = labels.clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        
        decoded = []
        for label in labels:
            text = self.tokenizer.decode(
                label,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded.append(text.strip())
        
        return decoded
    def _decode_inputs(self, input_ids: torch.Tensor) -> List[str]:
        """Decode input token IDs to text."""
        assert self.tokenizer is not None  # Ensure tokenizer is initialized
        decoded = []
        for inp in input_ids:
            text = self.tokenizer.decode(
                inp,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded.append(text.strip())
        return decoded    

    def configure_optimizers(self) -> Any:
        """Configure optimizers and learning rate schedulers."""
        # Setup optimizer
        optimizer_cfg = self.training_cfg.optimizer
        
        if optimizer_cfg.name.lower() == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay,
                betas=optimizer_cfg.get("betas", [0.9, 0.999]),
                eps=optimizer_cfg.get("eps", 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name}")
        
        # Setup scheduler
        scheduler_cfg = self.training_cfg.lr_scheduler
        
        if scheduler_cfg.name.lower() == "cosine":
            # Calculate total steps for warmup
            if hasattr(self.trainer, "estimated_stepping_batches"):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                # Fallback calculation
                total_steps = self.training_cfg.max_epochs * 1000  # Rough estimate
            
            warmup_steps = int(total_steps * scheduler_cfg.warmup_ratio)
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=int(total_steps)
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
        else:
            return {"optimizer": optimizer}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        if self.model is None:
            return {"error": "Model not initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming fp32
        }
    
    def on_test_epoch_end(self) -> None:
        """
        Process test epoch end.
        """
        outputs = self.test_step_outputs
        
        if not outputs:
            return

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        
        all_predictions = []
        all_targets = []
        for output in outputs:
            all_predictions.extend(output["predictions"])
            all_targets.extend(output["targets"])
        
        rouge_scores = self.rouge_calculator.calculate_rouge(
            predictions=all_predictions,
            references=all_targets,
            average=True
        )
        
        flat_scores = {}
        for main_key, value_dict in rouge_scores.items():
            if isinstance(value_dict, dict):
                for sub_key, score in value_dict.items():
                    # Creates keys like "rouge1_f", "rouge1_p", etc.
                    flat_scores[f"{main_key}_{sub_key}"] = score
            else:
                # Handles cases where the score is already flat
                flat_scores[main_key] = value_dict

        # Log the flattened scores
        self.log_dict(
            {f"eval/{k}": v for k, v in flat_scores.items()},
            sync_dist=True
)
        
        ic(f"Final Evaluation metrics: {rouge_scores}")
        self.test_step_outputs.clear()   