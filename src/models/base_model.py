# FILE: src/models/base_model.py
"""
Base Lightning module for dialogue summarization models.
Provides common functionality for training, validation, and testing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from icecream import ic
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from utils.config_utils import ConfigManager
from evaluation.metrics import RougeCalculator

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
        self.rouge_calculator = RougeCalculator()
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

    def forward(self, **kwargs: torch.Tensor) -> Any:
        """Forward pass through the model."""
        assert self.model is not None, "Model not initialized."
        return self.model(**kwargs)

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
            loss = outputs.loss
            self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            
            labels = batch["labels"]
            labels[labels == -100] = self.tokenizer.pad_token_id
            raw_targets = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
            targets = [self._apply_post_processing(text, stage=stage) for text in raw_targets]

        # Generate predictions
        generated_tokens = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **self.generation_config
        )
        raw_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        predictions = [self._apply_post_processing(text, stage=stage) for text in raw_preds]

        # Decode inputs for logging purposes
        raw_inputs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
        inputs = [self._apply_post_processing(text, stage=stage) for text in raw_inputs]

        return {"loss": loss, "predictions": predictions, "targets": targets, "inputs": inputs}

    def _on_eval_epoch_end(self, outputs: List[Dict], stage: str):
        """Centralized logic for processing epoch-end outputs."""
        if not outputs:
            return

        all_predictions = [p for out in outputs for p in out["predictions"]]
        all_targets = [t for out in outputs for t in out["targets"]]
        all_inputs = [i for out in outputs for i in out["inputs"]]

       
        if all_targets:
            rouge_scores = self.rouge_calculator.calculate_rouge(all_predictions, all_targets)
            
            # Filter for scalar metrics before logging
            scalar_rouge_scores = {k: v for k, v in rouge_scores.items() if not isinstance(v, list)}
            self.log_dict({f"{stage}/{k}": v for k, v in scalar_rouge_scores.items()})

        if stage == "val":
            self._log_wandb_validation_table(all_inputs, all_targets, all_predictions)
        
        outputs.clear()
        self.clear_gpu_memory()

    # def _log_wandb_validation_table(self, inputs: List[str], targets: List[str], predictions: List[str]):
    #     """Logs a sample of predictions, targets, and metrics to a wandb.Table."""
    #     try:
    #         if isinstance(self.logger, WandbLogger) and self.logger.experiment:
    #             table = wandb.Table(columns=["Epoch", "Input", "Ground Truth", "Prediction", "ROUGE-1", "ROUGE-2", "ROUGE-L"])
                
    #             for i in range(min(len(predictions), 5)):
    #                 sample_rouge = self.rouge_calculator.calculate_rouge([predictions[i]], [targets[i]])
    #                 table.add_data(
    #                     self.current_epoch,
    #                     inputs[i],
    #                     targets[i],
    #                     predictions[i],
    #                     f"{sample_rouge.get('rouge1_f', 0.0):.4f}",
    #                     f"{sample_rouge.get('rouge2_f', 0.0):.4f}",
    #                     f"{sample_rouge.get('rougeL_f', 0.0):.4f}"
    #                 )
                
    #             self.logger.experiment.log({"validation_samples": table})
    #     except Exception as e:
    #         ic(f"WandB table logging failed (this is OK): {e}")

    def _log_wandb_validation_table(self, inputs: List[str], targets: List[str], predictions: List[str]):
        """Logs a sample of predictions and metrics to a wandb.Table."""
        # Ensure we have a WandbLogger and the experiment is running
        if not isinstance(self.logger, WandbLogger) or not self.logger.experiment:
            return

        try:
            table = wandb.Table(columns=["Epoch", "Input", "Ground Truth", "Prediction", "ROUGE-1", "ROUGE-2", "ROUGE-L"])
            
            # Log up to 5 samples from the validation set
            for i in range(min(len(predictions), 5)):
                # Calculate ROUGE scores for each individual sample
                sample_rouge = self.rouge_calculator.calculate_rouge(
                    predictions=[predictions[i]],
                    references=[targets[i]]
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
        # try:
        #     post_cfg_name = self.cfg.postprocessing.get(stage, "validation")
        #     post_cfg = self.config_manager.load_postprocessing_config(post_cfg_name)
        # except (FileNotFoundError, AttributeError):
        #     post_cfg = {}
        
        # tokens_to_remove = post_cfg.get("remove_tokens", ["<s>", "</s>", "<pad>","<usr>"])
        # for token in tokens_to_remove:
        #     text = text.replace(token, "")

        # text = text.strip()
        # text = " ".join(text.split())
        # return text

            # Get post-processing config
        post_cfg = self.cfg.get("postprocessing", {})
        
        if not post_cfg:
            # Fallback to basic cleaning if no config
            text = text.replace("<usr>", "").replace("<s>", "").replace("</s>", "").replace("<pad>", "")
            return text.strip()
        
        # # Use the comprehensive post-processing method
        return self._apply_comprehensive_post_processing(text, post_cfg)
    
    def _apply_comprehensive_post_processing(self, text: str, post_cfg: dict) -> str:
        """Apply comprehensive post-processing."""
        
        # 1. Remove unwanted tokens (including <usr>)
        remove_tokens = post_cfg.get("remove_tokens", [])
        # Always remove <usr> token
        if "<usr>" not in remove_tokens:
            remove_tokens = remove_tokens + ["<usr>"]
        
        for token in remove_tokens:
            text = text.replace(token, "")
        
        # 2. Text cleaning
        text_cleaning = post_cfg.get("text_cleaning", {})
        
        if text_cleaning.get("strip_whitespace", True):
            text = text.strip()
        
        if text_cleaning.get("normalize_whitespace", True):
            import re
            text = re.sub(r'\s+', ' ', text)
        
        if text_cleaning.get("remove_empty_lines", True):
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = ' '.join(lines)
        
        # 3. Remove repeated phrases (fix the repetitive generation)
        if text_cleaning.get("remove_repetitive_phrases", True):
            text = self._remove_repetitive_phrases(text)
        
        # 4. Fix incomplete sentences
        advanced_cfg = post_cfg.get("advanced", {})
        if advanced_cfg.get("fix_incomplete_sentences", True):
            text = self._fix_incomplete_sentences(text, advanced_cfg)
        
        # 5. Korean-specific cleaning
        korean_cfg = post_cfg.get("korean_specific", {})
        
        # Only remove special markers if explicitly requested (keep #Person# tokens)
        if korean_cfg.get("remove_special_markers", False):
            import re
            # Only remove non-person markers
            text = re.sub(r'#(?!Person\d+#)\w+#', '', text)
            text = ' '.join(text.split())
        
        # 5. Advanced cleaning
        advanced_cfg = post_cfg.get("advanced", {})
        min_length = advanced_cfg.get("min_length", 5)
        
        if len(text.strip()) < min_length:
            return "요약을 생성할 수 없습니다."
        
        return text.strip()

    def _remove_repetitive_phrases(self, text: str) -> str:
        """Remove repetitive phrases from generated text."""
        sentences = text.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        return '. '.join(unique_sentences).strip()
    
    def _fix_incomplete_sentences(self, text: str, advanced_cfg: dict) -> str:
        """Fix incomplete sentences by removing them or adding proper endings."""
        import re
        
        # Korean sentence endings that indicate completeness
        korean_endings = advanced_cfg.get("required_punctuation", 
                                        [".", "!", "?", "다", "요", "습니다", "니다"])
        
        # Check if text ends with proper punctuation or Korean endings
        text = text.strip()
        if not text:
            return text
        
        # Check for proper endings
        has_proper_ending = False
        
        # Check for punctuation
        if text[-1] in [".", "!", "?"]:
            has_proper_ending = True
        
        # Check for Korean verb/adjective endings
        # for ending in ["다", "요", "습니다", "니다", "네요", "어요", "아요"]:
        #     if text.endswith(ending):
        #         has_proper_ending = True
        #         break
        
        if has_proper_ending:
            return text
        
        # If incomplete, try to fix it
        # Option 1: Remove the last incomplete sentence
        sentences = re.split(r'[.!?]', text)
        if len(sentences) > 1:
            # Keep all complete sentences, remove the last incomplete one
            complete_sentences = sentences[:-1]
            if complete_sentences:
                # Reconstruct with original punctuation
                result = ""
                parts = re.split(r'([.!?])', text)
                for i in range(0, len(parts)-2, 2):  # Skip the last incomplete part
                    result += parts[i] + (parts[i+1] if i+1 < len(parts) else "")
                return result.strip()
        
        # Option 2: Add appropriate ending based on context
        # For Korean, add "습니다" for formal endings
        # if text.endswith(("합니", "입니", "갑니", "습니")):
        #     return text + "다."
        # elif text.endswith(("해", "가", "는", "을", "를", "에", "의")):
        #     return text + "요."
        # else:
        #     # If we can't fix it properly, add a period
        #     return text + "."
        
        # Fallback: if no other condition is met, return the original text.
        return text
    
    def clear_gpu_memory(self):
        """Clears GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()