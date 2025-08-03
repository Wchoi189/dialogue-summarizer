# FILE: src/models/kobart_model.py
"""
KoBART model implementation for Korean dialogue summarization.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional
import torch
from icecream import ic
from omegaconf import DictConfig
from transformers import AutoTokenizer, BartForConditionalGeneration
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from .base_model import BaseSummarizationModel
import wandb
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandbRun
    from pytorch_lightning.loggers import Logger
class KoBARTSummarizationModel(BaseSummarizationModel):
    """KoBART-specific implementation of the BaseSummarizationModel."""
    
    # ADD THIS TYPE HINT
    # model: BartForConditionalGeneration
    # We remove the type hint on this line and let the base class type declaration take precedence.
    
    def __init__(self, cfg: DictConfig):
        """Initializes the tokenizer, model, and resizes embeddings."""
        super().__init__(cfg)
        
        # 1. Setup the tokenizer and model
        self._setup_tokenizer()
        self._setup_model()
        
        # 2. Assert that components are valid before proceeding
        assert self.model is not None, "Model failed to initialize in _setup_model."
        assert self.tokenizer is not None, "Tokenizer failed to initialize in _setup_tokenizer."
        
        # 3. CRITICAL: Resize model embeddings to match the tokenizer's new vocabulary size
        original_vocab_size = self.model.config.vocab_size
        current_vocab_size = len(self.tokenizer)
        if current_vocab_size > original_vocab_size:
            ic(f"Resizing token embeddings: {original_vocab_size} -> {current_vocab_size}")
            self.model.resize_token_embeddings(current_vocab_size)

        # 4. Setup generation config for validation logging (AFTER tokenizer setup)
        self.generation_config = {
            "max_length": cfg.generation.get("max_length", 50),
            "num_beams": cfg.generation.get("num_beams", 4),
            "repetition_penalty": cfg.generation.get("repetition_penalty", 1.4),
            "early_stopping": True,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
        }

        ic(f"KoBARTSummarizationModel initialized with {self.get_parameter_count()} parameters")

    def _setup_tokenizer(self) -> None:
        """Initializes the tokenizer and adds special tokens from the config."""
        tokenizer_cfg = self.model_cfg.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_cfg.name_or_path,
            use_fast=tokenizer_cfg.get("use_fast", True)
        )
        # This assertion informs Pylance that the tokenizer is now valid
        assert self.tokenizer is not None, "Tokenizer failed to load from pretrained."
        
        # BEFORE:
        # special_tokens_list = self.cfg.preprocessing.get("special_tokens", [])

        # AFTER (✅ Add this logic to handle old and new checkpoints)
        if hasattr(self.cfg, "preprocessing"):
            # For new checkpoints with the centralized structure
            special_tokens_list = self.cfg.preprocessing.get("special_tokens", [])
        else:
            # Fallback for old checkpoints with the nested structure
            # special_tokens_list = self.cfg.dataset.preprocessing.get("special_tokens", [])
            special_tokens_list = self.cfg.dataset.preprocessing.get("special_tokens", [])

        if special_tokens_list:
            self.tokenizer.add_tokens([str(t) for t in special_tokens_list])


    def _setup_model(self) -> None:
        """Initializes the BART model."""
        model_obj = BartForConditionalGeneration.from_pretrained(
            self.model_cfg.model_name_or_path
        )
        
        # This logic correctly handles the two possible return types of from_pretrained
        if isinstance(model_obj, tuple):
            self.model = model_obj[0]
        else:
            self.model = model_obj
        
        # This assertion informs Pylance that the model is now valid
        assert self.model is not None, "Model failed to load from pretrained."

        if self.model_cfg.get("training_mode", {}).get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            ic("Gradient checkpointing enabled")

    def forward(self, **kwargs: torch.Tensor) -> Any:
        """Forward pass through the model."""
        assert self.model is not None, "Model has not been initialized."
        return self.model(**kwargs)

    def get_parameter_count(self) -> Dict[str, int]:
        """Gets a detailed count of model parameters."""
        if not self.model: return {"total": 0, "trainable": 0}
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
    
    def validation_step(self, batch, batch_idx):
        """Validation step with sample logging."""
        # Call parent validation step
        outputs = super().validation_step(batch, batch_idx)
        
        # Log validation samples table (only for first batch of first validation)
        if batch_idx == 0 and self.current_epoch == 0:
            self._log_validation_samples(batch, outputs)
        
        return outputs

    def _log_validation_samples(self, batch, outputs, num_samples: int = 5):
        """Log validation samples to WandB."""
        try:
            # Find WandB logger with proper typing
            wandb_logger: Optional[WandbLogger] = None
            
            if self.trainer and hasattr(self.trainer, 'loggers'):
                for logger in self.trainer.loggers:
                    if isinstance(logger, WandbLogger):
                        wandb_logger = logger
                        break
            elif hasattr(self.trainer, 'logger') and isinstance(self.trainer.logger, WandbLogger):
                wandb_logger = self.trainer.logger
            
            if not wandb_logger:
                ic("WandB logger not found")
                return

            # Type-safe access to experiment
            experiment: Optional["WandbRun"] = wandb_logger.experiment
            if not experiment:
                ic("WandB experiment not initialized")
                return
            
            # Add an assertion to satisfy the type checker that the tokenizer is not None
            assert self.tokenizer is not None, "Tokenizer is None in _log_validation_samples"
            ic(f"Found WandB logger: {type(wandb_logger)}")
            
            # Get predictions for the batch
            input_ids = batch["input_ids"][:num_samples]
            attention_mask = batch["attention_mask"][:num_samples]
            
            # Add an assertion to satisfy the type checker that the model is not None
            assert self.model is not None, "Model is None in _log_validation_samples"            
            
            # Generate predictions
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **self.generation_config
                )
            
            # Decode inputs, targets, and predictions
            samples_data = []
            for i in range(min(num_samples, len(input_ids))):
                # Decode input
                input_text = self.tokenizer.decode(
                    input_ids[i], skip_special_tokens=True
                )
                
                # Decode target (handle -100 labels)
                target_ids = batch["labels"][i]
                target_ids = target_ids[target_ids != -100]  # Remove padding tokens
                target_text = self.tokenizer.decode(
                    target_ids, skip_special_tokens=True
                )
                
                # Decode prediction
                pred_text = self.tokenizer.decode(
                    generated_ids[i], skip_special_tokens=True
                )
                
                # Calculate ROUGE scores for this sample
                rouge_scores = self._calculate_sample_rouge(pred_text, target_text)
                
                samples_data.append([
                    input_text[:200] + "..." if len(input_text) > 200 else input_text,
                    target_text,
                    pred_text,
                    rouge_scores["rouge1_f"],
                    rouge_scores["rouge2_f"],
                    rouge_scores["rougeL_f"]
                ])

            # Create WandB table
            table = wandb.Table(
                columns=["Input", "Ground Truth", "Prediction", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
                data=samples_data
            )
            # ✅ FIX: Log the table using the global `wandb.log` function
            wandb.log({"validation_samples": table})
            
            # Log the table
            wandb_logger.experiment.log({"validation_samples": table})
            ic(f"✅ Logged {len(samples_data)} validation samples to WandB")
                
        except Exception as e:
            ic(f"❌ Failed to log validation samples: {e}")
            import traceback
            ic(f"Full traceback: {traceback.format_exc()}")

    def _calculate_sample_rouge(self, prediction: str, reference: str) -> Dict[str, Any]:
        """Calculate ROUGE scores for a single sample."""
        try:
            # ✅ Import and use the new centralized function from metrics.py
            from evaluation.metrics import calculate_rouge_scores
            scores = calculate_rouge_scores(
                predictions=[prediction],
                references=[reference],
                average=True
            )
            assert isinstance(scores, dict)
            return scores
        except Exception as e:
            ic(f"Failed to calculate ROUGE for validation sample: {e}")
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    
    def on_validation_epoch_start(self):
        """Called at the start of validation epoch."""
        ic(f"Validation epoch {self.current_epoch} starting")
        super().on_validation_epoch_start()

    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        ic(f"Validation epoch {self.current_epoch} ending")
        
        # Add explicit type checks before accessing attributes
        if hasattr(self, 'logger') and self.logger:
            if isinstance(self.logger, list):
                ic(f"Multiple loggers found: {len(self.logger)}")
                for i, logger in enumerate(self.logger):
                    ic(f"Logger {i}: {type(logger)}")
                    if isinstance(logger, WandbLogger) and hasattr(logger, 'experiment'):
                        ic(f"Logger {i} experiment type: {type(logger.experiment)}")
            else:
                ic(f"Single logger type: {type(self.logger)}")
                if isinstance(self.logger, WandbLogger) and hasattr(self.logger, 'experiment'): 
                    ic(f"Logger experiment type: {type(self.logger.experiment)}")
        
        super().on_validation_epoch_end()