# FILE: src/models/kobart_model.py
"""
KoBART model implementation for Korean dialogue summarization.
"""
import logging
from typing import Any, Dict
import torch
from icecream import ic
from omegaconf import DictConfig
from transformers import AutoTokenizer, BartForConditionalGeneration

from .base_model import BaseSummarizationModel

logger = logging.getLogger(__name__)

class KoBARTSummarizationModel(BaseSummarizationModel):
    """KoBART-specific implementation of the BaseSummarizationModel."""

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