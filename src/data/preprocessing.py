# FILE: src/data/preprocessing.py
"""
Data preprocessing utilities for Korean dialogue summarization.
Handles text cleaning, tokenization, and special token management.
"""

import logging
import re
from typing import Dict, List, Optional

import pandas as pd
from icecream import ic
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)


class DialoguePreprocessor:
    """Enhanced preprocessor for Korean dialogue data."""
    
    def __init__(self, preprocessing_cfg: DictConfig, tokenizer: PreTrainedTokenizerFast):
        """Initializes the preprocessor with configuration and a tokenizer."""
        self.cfg = preprocessing_cfg
        self.tokenizer = tokenizer
        
        self._setup_special_tokens()
        
        ic(f"DialoguePreprocessor initialized with {len(self.tokenizer)} tokens")
    
    def _setup_special_tokens(self) -> None:
        """Adds special tokens from the config to the tokenizer's vocabulary."""
        additional_tokens = self.cfg.get("special_tokens", [])
        if additional_tokens:
            num_added = self.tokenizer.add_tokens([str(t) for t in additional_tokens])
            if num_added > 0:
                ic(f"Added {num_added} new tokens to the tokenizer vocabulary.")

    def _clean_text(self, text: str) -> str:
        """Performs basic cleaning of raw text strings."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.replace('\\n', ' ')
        text = re.sub(r'<[^>]+>', ' ', text)
        if self.cfg.get("normalize_whitespace", True):
            text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def preprocess_dialogue(self, dialogue: str) -> str:
        """Cleans and prepares a single dialogue string."""
        return self._clean_text(dialogue)

    def preprocess_summary(self, summary: str) -> str:
        """Cleans and prepares a single summary string."""
        return self._clean_text(summary)

    def prepare_inputs(
        self,
        dialogue: str,
        summary: Optional[str] = None,
        is_inference: bool = False
    ) -> BatchEncoding:
        """Preprocesses and tokenizes a single dialogue-summary pair."""
        dialogue = self.preprocess_dialogue(dialogue)
        
        model_inputs = self.tokenizer(
            dialogue,
            max_length=self.cfg.max_input_length,
            truncation=True,
            padding=False,  # Padding is handled by the collate function
            return_tensors="pt" if not is_inference else None
        )

        if summary is not None:
            summary = self.preprocess_summary(summary)
            labels = self.tokenizer(
                summary,
                max_length=self.cfg.max_target_length,
                truncation=True,
                padding=False,
                return_tensors="pt" if not is_inference else None
            )
            model_inputs['labels'] = labels['input_ids']

        return model_inputs

    def batch_preprocess(
        self,
        dialogues: List[str],
        summaries: Optional[List[str]] = None,
        is_inference: bool = False
    ) -> BatchEncoding:
        """Preprocesses and tokenizes a batch of dialogues and summaries."""
        ic(f"Preprocessing batch of {len(dialogues)} samples")
        
        processed_dialogues = [self.preprocess_dialogue(d) for d in dialogues]
        
        model_inputs = self.tokenizer(
            text=processed_dialogues,
            max_length=self.cfg.max_input_length,
            truncation=True,
            padding=True, # Batch tokenization can handle padding directly
            return_tensors="pt"
        )

        if summaries is not None and not is_inference:
            processed_summaries = [self.preprocess_summary(s) for s in summaries]
            labels = self.tokenizer(
                text=processed_summaries,
                max_length=self.cfg.max_target_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            model_inputs['labels'] = labels['input_ids']

        ic(f"Batch preprocessing complete for {len(dialogues)} samples")
        return model_inputs

    def decode_outputs(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Decodes token IDs back to a clean text string."""
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        return self._clean_text(text)


def create_preprocessor(cfg: DictConfig) -> DialoguePreprocessor:
    """Factory function to create a preprocessor with a tokenizer."""
    model_name = cfg.model.tokenizer.name_or_path
    ic(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return DialoguePreprocessor(cfg.preprocessing, tokenizer)