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
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)


class DialoguePreprocessor:
    # ✅ Change the __init__ signature to accept the full config
    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizerFast):
        """Initializes the preprocessor with the full configuration."""
        self.cfg = cfg
        # We get the specific preprocessing block from the full cfg
        self.preprocessing_cfg = cfg.preprocessing 
        # self.cfg = cfg.preprocessing
        self.tokenizer = tokenizer
        
        # Handle token swapping config
        self.token_swapping_cfg = cfg.preprocessing.get("token_swapping", {"enable": False})
        if self.token_swapping_cfg.get("enable"):
            ic("Token swapping enabled for preprocessing.")
            self.token_map = self.token_swapping_cfg.get("token_map", {})

        # ✅ FIX: Change the access path to the preprocessing_cfg object
        additional_special_tokens = self.preprocessing_cfg.get("additional_special_tokens", [])
        if additional_special_tokens:
            self.tokenizer.add_tokens(additional_special_tokens) 
        
        ic(f"DialoguePreprocessor initialized with {len(self.tokenizer)} tokens")


    def _swap_tokens(self, text: str) -> str:
        """Applies the token map to a given text."""
        if not self.token_swapping_cfg.get("enable"):
            return text
        
        for original, replacement in self.token_map.items():
            text = text.replace(original, replacement)
        return text
       
    def _setup_special_tokens(self) -> None:
        """Adds special tokens only if token swapping is disabled."""
        # ✅ If we are swapping tokens for names, we don't need to add them to the vocab
        if not self.token_swapping_cfg.get("enable"):
            additional_tokens = self.preprocessing_cfg.get("special_tokens", [])
            if additional_tokens:
                self.tokenizer.add_tokens([str(t) for t in additional_tokens])

    def _swap_special_tokens(self, text: str) -> str:
        """Replaces #Person# tokens with mapped names from the config."""
        if self.token_swapping_cfg.get("enable"):
            token_map = self.token_swapping_cfg.get("token_map", {})
            for original, replacement in token_map.items():
                text = text.replace(original, replacement)
        return text
    

    def _clean_text(self, text: str) -> str:
        """Performs basic cleaning of raw text strings."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.replace('\\n', ' ')
        text = re.sub(r'<[^>]+>', ' ', text)
        if self.preprocessing_cfg.get("normalize_whitespace", True):
            text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess_dialogue(self, dialogue: str) -> str:
        """Cleans and prepares a single dialogue string."""
        # ✅ Add the swap step
        dialogue = self._swap_special_tokens(dialogue)
        return self._clean_text(dialogue)

    def preprocess_summary(self, summary: str) -> str:
        """Cleans and prepares a single summary string."""
        # ✅ Add the swap step
        summary = self._swap_special_tokens(summary)
        return self._clean_text(summary)

    def prepare_inputs(
        self,
        dialogue: str,
        summary: Optional[str] = None,
        is_inference: bool = False
    ) -> BatchEncoding:
        """Preprocesses and tokenizes a single dialogue-summary pair."""
        #  The prepare_inputs function in preprocessing.py expects a configuration object where it can directly access max_input_length and max_target_length at the top level of its configuration attribute, like self.preprocessing_cfg.max_input_length
        dialogue = self.preprocess_dialogue(dialogue)
        
        # ✅ FIX: Call the local _swap_tokens method before tokenization
        if self.token_swapping_cfg.get("enable"):
            dialogue = self._swap_tokens(dialogue)
        
        model_inputs = self.tokenizer(
            dialogue,
            max_length=self.preprocessing_cfg.max_input_length, # ERROR
            truncation=True,
            padding=False,  # Padding is handled by the collate function
            return_tensors="pt" if not is_inference else None
        )

        if summary is not None:
            summary = self.preprocess_summary(summary)
            labels = self.tokenizer(
                summary,
                max_length=self.preprocessing_cfg.max_target_length,
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
            max_length=self.cfg.preprocessing.max_input_length,
            truncation=True,
            padding=True, # Batch tokenization can handle padding directly
            return_tensors="pt"
        )

        if summaries is not None and not is_inference:
            processed_summaries = [self.preprocess_summary(s) for s in summaries]
            labels = self.tokenizer(
                text=processed_summaries,
                max_length=self.preprocessing_cfg.max_target_length,
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
    model_name = cfg.model.tokenizer.name_or_path
    ic(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 📝 NEW: Log the state of token swapping at the start of the pipeline
    swap_enabled = cfg.preprocessing.get("token_swapping", {}).get("enable", False)
    if swap_enabled:
        ic("✅ Preprocessing: Token swapping is ENABLED. Replacing #PersonN# with names.")
    else:
        ic("❌ Preprocessing: Token swapping is DISABLED.")

    return DialoguePreprocessor(cfg, tokenizer)