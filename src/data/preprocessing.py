"""
Data preprocessing utilities for Korean dialogue summarization.
Handles text cleaning, tokenization, and special token management.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from icecream import ic
from omegaconf import DictConfig
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer
import transformers
import torch

logger = logging.getLogger(__name__)


class DialoguePreprocessor:
    """Enhanced preprocessor for Korean dialogue data."""
    
    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizer):
        """
        Initialize preprocessor with configuration and tokenizer.
        
        Args:
            cfg: Dataset configuration
            tokenizer: Pre-trained tokenizer
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.preprocessing_cfg = cfg.preprocessing
        
        # Add special tokens to tokenizer
        self._setup_special_tokens()
        
        ic(f"DialoguePreprocessor initialized with {len(self.tokenizer)} tokens")
    
    def _setup_special_tokens(self) -> None:
        """Setup special tokens in tokenizer."""
        special_tokens = self.preprocessing_cfg.special_tokens
        
        if special_tokens:
            # Ensure special tokens are strings
            special_tokens_list = [str(token) for token in special_tokens]
            special_tokens_dict = {"additional_special_tokens": special_tokens_list}
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            ic(f"Added {num_added} special tokens: {special_tokens_list}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean Korean dialogue text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        if self.preprocessing_cfg.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
        
        # Remove extra newlines but preserve dialogue structure
        if self.preprocessing_cfg.remove_extra_newlines:
            # Replace multiple newlines with single newline
            text = re.sub(r'\n\s*\n', '\n', text)
            # Clean up spaces around newlines
            text = re.sub(r' *\n *', '\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def validate_dialogue_format(self, dialogue: str) -> bool:
        """
        Validate that dialogue contains proper speaker labels.
        
        Args:
            dialogue: Dialogue text to validate
            
        Returns:
            True if format is valid
        """
        if not dialogue:
            return False
        
        # Check for speaker labels
        speaker_pattern = r'#Person\d+#'
        speakers = re.findall(speaker_pattern, dialogue)
        
        if not speakers:
            logger.warning(f"No speaker labels found in dialogue: {dialogue[:100]}...")
            return False
        
        # Should have at least 2 speakers for dialogue
        unique_speakers = set(speakers)
        if len(unique_speakers) < 2:
            logger.warning(f"Less than 2 speakers found: {unique_speakers}")
            return False
        
        return True
    
    def preprocess_dialogue(self, dialogue: str) -> str:
        """
        Preprocess dialogue text with Korean-specific handling.
        
        Args:
            dialogue: Raw dialogue text
            
        Returns:
            Preprocessed dialogue
        """
        # Clean text
        dialogue = self.clean_text(dialogue)
        
        # Validate format
        if not self.validate_dialogue_format(dialogue):
            logger.warning("Invalid dialogue format detected")
        
        return dialogue
    
    def preprocess_summary(self, summary: str) -> str:
        """
        Preprocess summary text.
        
        Args:
            summary: Raw summary text
            
        Returns:
            Preprocessed summary
        """
        return self.clean_text(summary)
    
    def tokenize_text(
        self,
        text: str,
        max_length: int,
        truncation: bool = True,
        padding: bool = False,
        return_tensors: Optional[str] = None
    ) -> BatchEncoding:
        """
        Tokenize text with specified parameters.
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            padding: Whether to pad sequences
            return_tensors: Format of return tensors ('pt', 'tf', None)
            
        Returns:
            Tokenized text dictionary
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            add_special_tokens=True,
            return_tensors=return_tensors,
            return_attention_mask=True,
            return_token_type_ids=False
        )
    
    from typing import Mapping, Any

    def prepare_inputs(
        self,
        dialogue: str,
        summary: Optional[str] = None,
        is_inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Prepare model inputs from dialogue and summary.
        """
        # Preprocess inputs
        dialogue = self.preprocess_dialogue(dialogue)
        if summary is not None:
            summary = self.preprocess_summary(summary)

        # ✅ REFACTORED TOKENIZATION
        # Let the tokenizer handle both dialogue and summary (as text_target)
        model_inputs = self.tokenizer(
            dialogue,
            text_target=summary,
            max_length=self.preprocessing_cfg.max_input_length,
            truncation=True, # Truncate dialogue if too long
            padding=self.preprocessing_cfg.padding,
            return_tensors="pt" if not is_inference else None
        )

        # For training/validation, truncate the labels separately
        if summary is not None:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    summary,
                    max_length=self.preprocessing_cfg.max_target_length,
                    truncation=True,
                    padding=self.preprocessing_cfg.padding,
                    return_tensors="pt" if not is_inference else None
                )
            model_inputs['labels'] = labels['input_ids']

        # No need to manually create decoder_input_ids, the model handles it.
        # The forward method of BartForConditionalGeneration creates them from labels.
        return model_inputs
    
    def decode_outputs(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Decode model outputs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        
        # Additional cleaning for Korean text
        text = self.clean_text(text)
        
        return text
    
    def batch_preprocess(
        self,
        dialogues: List[str],
        summaries: Optional[List[str]] = None,
        is_inference: bool = False
    ) -> Dict[str, List]:
        """
        Preprocess a batch of dialogues and summaries.
        
        Args:
            dialogues: List of dialogue texts
            summaries: List of summary texts (None for inference)
            is_inference: Whether this is for inference
            
        Returns:
            Dictionary with batched inputs
        """
        batch_size = len(dialogues)
        ic(f"Preprocessing batch of {batch_size} samples")
        
        # Initialize result containers
        result = {
            "input_ids": [],
            "attention_mask": []
        }
        
        if summaries is not None:
            result.update({
                "decoder_input_ids": [],
                "decoder_attention_mask": [],
                "labels": []
            })
        
        # Process each sample
        for i, dialogue in enumerate(dialogues):
            summary = summaries[i] if summaries else None
            
            sample_inputs = self.prepare_inputs(
                dialogue=dialogue,
                summary=summary,
                is_inference=is_inference
            )
            
            # Add to batch
            for key, value in sample_inputs.items():
                if key in result:
                    result[key].append(value)
        
        ic(f"Batch preprocessing complete: {len(result['input_ids'])} samples")
        return result


class DataValidator:
    """Validator for dialogue data quality."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize data validator.
        
        Args:
            cfg: Dataset configuration
        """
        self.cfg = cfg
    
    def validate_dataframe(self, df: pd.DataFrame, split: str = "unknown") -> bool:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            split: Data split name for logging
            
        Returns:
            True if validation passes
        """
        ic(f"Validating {split} DataFrame: {len(df)} rows")
        
        # Check required columns
        required_cols = [self.cfg.columns.id, self.cfg.columns.input]
        if split in ["train", "dev"]:
            required_cols.append(self.cfg.columns.target)
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in {split}: {missing_cols}")
            return False
        
        # Check for empty values
        for col in required_cols:
            empty_count = df[col].isna().sum()
            if empty_count > 0:
                logger.warning(f"{split} has {empty_count} empty values in {col}")
        
        # Validate dialogue format
        if self.cfg.columns.input in df.columns:
            invalid_dialogues = 0
            for idx, dialogue in enumerate(df[self.cfg.columns.input]):
                if not self._validate_dialogue_content(dialogue):
                    invalid_dialogues += 1
            
            if invalid_dialogues > 0:
                logger.warning(f"{split} has {invalid_dialogues} invalid dialogues")
        
        ic(f"{split} validation complete")
        return True
    
    def _validate_dialogue_content(self, dialogue: str) -> bool:
        """Validate individual dialogue content."""
        if pd.isna(dialogue) or not isinstance(dialogue, str):
            return False
        
        # Check for speaker labels
        speaker_pattern = r'#Person\d+#'
        speakers = re.findall(speaker_pattern, dialogue)
        
        return len(set(speakers)) >= 2
    
    def compute_statistics(self, df: pd.DataFrame, split: str = "unknown") -> Dict[str, float]:
        """
        Compute dataset statistics.
        
        Args:
            df: DataFrame to analyze
            split: Data split name
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "num_samples": len(df),
            "avg_dialogue_length": 0.0,
            "avg_summary_length": 0.0,
            "avg_dialogue_chars": 0.0,
            "avg_summary_chars": 0.0
        }
        
        # Dialogue statistics
        if self.cfg.columns.input in df.columns:
            dialogue_lengths = df[self.cfg.columns.input].str.split().str.len()
            dialogue_chars = df[self.cfg.columns.input].str.len()
            
            stats["avg_dialogue_length"] = dialogue_lengths.mean()
            stats["avg_dialogue_chars"] = dialogue_chars.mean()
        
        # Summary statistics (if available)
        if self.cfg.columns.target in df.columns:
            summary_lengths = df[self.cfg.columns.target].str.split().str.len()
            summary_chars = df[self.cfg.columns.target].str.len()
            
            stats["avg_summary_length"] = summary_lengths.mean()
            stats["avg_summary_chars"] = summary_chars.mean()
        
        ic(f"{split} statistics: {stats}")
        return stats


def create_preprocessor(cfg: DictConfig, model_name: Optional[str] = None) -> DialoguePreprocessor:
    """
    Create preprocessor with tokenizer.
    
    Args:
        cfg: Configuration
        model_name: Model name for tokenizer (uses config if None)
        
    Returns:
        Configured preprocessor
    """
    if model_name is None:
        model_name = cfg.model.tokenizer.name_or_path
    
    ic(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return DialoguePreprocessor(cfg, tokenizer)


def preprocess_submission_data(
    df: pd.DataFrame,
    cfg: DictConfig,
    preprocessor: DialoguePreprocessor
) -> pd.DataFrame:
    """
    Preprocess data for submission format.
    
    Args:
        df: DataFrame with predictions
        cfg: Configuration
        preprocessor: Text preprocessor
        
    Returns:
        Cleaned DataFrame ready for submission
    """
    ic(f"Preprocessing submission data: {len(df)} samples")
    
    # Clean summary texts
    if "summary" in df.columns:
        df["summary"] = df["summary"].apply(preprocessor.preprocess_summary)
    
    # Ensure correct column order for submission
    submission_cols = cfg.submission.columns
    df = df[submission_cols].copy()
    
    # Add index if required
    if cfg.submission.include_index:
        df.reset_index(inplace=True)
        if cfg.submission.index_name:
            df.rename(columns={"index": cfg.submission.index_name}, inplace=True)
    
    ic(f"Submission data ready: {list(df.columns)}")
    return df