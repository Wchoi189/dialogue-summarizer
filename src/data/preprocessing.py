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
    
    def __init__(self, dataset_cfg: DictConfig, tokenizer: PreTrainedTokenizer):
        """
        Initialize preprocessor with configuration and tokenizer.
        
        Args:
            dataset_cfg: Dataset configuration (not full config)
            tokenizer: Pre-trained tokenizer
        """
        self.dataset_cfg = dataset_cfg
        self.tokenizer = tokenizer
        
        self.preprocessing_cfg = dataset_cfg.preprocessing
        
        # Add special tokens to tokenizer
        self._setup_special_tokens()
        
        ic(f"DialoguePreprocessor initialized with {len(self.tokenizer)} tokens")
    
    def _setup_special_tokens(self) -> None:
        """
        Safely adds additional tokens (like #Person1#) to the tokenizer's
        regular vocabulary.
        """
        additional_tokens = self.preprocessing_cfg.get("special_tokens", [])

        if not additional_tokens:
            return

        # ✅ Use add_tokens instead of add_special_tokens
        num_added = self.tokenizer.add_tokens([str(t) for t in additional_tokens])

        if num_added > 0:
            ic(f"Added {num_added} new tokens to the tokenizer vocabulary.")
        else:
            ic("No new tokens were added to the tokenizer vocabulary.")
            
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
        # Handle literal "\\n" characters
        text = text.replace('\\n', '\n')
        
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
            """
            if not dialogue:
                return False
            
            # ✅ FIX: Update the pattern to look for "#Person#" (speaker)
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
    ) -> BatchEncoding:
        """
        FINAL FIXED: Prepare model inputs from dialogue and summary.
        """
        # Preprocess inputs
        dialogue = self.preprocess_dialogue(dialogue)
        if summary is not None:
            summary = self.preprocess_summary(summary)

        # Tokenize dialogue (input)
        model_inputs = self.tokenizer(
            dialogue,
            max_length=self.preprocessing_cfg.max_input_length,
            truncation=True,
            padding=self.preprocessing_cfg.padding,
            return_tensors="pt" if not is_inference else None
        )

        # For training/validation, tokenize the summary (target)
        if summary is not None:
            # ✅ CRITICAL FIX: Do NOT pad labels here - let collate function handle it with -100
            labels = self.tokenizer(
                summary,
                max_length=self.preprocessing_cfg.max_target_length,
                truncation=True,
                padding=False,  # ← CRITICAL: No padding here!
                return_tensors="pt" if not is_inference else None,
                add_special_tokens=True
            )
            model_inputs['labels'] = labels['input_ids']

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
    ) -> BatchEncoding:
        """
        Preprocess a batch of dialogues and summaries using vectorization.
        """
        ic(f"Preprocessing batch of {len(dialogues)} samples")

        
        model_inputs = self.tokenizer(
            # processed_dialogues,
            max_length=self.preprocessing_cfg.max_input_length,
            truncation=True,
            padding=True, # Padding is handled by the tokenizer for the whole batch
            return_tensors="pt"
        )

        if summaries is not None and not is_inference:
            processed_summaries = [self.preprocess_summary(s) for s in summaries]
            labels = self.tokenizer(
                processed_summaries,
                max_length=self.preprocessing_cfg.max_target_length,
                truncation=True,
                padding=True, # Padding is handled by the tokenizer for the whole batch
                return_tensors="pt"
            )
            model_inputs['labels'] = labels['input_ids']

        ic(f"Batch preprocessing complete for {len(dialogues)} samples")
        return model_inputs


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
    # ✅ CHANGE: Use dataset config instead of preprocessing config
    return DialoguePreprocessor(cfg.dataset, tokenizer)  # Changed from cfg to cfg.dataset


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

# Enhanced preprocessing to force summarization behavior
# Add this to your preprocessing.py

def prepare_summarization_inputs(self, dialogue: str, summary: Optional[str] = None, is_inference: bool = False):
    """
    Enhanced input preparation that forces summarization behavior.
    Based on EDA insights about Korean dialogue structure.
    """
    # Clean dialogue
    dialogue = self.preprocess_dialogue(dialogue)
    
    # 🔥 CRITICAL: Add explicit summarization prompt in Korean
    dialogue_with_prompt = f"다음 대화를 요약하세요:\n{dialogue}\n\n요약:"
    
    # Tokenize with summarization context
    model_inputs = self.tokenizer(
        dialogue_with_prompt,
        max_length=self.preprocessing_cfg.max_input_length,
        truncation=True,
        padding=self.preprocessing_cfg.padding,
        return_tensors="pt" if not is_inference else None
    )
    
    if summary is not None:
        # 🔥 CRITICAL: Ensure summary starts with clear marker
        summary_clean = self.preprocess_summary(summary)
        
        # Add summarization markers
        summary_with_markers = f"{summary_clean}"
        
        labels = self.tokenizer(
            summary_with_markers,
            max_length=self.preprocessing_cfg.max_target_length,
            truncation=True,
            padding=False,  # Let collate function handle padding
            return_tensors="pt" if not is_inference else None,
            add_special_tokens=True
        )
        model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# Enhanced Korean summary validation
def validate_summary_quality(self, generated_text: str, input_dialogue: str) -> str:
    """
    Validate and fix generated summary based on Korean dialogue patterns.
    """
    # Remove the input prompt if it was copied
    if "다음 대화를 요약하세요:" in generated_text:
        generated_text = generated_text.split("요약:")[-1].strip()
    
    # Check if it's copying the input (major issue)
    dialogue_words = set(input_dialogue.lower().split())
    summary_words = set(generated_text.lower().split())
    
    # If >70% overlap, it's probably copying not summarizing
    if len(summary_words & dialogue_words) / len(summary_words) > 0.7:
        # Force a generic summary
        return "대화 내용을 요약했습니다."
    
    # Check length constraints based on EDA
    words = generated_text.split()
    
    # Based on EDA: summaries should be 8-30 words
    if len(words) > 30:
        # Truncate to first sentence or first 20 words
        sentences = generated_text.split('.')
        if len(sentences) > 1 and len(sentences[0].split()) <= 25:
            generated_text = sentences[0] + '.'
        else:
            generated_text = ' '.join(words[:20]) + '.'
    
    elif len(words) < 5:
        # Too short, probably failed
        return "대화 내용을 요약했습니다."
    
    # Ensure proper Korean ending
    if not generated_text.endswith(('다.', '요.', '습니다.', '니다.')):
        generated_text = generated_text.rstrip('.') + '다.'
    
    return generated_text

# Updated postprocessing method
def _apply_comprehensive_post_processing(self, text: str, post_cfg: dict) -> str:
    """Apply comprehensive post-processing with summarization focus."""
    
    # 1. Remove unwanted tokens
    remove_tokens = post_cfg.get("remove_tokens", [])
    for token in remove_tokens:
        text = text.replace(token, "")
    
    # 2. Basic cleaning
    text = text.strip()
    if not text:
        return "요약을 생성할 수 없습니다."
    
    # 3. 🔥 CRITICAL: Validate it's a summary not continuation
    text = self.validate_summary_quality(text, "")  # You'd need to pass original dialogue
    
    # 4. Korean-specific cleaning
    korean_cfg = post_cfg.get("korean_specific", {})
    if korean_cfg.get("normalize_punctuation", True):
        # Normalize Korean punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        text = re.sub(r'([,.!?])\s*', r'\1 ', text).strip()
    
    # 5. Final length check based on EDA insights
    words = text.split()
    if len(words) > 30:  # Based on EDA: avg 16 words, max should be ~30
        text = ' '.join(words[:25]) + '.'
    
    return text