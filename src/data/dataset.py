"""
PyTorch Dataset classes for dialogue summarization.
Handles Korean text data with proper tokenization and formatting.
"""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from icecream import ic
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .preprocessing import DialoguePreprocessor

logger = logging.getLogger(__name__)


class DialogueDataset(Dataset):
    """Base dataset class for dialogue summarization."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: DialoguePreprocessor,
        cfg: DictConfig,
        split: str = "train",
        is_inference: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data: DataFrame with dialogue data
            preprocessor: Text preprocessor
            cfg: Dataset configuration
            split: Data split name (train, dev, test)
            is_inference: Whether this is for inference
        """
        self.data = data.copy()
        self.preprocessor = preprocessor
        self.cfg = cfg
        self.split = split
        self.is_inference = is_inference
        
        # Column mappings
        self.id_col = cfg.columns.id
        self.input_col = cfg.columns.input
        self.target_col = cfg.columns.target
        
        # Validate data
        self._validate_data()
        
        ic(f"DialogueDataset initialized: {len(self)} samples, split={split}")
    
    def _validate_data(self) -> None:
        """Validate dataset structure."""
        required_cols = [self.id_col, self.input_col]
        
        if not self.is_inference and self.target_col in self.data.columns:
            required_cols.append(self.target_col)
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing dialogues
        initial_len = len(self.data)
        self.data = self.data.dropna(subset=[self.input_col])
        
        if len(self.data) < initial_len:
            dropped = initial_len - len(self.data)
            logger.warning(f"Dropped {dropped} rows with missing dialogues")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with model inputs
        """
        row = self.data.iloc[idx]
        
        # Get dialogue text
        dialogue = str(row[self.input_col])
        
        # Get summary if available
        summary = None
        if not self.is_inference and self.target_col in row:
            summary = str(row[self.target_col])
        
        # Preprocess and tokenize
        inputs = self.preprocessor.prepare_inputs(
            dialogue=dialogue,
            summary=summary,
            is_inference=self.is_inference
        )
        
        # Convert to tensors and squeeze batch dimension
        tensor_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                tensor_inputs[key] = value.squeeze(0)  # Remove batch dim
            else:
                tensor_inputs[key] = torch.tensor(value)
        
        # Add sample ID for tracking
        tensor_inputs["sample_id"] = row[self.id_col]
        
        return tensor_inputs


class TrainingDataset(DialogueDataset):
    """Dataset class specifically for training."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: DialoguePreprocessor,
        cfg: DictConfig
    ):
        """
        Initialize training dataset.
        
        Args:
            data: Training DataFrame
            preprocessor: Text preprocessor
            cfg: Dataset configuration
        """
        super().__init__(
            data=data,
            preprocessor=preprocessor,
            cfg=cfg,
            split="train",
            is_inference=False
        )
        
        # Ensure we have target column
        if self.target_col not in self.data.columns:
            raise ValueError(f"Training data must have {self.target_col} column")


class ValidationDataset(DialogueDataset):
    """Dataset class specifically for validation."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: DialoguePreprocessor,
        cfg: DictConfig
    ):
        """
        Initialize validation dataset.
        
        Args:
            data: Validation DataFrame
            preprocessor: Text preprocessor
            cfg: Dataset configuration
        """
        super().__init__(
            data=data,
            preprocessor=preprocessor,
            cfg=cfg,
            split="dev",
            is_inference=False
        )
        
        # Ensure we have target column
        if self.target_col not in self.data.columns:
            raise ValueError(f"Validation data must have {self.target_col} column")


class InferenceDataset(DialogueDataset):
    """Dataset class specifically for inference."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: DialoguePreprocessor,
        cfg: DictConfig
    ):
        """
        Initialize inference dataset.
        
        Args:
            data: Test DataFrame
            preprocessor: Text preprocessor
            cfg: Dataset configuration
        """
        super().__init__(
            data=data,
            preprocessor=preprocessor,
            cfg=cfg,
            split="test",
            is_inference=True
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample for inference.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with model inputs (no labels)
        """
        row = self.data.iloc[idx]
        dialogue = str(row[self.input_col])
        
        # Preprocess and tokenize (no summary for inference)
        inputs = self.preprocessor.prepare_inputs(
            dialogue=dialogue,
            summary=None,
            is_inference=True
        )
        
        # Convert to tensors
        tensor_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                tensor_inputs[key] = value.squeeze(0)
            else:
                tensor_inputs[key] = torch.tensor(value)
        
        # Add sample ID
        tensor_inputs["sample_id"] = row[self.id_col]
        
        return tensor_inputs

class CollateFunction:
    """Fixed collate function for dialogue data."""
    
    def __init__(self, tokenizer, is_inference: bool = False):
        """
        Initialize collate function.
        
        Args:
            tokenizer: Tokenizer for padding
            is_inference: Whether this is for inference
        """
        self.tokenizer = tokenizer
        self.is_inference = is_inference
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched tensors
        """
        # Separate keys
        tensor_keys = ["input_ids", "attention_mask"]
        if not self.is_inference:
            tensor_keys.append("labels")
        
        # Initialize batch dict
        batched = {}
        
        # Collect sample IDs
        sample_ids = [sample["sample_id"] for sample in batch]
        batched["sample_ids"] = sample_ids
        
        # Pad and batch tensor data
        for key in tensor_keys:
            if key in batch[0]:
                sequences = [sample[key] for sample in batch]
                
                # Ensure all sequences are 1D tensors
                sequences = [seq.squeeze() if seq.dim() > 1 else seq for seq in sequences]
                
                # Determine padding value - THIS IS THE CRITICAL FIX
                if key == "labels":
                    padding_value = -100  # Use -100 directly for labels
                elif key == "input_ids":
                    padding_value = self.tokenizer.pad_token_id or 0
                else:
                    padding_value = 0  # For attention masks
                
                # Pad sequences
                padded = torch.nn.utils.rnn.pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=padding_value
                )
                
                batched[key] = padded
        
        # ❌ CRITICAL: REMOVE ALL POST-PROCESSING OF LABELS
        # DO NOT ADD ANYTHING LIKE:
        # if "labels" in batched:
        #     labels = batched["labels"]
        #     labels[labels == self.tokenizer.pad_token_id] = -100
        #     batched["labels"] = labels
        
        return batched

def create_datasets(
    cfg: DictConfig,
    preprocessor: DialoguePreprocessor,
    train_data: Optional[pd.DataFrame] = None,
    val_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None
) -> Dict[str, Optional[DialogueDataset]]:
    """
    Create datasets for training, validation, and testing.
    
    Args:
        cfg: Dataset configuration
        preprocessor: Text preprocessor
        train_data: Training DataFrame (optional)
        val_data: Validation DataFrame (optional)
        test_data: Test DataFrame (optional)
        
    Returns:
        Dictionary with dataset instances
    """
    datasets = {}
    
    if train_data is not None:
        ic(f"Creating training dataset: {len(train_data)} samples")
        datasets["train"] = TrainingDataset(train_data, preprocessor, cfg)
    
    if val_data is not None:
        ic(f"Creating validation dataset: {len(val_data)} samples")
        datasets["val"] = ValidationDataset(val_data, preprocessor, cfg)
    
    if test_data is not None:
        ic(f"Creating inference dataset: {len(test_data)} samples")
        datasets["test"] = InferenceDataset(test_data, preprocessor, cfg)
    
    ic(f"Created {len(datasets)} datasets")
    return datasets


def create_collate_fn(tokenizer, is_inference: bool = False) -> CollateFunction:
    """
    Create collate function for DataLoader.
    
    Args:
        tokenizer: Tokenizer for padding
        is_inference: Whether for inference
        
    Returns:
        Collate function instance
    """
    return CollateFunction(tokenizer, is_inference)