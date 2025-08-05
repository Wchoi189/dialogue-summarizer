# FILE: src/data/dataset.py
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
        cfg: DictConfig, # expects dataset config block
        split: str = "train",
        is_inference: bool = False
    ):
        """
        Initialize dataset.
        """
        self.data = data.copy()
        self.preprocessor = preprocessor # preprocessor.prepare_inputs passes the configuration that the preprocessor was initialized with
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
        """
        row = self.data.iloc[idx]
        dialogue = str(row[self.input_col])
        sample = self.data.iloc[idx]
        # ✅ CORRECT: This finds "#Person1#" and replaces it with "화자1"
        # dialogue = dialogue.replace("#Person1#", "화자1").replace("#Person2#", "화자2").replace("#Person3#", "화자3")

        
        # ✅ FIX: Apply token swapping to the dialogue text before tokenization
        dialogue = self.preprocessor._swap_tokens(str(sample[self.cfg.columns.input]))
        
        # Get target if it exists
        summary = None
        if self.cfg.columns.target in self.data.columns:
            summary = self.preprocessor._swap_tokens(str(sample[self.cfg.columns.target]))     

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
        cfg: DictConfig  # This cfg is the DATASET config block
    ):
        """
        Initialize training dataset.
        """
        super().__init__(
            data=data,
            preprocessor=preprocessor,
            cfg=cfg,
            # cfg=cfg.preprocessing,
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
        """
        super().__init__(
            data=data,
            preprocessor=preprocessor,
            cfg=cfg,
            # cfg=cfg.preprocessing,
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
        """
        super().__init__(
            data=data,
            preprocessor=preprocessor,
            cfg=cfg,
            # cfg=cfg.preprocessing,
            split="test",
            is_inference=True
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample for inference.
        This now automatically uses the corrected logic from the parent DialogueDataset.
        """
        # ✅ FIX: Simply call the parent method, as the logic is now there.
        return super().__getitem__(idx)

class CollateFunction:
    """Fixed collate function for dialogue data."""
    
    def __init__(self, tokenizer, is_inference: bool = False):
        """
        Initialize collate function.
        """
        self.tokenizer = tokenizer
        self.is_inference = is_inference
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
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
                
                # Determine padding value
                if key == "labels":
                    padding_value = -100
                elif key == "input_ids":
                    padding_value = self.tokenizer.pad_token_id or 0
                else:
                    padding_value = 0
                
                # Pad sequences
                padded = torch.nn.utils.rnn.pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=padding_value
                )
                
                batched[key] = padded
        
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
    """
    datasets = {}

    # The cfg object here is the full config, so we need to pass
    # the specific 'dataset' block to our dataset classes.
    # The DialogueDataset class expects to be initialized with the dataset block of your config, which contains the columns mappings.
    dataset_cfg = cfg.dataset

    if train_data is not None:
        ic(f"Creating training dataset: {len(train_data)} samples")
        datasets["train"] = TrainingDataset(train_data, preprocessor, dataset_cfg)
    
    if val_data is not None:
        ic(f"Creating validation dataset: {len(val_data)} samples")
        datasets["val"] = ValidationDataset(val_data, preprocessor, dataset_cfg)
    
    if test_data is not None:
        ic(f"Creating inference dataset: {len(test_data)} samples")
        datasets["test"] = InferenceDataset(test_data, preprocessor, dataset_cfg)
    
    ic(f"Created {len(datasets)} datasets")
    return datasets


def create_collate_fn(tokenizer, is_inference: bool = False) -> CollateFunction:
    """
    Create collate function for DataLoader.
    """
    return CollateFunction(tokenizer, is_inference)