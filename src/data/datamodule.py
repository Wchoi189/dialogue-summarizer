"""
PyTorch Lightning DataModule for dialogue summarization.
Handles data loading, preprocessing, and batch creation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import pytorch_lightning as pl
from icecream import ic
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.file_utils import FileManager
from .dataset import create_collate_fn, create_datasets
from .preprocessing import DataValidator, DialoguePreprocessor, create_preprocessor

logger = logging.getLogger(__name__)


class DialogueDataModule(pl.LightningDataModule):
    """Lightning DataModule for dialogue summarization."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize DataModule.
        
        Args:
            cfg: Complete configuration including dataset settings
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg.dataset
        
        # Initialize components
        self.file_manager = FileManager()
        self.preprocessor = None
        self.datasets = {}
        self.data_validator = DataValidator(self.dataset_cfg)
        
        # Data storage
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Statistics
        self.data_stats = {}
        
        ic(f"DialogueDataModule initialized")
    
    def prepare_data(self) -> None:
        """
        Download/prepare data (called only on main process).
        In our case, just validate that data files exist.
        """
        ic("Preparing data...")
        
        # Validate data files exist
        self.file_manager.validate_data_files(self.cfg)
        
        ic("Data preparation complete")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for training/validation/testing.
        
        Args:
            stage: Stage name (fit, validate, test, predict)
        """
        ic(f"Setting up data for stage: {stage}")
        
        # Initialize preprocessor
        if self.preprocessor is None:
            self.preprocessor = create_preprocessor(self.cfg)
        
        # Load data based on stage
        if stage == "fit" or stage is None:
            self._setup_train_val_data()
        
        if stage == "test" or stage == "predict" or stage is None:
            self._setup_test_data()
        
        # Create datasets
        self.datasets = create_datasets(
            cfg=self.dataset_cfg,
            preprocessor=self.preprocessor,
            train_data=self.train_data,
            val_data=self.val_data,
            test_data=self.test_data
        )
        
        ic(f"Data setup complete for stage: {stage}")
    
    def _setup_train_val_data(self) -> None:
        """Setup training and validation data."""
        data_path = Path(self.dataset_cfg.data_path)
        
        # Load training data
        train_file = data_path / self.dataset_cfg.files.train
        ic(f"Loading training data: {train_file}")
        self.train_data = self.file_manager.load_csv(train_file)
        
        # Load validation data
        val_file = data_path / self.dataset_cfg.files.dev
        ic(f"Loading validation data: {val_file}")
        self.val_data = self.file_manager.load_csv(val_file)
        
        # Validate data
        self.data_validator.validate_dataframe(self.train_data, "train")
        self.data_validator.validate_dataframe(self.val_data, "dev")
        
        # Compute statistics
        self.data_stats["train"] = self.data_validator.compute_statistics(
            self.train_data, "train"
        )
        self.data_stats["val"] = self.data_validator.compute_statistics(
            self.val_data, "val"
        )
        
        ic(f"Train/Val data loaded: {len(self.train_data)}/{len(self.val_data)} samples")
    
    def _setup_test_data(self) -> None:
        """Setup test data."""
        data_path = Path(self.dataset_cfg.data_path)
        
        # Load test data
        test_file = data_path / self.dataset_cfg.files.test
        ic(f"Loading test data: {test_file}")
        self.test_data = self.file_manager.load_csv(test_file)
        
        # Validate test data
        self.data_validator.validate_dataframe(self.test_data, "test")
        
        # Compute statistics
        self.data_stats["test"] = self.data_validator.compute_statistics(
            self.test_data, "test"
        )
        
        ic(f"Test data loaded: {len(self.test_data)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if "train" not in self.datasets:
            raise RuntimeError("Training dataset not available. Call setup('fit') first.")
        
        collate_fn = create_collate_fn(
            tokenizer=self.preprocessor.tokenizer,
            is_inference=False
        )
        
        dataloader = DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.dataset_cfg.batch_size,
            shuffle=self.dataset_cfg.shuffle_train,
            num_workers=self.dataset_cfg.get("num_workers", 4),
            pin_memory=self.dataset_cfg.get("pin_memory", True),
            drop_last=self.dataset_cfg.drop_last,
            collate_fn=collate_fn,
            persistent_workers=True if self.dataset_cfg.get("num_workers", 4) > 0 else False
        )
        
        ic(f"Train DataLoader created: {len(dataloader)} batches")
        return dataloader
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if "val" not in self.datasets:
            raise RuntimeError("Validation dataset not available. Call setup('fit') first.")
        
        collate_fn = create_collate_fn(
            tokenizer=self.preprocessor.tokenizer,
            is_inference=False
        )
        
        dataloader = DataLoader(
            dataset=self.datasets["val"],
            batch_size=self.dataset_cfg.eval_batch_size,
            shuffle=self.dataset_cfg.shuffle_val,
            num_workers=self.dataset_cfg.get("num_workers", 4),
            pin_memory=self.dataset_cfg.get("pin_memory", True),
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=True if self.dataset_cfg.get("num_workers", 4) > 0 else False
        )
        
        ic(f"Val DataLoader created: {len(dataloader)} batches")
        return dataloader
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if "test" not in self.datasets:
            raise RuntimeError("Test dataset not available. Call setup('test') first.")
        
        collate_fn = create_collate_fn(
            tokenizer=self.preprocessor.tokenizer,
            is_inference=True
        )
        
        dataloader = DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.dataset_cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.dataset_cfg.get("num_workers", 4),
            pin_memory=self.dataset_cfg.get("pin_memory", True),
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=True if self.dataset_cfg.get("num_workers", 4) > 0 else False
        )
        
        ic(f"Test DataLoader created: {len(dataloader)} batches")
        return dataloader
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader (same as test)."""
        return self.test_dataloader()
    
    def get_data_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics for each split
        """
        return self.data_stats.copy()
    
    def get_sample_data(self, split: str = "train", num_samples: int = 3) -> pd.DataFrame:
        """
        Get sample data for inspection.
        
        Args:
            split: Data split (train, val, test)
            num_samples: Number of samples to return
            
        Returns:
            Sample DataFrame
        """
        data_map = {
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data
        }
        
        if split not in data_map or data_map[split] is None:
            raise ValueError(f"Data for split '{split}' not available")
        
        data = data_map[split]
        sample_data = data.head(num_samples).copy()
        
        ic(f"Retrieved {len(sample_data)} samples from {split}")
        return sample_data
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        if self.preprocessor is None:
            self.preprocessor = create_preprocessor(self.cfg)
        
        return len(self.preprocessor.tokenizer)
    
    def get_tokenizer(self):
        """Get the tokenizer instance."""
        if self.preprocessor is None:
            self.preprocessor = create_preprocessor(self.cfg)
        
        return self.preprocessor.tokenizer
    
    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens from tokenizer."""
        tokenizer = self.get_tokenizer()
        
        special_tokens = {
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token,
            "unk_token": tokenizer.unk_token,
        }
        
        return special_tokens
    
    def save_processed_data(self, output_dir: Union[str, Path]) -> None:
        """
        Save processed data for future use.
        
        Args:
            output_dir: Directory to save processed data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        if self.train_data is not None:
            train_path = output_dir / "processed_train.csv"
            self.file_manager.save_csv(self.train_data, train_path)
            ic(f"Saved processed training data: {train_path}")
        
        if self.val_data is not None:
            val_path = output_dir / "processed_val.csv"
            self.file_manager.save_csv(self.val_data, val_path)
            ic(f"Saved processed validation data: {val_path}")
        
        if self.test_data is not None:
            test_path = output_dir / "processed_test.csv"
            self.file_manager.save_csv(self.test_data, test_path)
            ic(f"Saved processed test data: {test_path}")
        
        # Save statistics
        if self.data_stats:
            stats_path = output_dir / "data_statistics.json"
            self.file_manager.save_json(self.data_stats, stats_path)
            ic(f"Saved data statistics: {stats_path}")


def create_datamodule(cfg: DictConfig) -> DialogueDataModule:
    """
    Create DataModule instance.
    
    Args:
        cfg: Complete configuration
        
    Returns:
        Configured DataModule
    """
    return DialogueDataModule(cfg)


# Utility functions for data analysis
def analyze_dataset(datamodule: DialogueDataModule, split: str = "train") -> Dict[str, any]:
    """
    Analyze dataset characteristics.
    
    Args:
        datamodule: DataModule instance
        split: Data split to analyze
        
    Returns:
        Analysis results
    """
    ic(f"Analyzing {split} dataset...")
    
    # Get sample data
    data = datamodule.get_sample_data(split, num_samples=100)
    
    analysis = {
        "num_samples": len(data),
        "columns": list(data.columns),
        "dialogue_stats": {},
        "summary_stats": {}
    }
    
    # Analyze dialogues
    input_col = datamodule.dataset_cfg.columns.input
    if input_col in data.columns:
        dialogues = data[input_col].dropna()
        
        analysis["dialogue_stats"] = {
            "avg_length_chars": dialogues.str.len().mean(),
            "avg_length_words": dialogues.str.split().str.len().mean(),
            "min_length_chars": dialogues.str.len().min(),
            "max_length_chars": dialogues.str.len().max(),
        }
    
    # Analyze summaries (if available)
    target_col = datamodule.dataset_cfg.columns.target
    if target_col in data.columns:
        summaries = data[target_col].dropna()
        
        analysis["summary_stats"] = {
            "avg_length_chars": summaries.str.len().mean(),
            "avg_length_words": summaries.str.split().str.len().mean(),
            "min_length_chars": summaries.str.len().min(),
            "max_length_chars": summaries.str.len().max(),
        }
    
    ic(f"Dataset analysis complete for {split}")
    return analysis


def print_data_sample(datamodule: DialogueDataModule, split: str = "train", idx: int = 0) -> None:
    """
    Print a sample from the dataset for inspection.
    
    Args:
        datamodule: DataModule instance
        split: Data split
        idx: Sample index
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    
    console = Console()
    
    # Get sample data
    data = datamodule.get_sample_data(split, num_samples=idx + 1)
    if len(data) <= idx:
        console.print(f"[red]Sample index {idx} not available in {split}[/red]")
        return
    
    sample = data.iloc[idx]
    
    # Create display
    input_col = datamodule.dataset_cfg.columns.input
    target_col = datamodule.dataset_cfg.columns.target
    id_col = datamodule.dataset_cfg.columns.id
    
    # Sample ID
    console.print(Panel.fit(f"[bold blue]Sample ID:[/bold blue] {sample[id_col]}"))
    
    # Dialogue
    dialogue_text = Text(sample[input_col])
    console.print(Panel(dialogue_text, title="[green]Dialogue[/green]", expand=False))
    
    # Summary (if available)
    if target_col in sample and pd.notna(sample[target_col]):
        summary_text = Text(sample[target_col])
        console.print(Panel(summary_text, title="[yellow]Summary[/yellow]", expand=False))
    
    ic(f"Displayed sample {idx} from {split}")