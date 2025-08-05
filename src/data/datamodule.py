# FILE: src/data/datamodule.py
"""
PyTorch Lightning DataModule for dialogue summarization.
Handles data loading, preprocessing, and batch creation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
from icecream import ic
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils.config_utils import ConfigManager
from utils.file_utils import FileManager
from .dataset import create_collate_fn, create_datasets
from .preprocessing import DialoguePreprocessor, create_preprocessor

logger = logging.getLogger(__name__)


class DialogueDataModule(pl.LightningDataModule):
    """Lightning DataModule for dialogue summarization."""
    
    def __init__(self, cfg: DictConfig):
        """Initializes the DataModule with the main configuration."""
        super().__init__()
        self.cfg = cfg
        self.file_manager = FileManager()
        
        self.preprocessor: Optional[DialoguePreprocessor] = None
        self.datasets = {}
        
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        ic("DialogueDataModule initialized")

    def _load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Loads the datasets (train, dev, test) from the configured paths.
        
        Returns:
            A tuple containing the train, dev, and optionally, test DataFrames.
        """
        cfg = self.cfg.dataset
        data_path = Path(cfg.data_path)

        train_data = self.file_manager.load_csv(data_path / cfg.files.train)
        dev_data = self.file_manager.load_csv(data_path / cfg.files.dev)

        test_data = None
        if "test" in cfg.files:
            test_file_path = data_path / cfg.files.test
            if test_file_path.exists():
                test_data = self.file_manager.load_csv(test_file_path)

        return train_data, dev_data, test_data    
    def prepare_data(self) -> None:
        """Downloads data and tokenizes if necessary."""
        ic("Preparing data...")
        self.file_manager = FileManager()
        

        self.train_data, self.val_data, self.test_data = self._load_datasets()
        ic("Data preparation complete")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Loads data, creates the preprocessor, and builds datasets."""
        ic(f"Setting up data for stage: {stage}")
        
        if self.preprocessor is None:
            self.preprocessor = create_preprocessor(self.cfg)
        
        data_path = Path(self.cfg.dataset.data_path)
        
        if stage in ("fit", None):
            self.train_data = self.file_manager.load_csv(data_path / self.cfg.dataset.files.train)
            self.val_data = self.file_manager.load_csv(data_path / self.cfg.dataset.files.dev)
            ic(f"Train/Val data loaded: {len(self.train_data)}/{len(self.val_data)} samples")
        
        if stage in ("test", "predict", None):
            self.test_data = self.file_manager.load_csv(data_path / self.cfg.dataset.files.test)
            ic(f"Test data loaded: {len(self.test_data)} samples")
        
        self.datasets = create_datasets(
            cfg=self.cfg,
            preprocessor=self.preprocessor,
            train_data=self.train_data,
            val_data=self.val_data,
            test_data=self.test_data
        )
        ic(f"Data setup complete for stage: {stage}")
    
    def _create_dataloader(self, split: str) -> DataLoader:
        """Helper function to create a DataLoader for a given data split."""
        dataset = self.datasets.get(split)
        if dataset is None:
            raise RuntimeError(f"{split} dataset not available or is None. Call setup() first.")
        
        is_inference = (split == "test")
        dataset_cfg = self.cfg.dataset

        # Add a check to ensure the preprocessor and tokenizer are initialized
        if self.preprocessor is None or self.preprocessor.tokenizer is None:
            raise ValueError("Preprocessor or its tokenizer is not initialized.")        
        collate_fn = create_collate_fn(
            tokenizer=self.preprocessor.tokenizer,
            is_inference=is_inference
        )
        
        batch_size = dataset_cfg.eval_batch_size if is_inference else dataset_cfg.batch_size
        shuffle = dataset_cfg.shuffle_train if split == "train" else False
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=dataset_cfg.get("num_workers", 4),
            pin_memory=dataset_cfg.get("pin_memory", True),
            collate_fn=collate_fn
        )
        
        ic(f"{split.capitalize()} DataLoader created: {len(dataloader)} batches")
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Creates the training DataLoader."""
        return self._create_dataloader("train")
    
    def val_dataloader(self) -> DataLoader:
        """Creates the validation DataLoader."""
        # The split name for validation data is 'val' in the create_datasets function.
        return self._create_dataloader("val")
    
    def test_dataloader(self) -> DataLoader:
        """Creates the test DataLoader."""
        return self._create_dataloader("test")
    
    def predict_dataloader(self) -> DataLoader:
        """Creates the prediction DataLoader."""
        return self.test_dataloader()