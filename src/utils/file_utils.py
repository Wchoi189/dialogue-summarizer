"""
File utilities for data loading, saving, and path management.
Enhanced with Korean text support and robust error handling.
"""

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from icecream import ic
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FileManager:
    """Enhanced file manager for dialogue summarization project."""
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize file manager.
        
        Args:
            base_path: Base directory for file operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        ic(f"FileManager initialized with base_path: {self.base_path}")
    
    def load_csv(
        self, 
        file_path: Union[str, Path], 
        encoding: str = "utf-8",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file with Korean text support.
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding (default: utf-8 for Korean)
            **kwargs: Additional pandas.read_csv arguments
            
        Returns:
            Loaded DataFrame
        """
        file_path = self._resolve_path(file_path)
        
        try:
            ic(f"Loading CSV: {file_path}")
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            ic(f"CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed, trying cp949 encoding")
            df = pd.read_csv(file_path, encoding="cp949", **kwargs)
            ic(f"CSV loaded with cp949 encoding: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            raise
    
    def save_csv(
        self, 
        df: pd.DataFrame, 
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        index: bool = False,
        **kwargs
    ) -> None:
        """
        Save DataFrame to CSV with Korean text support.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            encoding: File encoding
            index: Whether to include row indices
            **kwargs: Additional pandas.to_csv arguments
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            ic(f"Saving CSV: {file_path}")
            df.to_csv(file_path, encoding=encoding, index=index, **kwargs)
            ic(f"CSV saved successfully: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Failed to save CSV {file_path}: {e}")
            raise
    
    def load_json(
        self, 
        file_path: Union[str, Path], 
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """
        Load JSON file with Korean text support.
        
        Args:
            file_path: Path to JSON file
            encoding: File encoding
            
        Returns:
            Loaded JSON data
        """
        file_path = self._resolve_path(file_path)
        
        try:
            ic(f"Loading JSON: {file_path}")
            with open(file_path, "r", encoding=encoding) as f:
                data = json.load(f)
            ic(f"JSON loaded successfully")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON {file_path}: {e}")
            raise
    
    def save_json(
        self, 
        data: Dict[str, Any], 
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        ensure_ascii: bool = False,
        indent: int = 2
    ) -> None:
        """
        Save data to JSON file with Korean text support.
        
        Args:
            data: Data to save
            file_path: Output file path
            encoding: File encoding
            ensure_ascii: Whether to escape non-ASCII characters
            indent: JSON indentation
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            ic(f"Saving JSON: {file_path}")
            with open(file_path, "w", encoding=encoding) as f:
                json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
            ic(f"JSON saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save JSON {file_path}: {e}")
            raise
    
    def load_yaml(
        self, 
        file_path: Union[str, Path], 
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """
        Load YAML file.
        
        Args:
            file_path: Path to YAML file
            encoding: File encoding
            
        Returns:
            Loaded YAML data
        """
        file_path = self._resolve_path(file_path)
        
        try:
            ic(f"Loading YAML: {file_path}")
            with open(file_path, "r", encoding=encoding) as f:
                data = yaml.safe_load(f)
            ic(f"YAML loaded successfully")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load YAML {file_path}: {e}")
            raise
    
    def save_yaml(
        self, 
        data: Dict[str, Any], 
        file_path: Union[str, Path],
        encoding: str = "utf-8"
    ) -> None:
        """
        Save data to YAML file.
        
        Args:
            data: Data to save
            file_path: Output file path
            encoding: File encoding
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            ic(f"Saving YAML: {file_path}")
            with open(file_path, "w", encoding=encoding) as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            ic(f"YAML saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save YAML {file_path}: {e}")
            raise
    
    def load_text(
        self, 
        file_path: Union[str, Path], 
        encoding: str = "utf-8"
    ) -> str:
        """
        Load text file.
        
        Args:
            file_path: Path to text file
            encoding: File encoding
            
        Returns:
            File contents as string
        """
        file_path = self._resolve_path(file_path)
        
        try:
            ic(f"Loading text file: {file_path}")
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            ic(f"Text file loaded: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            raise
    
    def save_text(
        self, 
        content: str, 
        file_path: Union[str, Path],
        encoding: str = "utf-8"
    ) -> None:
        """
        Save text to file.
        
        Args:
            content: Text content to save
            file_path: Output file path
            encoding: File encoding
        """
        file_path = self._resolve_path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            ic(f"Saving text file: {file_path}")
            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)
            ic(f"Text file saved: {len(content)} characters")
            
        except Exception as e:
            logger.error(f"Failed to save text file {file_path}: {e}")
            raise
    
    def create_directory(self, dir_path: Union[str, Path]) -> Path:
        """
        Create directory if it doesn't exist.
        
        Args:
            dir_path: Directory path to create
            
        Returns:
            Created directory path
        """
        dir_path = self._resolve_path(dir_path)
        
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            ic(f"Directory created/verified: {dir_path}")
            return dir_path
            
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            raise
    
    def copy_file(
        self, 
        source: Union[str, Path], 
        destination: Union[str, Path]
    ) -> None:
        """
        Copy file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
        """
        source = self._resolve_path(source)
        destination = self._resolve_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            ic(f"Copying file: {source} -> {destination}")
            shutil.copy2(source, destination)
            ic(f"File copied successfully")
            
        except Exception as e:
            logger.error(f"Failed to copy file {source} -> {destination}: {e}")
            raise
    
    def backup_file(
        self, 
        file_path: Union[str, Path], 
        backup_suffix: str = ".bak"
    ) -> Path:
        """
        Create backup of file.
        
        Args:
            file_path: File to backup
            backup_suffix: Suffix for backup file
            
        Returns:
            Path to backup file
        """
        file_path = self._resolve_path(file_path)
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        
        if file_path.exists():
            self.copy_file(file_path, backup_path)
            ic(f"Backup created: {backup_path}")
        
        return backup_path
    
    def validate_data_files(self, cfg: DictConfig) -> bool:
        """
        Validate that all required data files exist.
        
        Args:
            cfg: Configuration containing data paths
            
        Returns:
            True if all files exist
            
        Raises:
            FileNotFoundError: If required files are missing
        """
        data_path = Path(cfg.dataset.data_path)
        required_files = {
            "train": cfg.dataset.files.train,
            "dev": cfg.dataset.files.dev,
            "test": cfg.dataset.files.test,
            "submission_template": cfg.dataset.files.submission_template
        }
        
        missing_files = []
        
        for file_type, filename in required_files.items():
            file_path = data_path / filename
            if not file_path.exists():
                missing_files.append(f"{file_type}: {file_path}")
            else:
                ic(f"✓ Found {file_type}: {file_path}")
        
        if missing_files:
            error_msg = f"Missing required data files:\n" + "\n".join(missing_files)
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        ic("All required data files validated successfully")
        return True
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            return {"exists": False}
        
        stat = file_path.stat()
        info = {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified_time": stat.st_mtime,
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir(),
            "suffix": file_path.suffix,
            "stem": file_path.stem,
        }
        
        ic(f"File info for {file_path}: {info}")
        return info
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Resolve path relative to base_path if not absolute.
        
        Args:
            path: Path to resolve
            
        Returns:
            Resolved Path object
        """
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_path / path


def create_submission_file(
    predictions_df: pd.DataFrame,
    output_path: Union[str, Path],
    template_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create submission file matching the exact format.
    
    Args:
        predictions_df: DataFrame with fname and summary columns
        output_path: Path to save submission file
        template_path: Path to submission template (optional)
    """
    ic(f"Creating submission file: {output_path}")
    
    # Ensure correct column order and format
    if template_path:
        template_df = pd.read_csv(template_path)
        ic(f"Template format: {list(template_df.columns)}")
        
        # Match template format exactly
        if template_df.columns[0] == "":  # Unnamed index column
            submission_df = predictions_df[["fname", "summary"]].copy()
            submission_df.reset_index(inplace=True)
            submission_df.rename(columns={"index": ""}, inplace=True)
        else:
            submission_df = predictions_df[["fname", "summary"]].copy()
    else:
        # Default format with index
        submission_df = predictions_df[["fname", "summary"]].copy()
        submission_df.reset_index(inplace=True)
    
    # Save submission file
    file_manager = FileManager()
    file_manager.save_csv(submission_df, output_path, index=False)
    
    ic(f"Submission file created: {len(submission_df)} predictions")


def load_dialogue_data(
    data_path: Union[str, Path],
    split: str = "train"
) -> pd.DataFrame:
    """
    Load dialogue data with proper Korean text handling.
    
    Args:
        data_path: Path to data directory
        split: Data split (train, dev, test)
        
    Returns:
        Loaded DataFrame
    """
    data_path = Path(data_path)
    file_path = data_path / f"{split}.csv"
    
    file_manager = FileManager(data_path)
    df = file_manager.load_csv(file_path)
    
    ic(f"Loaded {split} data: {len(df)} samples")
    return df