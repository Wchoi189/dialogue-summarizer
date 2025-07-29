## **File 3: Configuration Utilities**

### `src/utils/config_utils.py`


"""
Configuration utilities for Hydra-based configuration management.
Adapted and enhanced from existing config utilities.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ConfigManager:
    """Enhanced configuration manager for Hydra configs."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing Hydra configs. If None, uses default.
        """
        self.config_dir = Path(config_dir) if config_dir else self._get_default_config_dir()
        self._hydra_initialized = False
    
    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory."""
        # Assume we're running from project root or scripts/
        current_dir = Path.cwd()
        
        # Check if we're in scripts/ directory
        if current_dir.name == "scripts":
            config_dir = current_dir.parent / "configs"
        else:
            config_dir = current_dir / "configs"
        
        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
        return config_dir
    
    def initialize_hydra(self, version_base: Optional[str] = None) -> None:
        """Initialize Hydra with the config directory."""
        if self._hydra_initialized:
            return
        
        try:
            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()
            
            # Initialize with config directory
            initialize_config_dir(
                config_dir=str(self.config_dir.absolute()),
                version_base=version_base
            )
            self._hydra_initialized = True
            logger.info(f"Hydra initialized with config dir: {self.config_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hydra: {e}")
            raise
    
    def load_config(
        self, 
        config_name: str = "config",
        overrides: Optional[List[str]] = None,
        return_hydra_config: bool = False
    ) -> DictConfig:
        """
        Load configuration using Hydra.
        
        Args:
            config_name: Name of the main config file (without .yaml)
            overrides: List of override strings (e.g., ["model=kobart", "training.batch_size=32"])
            return_hydra_config: Whether to return Hydra's runtime config as well
            
        Returns:
            Loaded configuration as DictConfig
        """
        if not self._hydra_initialized:
            self.initialize_hydra()
        
        try:
            cfg = compose(
                config_name=config_name,
                overrides=overrides or []
            )
            
            logger.info(f"Configuration loaded: {config_name}")
            if overrides:
                logger.info(f"Applied overrides: {overrides}")
            
            return cfg
            
        except Exception as e:
            logger.error(f"Failed to load config {config_name}: {e}")
            raise
    
    def validate_config(self, cfg: DictConfig) -> bool:
            """
            Validate configuration for required fields and data types.
            
            Args:
                cfg: Configuration to validate
                
            Returns:
                True if configuration is valid
                
            Raises:
                ValueError: If configuration is invalid
            """
            required_sections = ["dataset", "model", "training"]
            
            for section in required_sections:
                if section not in cfg:
                    raise ValueError(f"Missing required config section: {section}")
            
            # Validate dataset config
            if "data_path" not in cfg.dataset:
                raise ValueError("dataset.data_path is required")
            
            data_path = Path(cfg.dataset.data_path)
            if not data_path.exists():
                raise ValueError(f"Data path does not exist: {data_path}")
            
            # Validate required data files
            required_files = ["train.csv", "dev.csv", "test.csv"]
            for file_name in required_files:
                file_path = data_path / file_name
                if not file_path.exists():
                    raise ValueError(f"Required data file not found: {file_path}")
            
            # Validate model config
            if "name" not in cfg.model:
                raise ValueError("model.name is required")
            
            # Validate training config (use max_epochs instead of num_epochs)
            if "max_epochs" not in cfg.training:
                raise ValueError("training.max_epochs is required")
            
            logger.info("Configuration validation passed")
            return True
    
    def save_config(
        self, 
        cfg: DictConfig, 
        output_path: Union[str, Path],
        resolve: bool = True
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            cfg: Configuration to save
            output_path: Path to save the configuration
            resolve: Whether to resolve interpolations before saving
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if resolve:
            yaml_content = OmegaConf.to_yaml(cfg, resolve=True)
        else:
            yaml_content = OmegaConf.to_yaml(cfg)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def merge_configs(
        self, 
        base_cfg: DictConfig, 
        override_cfg: DictConfig
    ) -> DictConfig:
        """
        Merge two configurations with override taking precedence.
        
        Args:
            base_cfg: Base configuration
            override_cfg: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = OmegaConf.merge(base_cfg, override_cfg)
        return DictConfig(merged)
    
    def get_config_summary(self, cfg: DictConfig) -> Dict[str, Any]:
        """
        Get a summary of key configuration parameters.
        """
        summary = {
            "model_name": cfg.get("model", {}).get("name", "unknown"),
            "dataset_path": cfg.get("dataset", {}).get("data_path", "unknown"),
            "batch_size": cfg.get("training", {}).get("batch_size", "unknown"),
            "learning_rate": cfg.get("training", {}).get("optimizer", {}).get("lr", "unknown"),
            "max_epochs": cfg.get("training", {}).get("max_epochs", "unknown"),
        }
        
        return summary


def setup_config_logging(cfg: DictConfig) -> None:
    """
    Setup logging based on configuration.
    
    Args:
        cfg: Configuration containing logging settings
    """
    log_level = cfg.get("logging", {}).get("level", "INFO").upper()
    log_format = cfg.get("logging", {}).get("format", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger.info(f"Logging configured: level={log_level}")


def create_output_dir(cfg: DictConfig) -> Path:
    """
    Create output directory based on configuration.
    
    Args:
        cfg: Configuration containing output settings
        
    Returns:
        Path to created output directory
    """
    output_base = Path(cfg.get("output_dir", "outputs"))
    
    # Create timestamped directory if not specified
    if "run_name" in cfg:
        output_dir = output_base / cfg.run_name
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_base / f"run_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")
    
    return output_dir