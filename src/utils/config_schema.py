# In src/utils/config_schema.py

from dataclasses import dataclass, field
from typing import Optional, List, Any

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 5e-5
    weight_decay: float = 0.05

@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_ratio: float = 0.1

@dataclass
class GenerationConfig:
    max_length: int = 64
    min_length: int = 15
    num_beams: int = 4
    length_penalty: float = 1.0

@dataclass
class TrainingConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # This makes it truly optional! Default is 1 epoch.
    log_every_n_steps: Optional[int] = None
    val_check_interval: float = 1.0
    
    max_epochs: int = 10
    batch_size: int = 16
    
    # ... add any other training parameters here

@dataclass
class ProjectConfig:
    # This is your main config class
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # ... add other top-level configs here (model, dataset, etc.)