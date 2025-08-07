# FILE: scripts/analyze_data.py
import sys
from pathlib import Path
from typing import Dict

import click
import pandas as pd
from icecream import ic
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add project root to the Python path to allow imports from `src`
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import DialogueDataModule
from src.utils.config_utils import ConfigManager


# --- ANALYSIS FUNCTIONS (Moved from datamodule.py) ---

def analyze_dataset(datamodule: DialogueDataModule, split: str = "train") -> Dict[str, any]:
    """Analyzes and returns statistics for a given data split."""
    ic(f"Analyzing {split} dataset...")
    data = datamodule.train_data if split == 'train' else datamodule.val_data if split == 'val' else datamodule.test_data
    if data is None:
        raise ValueError(f"Data for split '{split}' has not been loaded. Run datamodule.setup() first.")

    analysis = {
        "num_samples": len(data),
        "columns": list(data.columns),
    }

    input_col = datamodule.cfg.dataset.columns.input
    dialogues = data[input_col].dropna()
    analysis["dialogue_stats"] = {
        "avg_length_chars": dialogues.str.len().mean(),
        "avg_length_words": dialogues.str.split().str.len().mean(),
    }
    
    target_col = datamodule.cfg.dataset.columns.target
    if target_col in data.columns:
        summaries = data[target_col].dropna()
        analysis["summary_stats"] = {
            "avg_length_chars": summaries.str.len().mean(),
            "avg_length_words": summaries.str.split().str.len().mean(),
        }
    
    ic(f"Dataset analysis complete for {split}")
    return analysis

def print_data_sample(datamodule: DialogueDataModule, split: str = "train", idx: int = 0) -> None:
    """Prints a formatted sample from the dataset for inspection."""
    console = Console()
    data = datamodule.train_data if split == 'train' else datamodule.val_data if split == 'val' else datamodule.test_data
    if data is None or len(data) <= idx:
        console.print(f"[red]Sample index {idx} not available in {split}[/red]")
        return
    
    sample = data.iloc[idx]
    
    console.print(Panel.fit(f"[bold blue]Sample ID:[/bold blue] {str(sample[datamodule.cfg.dataset.columns.id])}"))
    console.print(Panel(Text(str(sample[datamodule.cfg.dataset.columns.input])), title="[green]Dialogue[/green]"))
    
    target_col = datamodule.cfg.dataset.columns.target
    if target_col in sample and pd.notna(sample[target_col]):
        console.print(Panel(Text(str(sample[target_col])), title="[yellow]Summary[/yellow]"))


# --- COMMAND-LINE INTERFACE ---

@click.group()
def cli():
    """Data analysis and inspection tools."""
    pass

def _setup_datamodule(config_name: str) -> DialogueDataModule:
    """Helper to load config and setup the datamodule."""
    config_manager = ConfigManager()
    cfg = config_manager.load_config(config_name)
    datamodule = DialogueDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule

@cli.command()
@click.option('--split', default='train', type=click.Choice(['train', 'val', 'test']), help='Dataset split to analyze.')
@click.option('--config-name', default='config-baseline-centralized', help='Name of the main config file.')
def analyze(split: str, config_name: str):
    """Analyze and print statistics for a data split."""
    datamodule = _setup_datamodule(config_name)
    analysis_results = analyze_dataset(datamodule, split)
    
    console = Console()
    console.print(analysis_results)

@cli.command()
@click.option('--split', default='train', type=click.Choice(['train', 'val', 'test']), help='Dataset split to sample from.')
@click.option('--index', default=0, type=int, help='Index of the sample to print.')
@click.option('--config-name', default='config-baseline-centralized', help='Name of the main config file.')
def sample(split: str, index: int, config_name: str):
    """Print a single formatted sample from a data split."""
    datamodule = _setup_datamodule(config_name)
    print_data_sample(datamodule, split, index)

if __name__ == "__main__":
    cli()