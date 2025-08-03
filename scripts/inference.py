#!/usr/bin/env python3
"""
Inference script for dialogue summarization using Click CLI.
Generates predictions for test data and creates submission files.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd
import pytorch_lightning as pl
import torch
from icecream import ic

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.datamodule import DialogueDataModule
from inference.predictor import DialoguePredictor
from models.kobart_model import KoBARTSummarizationModel
from utils.config_utils import ConfigManager
from utils.file_utils import FileManager, create_submission_file
from utils.logging_utils import setup_logging


class InferenceRunner:
    """Main inference runner for dialogue summarization."""
    
    def __init__(self):
        """Initialize inference runner."""
        self.config_manager = ConfigManager()
        self.file_manager = FileManager()
        self.cfg = None
    
    def predict(
        self,
        checkpoint_path: str,
        test_file: str,
        output_file: str,
        config_name: str = "config",
        config_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        overrides: Optional[List[str]] = None
    ) -> str:
        """
        Generate predictions for test data.
        
        Args:
            checkpoint_path: Path to model checkpoint
            test_file: Path to test CSV file
            output_file: Path to save predictions
            config_name: Configuration name
            config_path: Custom config directory
            batch_size: Override batch size
            overrides: Config overrides
            
        Returns:
            Path to output file
        """
        ic(f"Starting inference with checkpoint: {checkpoint_path}")
        ic(f"Test file: {test_file}, Output: {output_file}")
        
        # Setup configuration
        if config_path:
            self.config_manager = ConfigManager(config_path)
        
        overrides = overrides or []
        
        if batch_size:
            overrides.append(f"dataset.eval_batch_size={batch_size}")
        
        self.cfg = self.config_manager.load_config(
            config_name=config_name,
            overrides=overrides
        )
        
        # Setup logging
        setup_logging(self.cfg)
        
        # Load model
        ic("Loading model from checkpoint...")
        # Before (Remove the cfg argument due to struct error.)
        # model = KoBARTSummarizationModel.load_from_checkpoint(
        #     checkpoint_path,
        #     cfg=self.cfg
        # )

        # AFTER (loads checkpoint without passing any extra arguments)
        model = KoBARTSummarizationModel.load_from_checkpoint(
            checkpoint_path
        )
        
        # Create predictor
        predictor = DialoguePredictor(model, self.cfg)
        
        # Load test data
        ic(f"Loading test data from: {test_file}")
        test_df = self.file_manager.load_csv(test_file)
        
        # Generate predictions
        ic("Generating predictions...")
        predictions_df = predictor.predict_dataframe(test_df)
        
        # Save predictions
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.file_manager.save_csv(predictions_df, output_path)
        ic(f"Predictions saved to: {output_path}")
        
        return str(output_path)
    
    def create_submission(
        self,
        checkpoint_path: str,
        output_file: str = "submission.csv",
        config_name: str = "config",
        config_path: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> str:
        """
        Create submission file for competition.
        
        Args:
            checkpoint_path: Path to model checkpoint
            output_file: Output submission file path
            config_name: Configuration name
            config_path: Custom config directory
            batch_size: Override batch size
            
        Returns:
            Path to submission file
        """
        ic("Creating submission file...")
        
        # Use default test file path from config
        if config_path:
            self.config_manager = ConfigManager(config_path)
        
        # Load config to get data paths
        self.cfg = self.config_manager.load_config(config_name)
        
        # Get test file path
        data_path = Path(self.cfg.dataset.data_path)
        test_file = data_path / self.cfg.dataset.files.test
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        # Generate predictions
        temp_predictions = "temp_predictions.csv"
        self.predict(
            checkpoint_path=checkpoint_path,
            test_file=str(test_file),
            output_file=temp_predictions,
            config_name=config_name,
            config_path=config_path,
            batch_size=batch_size
        )
        
        # Load predictions and format for submission
        predictions_df = self.file_manager.load_csv(temp_predictions)
        
        # Get submission template path
        template_path = data_path / self.cfg.dataset.files.submission_template
        
        # Create properly formatted submission file
        create_submission_file(
            predictions_df=predictions_df,
            output_path=output_file,
            template_path=template_path if template_path.exists() else None
        )
        
        # Clean up temp file
        Path(temp_predictions).unlink(missing_ok=True)
        
        ic(f"Submission file created: {output_file}")
        return output_file
    
    def batch_predict(
        self,
        checkpoint_path: str,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.csv",
        config_name: str = "config"
    ) -> List[str]:
        """
        Run batch prediction on multiple files.
        
        Args:
            checkpoint_path: Path to model checkpoint
            input_dir: Directory with input CSV files
            output_dir: Directory to save outputs
            pattern: File pattern to match
            config_name: Configuration name
            
        Returns:
            List of output file paths
        """
        ic(f"Running batch prediction on {input_dir}")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find input files
        input_files = list(input_path.glob(pattern))
        ic(f"Found {len(input_files)} files to process")
        
        output_files = []
        
        for input_file in input_files:
            output_file = output_path / f"{input_file.stem}_predictions.csv"
            
            try:
                ic(f"Processing: {input_file.name}")
                self.predict(
                    checkpoint_path=checkpoint_path,
                    test_file=str(input_file),
                    output_file=str(output_file),
                    config_name=config_name
                )
                output_files.append(str(output_file))
                
            except Exception as e:
                ic(f"Error processing {input_file.name}: {e}")
        
        ic(f"Batch prediction complete: {len(output_files)} files processed")
        return output_files


@click.group()
def cli():
    """Dialogue Summarization Inference CLI."""
    pass


@cli.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
# @click.option('--config-name', default='config', help='Configuration name') # Uncomment for phase specific config loading
@click.option('--config-name', default='config-baseline-centralized', help='Configuration name')
@click.option('--config-path', type=click.Path(), help='Custom config directory')
@click.option('--batch-size', type=int, help='Override evaluation batch size')
@click.option(
    '--override', 
    'overrides', 
    multiple=True, 
    help='Custom Hydra overrides (e.g., "generation.max_length=80")'
)
def predict(checkpoint_path, test_file, output_file, config_name, config_path, batch_size, overrides):
    """Generate predictions for test data."""
    runner = InferenceRunner()
    
    # Convert the 'overrides' tuple to a list so we can modify it
    final_overrides = list(overrides)
    
    # The existing --batch-size flag is a convenient shortcut
    if batch_size:
        final_overrides.append(f"dataset.eval_batch_size={batch_size}")
    
    output_path = runner.predict(
        checkpoint_path=checkpoint_path,
        test_file=test_file,
        output_file=output_file,
        config_name=config_name,
        config_path=config_path,
        overrides=final_overrides
    )
    
    click.echo(f"Predictions saved to: {output_path}")


@cli.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.option('--output-file', default='submission.csv', help='Output submission file')
@click.option('--config-name', default='config', help='Configuration name')
@click.option('--config-path', type=click.Path(), help='Custom config directory')
@click.option('--batch-size', type=int, help='Override batch size')
def submission(checkpoint_path, output_file, config_name, config_path, batch_size):
    """Create submission file for competition."""
    runner = InferenceRunner()
    
    submission_path = runner.create_submission(
        checkpoint_path=checkpoint_path,
        output_file=output_file,
        config_name=config_name,
        config_path=config_path,
        batch_size=batch_size
    )
    
    click.echo(f"Submission file created: {submission_path}")


@cli.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--pattern', default='*.csv', help='File pattern to match')
@click.option('--config-name', default='config', help='Configuration name')
def batch(checkpoint_path, input_dir, output_dir, pattern, config_name):
    """Run batch prediction on multiple files."""
    runner = InferenceRunner()
    
    output_files = runner.batch_predict(
        checkpoint_path=checkpoint_path,
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
        config_name=config_name
    )
    
    click.echo(f"Batch prediction complete:")
    for output_file in output_files:
        click.echo(f"  {output_file}")


def main():
    """Main entry point using Click CLI."""
    # Set environment variables
    os.environ["PYTHONHASHSEED"] = "0"
    
    cli()


if __name__ == "__main__":
    main()