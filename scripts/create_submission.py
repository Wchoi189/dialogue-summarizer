#!/usr/bin/env python3
"""
Submission creation script for dialogue summarization competition.
Creates properly formatted submission files from model predictions.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from icecream import ic

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from utils.file_utils import FileManager, create_submission_file


class SubmissionCreator:
    """Creates competition submission files."""
    
    def __init__(self):
        """Initialize submission creator."""
        self.config_manager = ConfigManager()
        self.file_manager = FileManager()
    
    def create_from_predictions(
        self,
        predictions_file: str,
        output_file: str,
        template_file: Optional[str] = None
    ) -> str:
        """
        Create submission file from predictions CSV.
        
        Args:
            predictions_file: Path to predictions CSV
            output_file: Path to output submission file
            template_file: Path to submission template (optional)
            
        Returns:
            Path to created submission file
        """
        ic(f"Creating submission from: {predictions_file}")
        
        # Load predictions
        predictions_df = self.file_manager.load_csv(predictions_file)
        ic(f"Loaded {len(predictions_df)} predictions")
        
        # Validate predictions format
        required_columns = ["fname", "summary"]
        missing_cols = [col for col in required_columns if col not in predictions_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in predictions: {missing_cols}")
        
        # Create submission file
        create_submission_file(
            predictions_df=predictions_df,
            output_path=output_file,
            template_path=template_file
        )
        
        ic(f"Submission file created: {output_file}")
        return output_file
    
    def create_from_model(
        self,
        checkpoint_path: str,
        output_file: str,
        config_name: str = "config",
        test_file: Optional[str] = None,
        template_file: Optional[str] = None
    ) -> str:
        """
        Create submission directly from model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            output_file: Path to output submission file
            config_name: Configuration name
            test_file: Path to test file (uses config default if None)
            template_file: Path to submission template
            
        Returns:
            Path to created submission file
        """
        ic(f"Creating submission from model: {checkpoint_path}")
        
        # Load configuration
        cfg = self.config_manager.load_config(config_name)
        
        # Get test file path
        if test_file is None:
            data_path = Path(cfg.dataset.data_path)
            test_file = data_path / cfg.dataset.files.test
        
        if not Path(test_file).exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        # Get template file path
        if template_file is None:
            data_path = Path(cfg.dataset.data_path)
            template_file = data_path / cfg.dataset.files.submission_template
            template_file = template_file if template_file.exists() else None
        
        # Generate predictions using inference script
        from inference import InferenceRunner
        
        inference_runner = InferenceRunner()
        temp_predictions = "temp_submission_predictions.csv"
        
        try:
            # Generate predictions
            inference_runner.predict(
                checkpoint_path=checkpoint_path,
                test_file=str(test_file),
                output_file=temp_predictions,
                config_name=config_name
            )
            
            # Create submission from predictions
            submission_path = self.create_from_predictions(
                predictions_file=temp_predictions,
                output_file=output_file,
                template_file=str(template_file) if template_file else None
            )
            
            return submission_path
            
        finally:
            # Clean up temporary file
            Path(temp_predictions).unlink(missing_ok=True)
    
    def validate_submission(
        self,
        submission_file: str,
        template_file: Optional[str] = None
    ) -> bool:
        """
        Validate submission file format.
        
        Args:
            submission_file: Path to submission file
            template_file: Path to template file for format comparison
            
        Returns:
            True if validation passes
        """
        ic(f"Validating submission: {submission_file}")
        
        try:
            # Load submission
            submission_df = self.file_manager.load_csv(submission_file)
            ic(f"Submission has {len(submission_df)} rows")
            
            # Check basic requirements
            if len(submission_df) == 0:
                ic("❌ Submission is empty")
                return False
            
            # Check required columns
            required_columns = ["fname", "summary"]
            missing_cols = [col for col in required_columns if col not in submission_df.columns]
            
            if missing_cols:
                ic(f"❌ Missing required columns: {missing_cols}")
                return False
            
            # Check for empty summaries
            empty_summaries = submission_df["summary"].isna().sum()
            if empty_summaries > 0:
                ic(f"⚠️  Found {empty_summaries} empty summaries")
            
            # Compare with template if provided
            if template_file and Path(template_file).exists():
                template_df = self.file_manager.load_csv(template_file)
                
                # Check column order
                if list(submission_df.columns) != list(template_df.columns):
                    ic(f"⚠️  Column order differs from template")
                    ic(f"Submission: {list(submission_df.columns)}")
                    ic(f"Template:   {list(template_df.columns)}")
                
                # Check number of rows
                if len(submission_df) != len(template_df):
                    ic(f"⚠️  Row count differs from template: {len(submission_df)} vs {len(template_df)}")
                
                # Check fname values match
                if "fname" in template_df.columns:
                    template_fnames = set(template_df["fname"])
                    submission_fnames = set(submission_df["fname"])
                    
                    if template_fnames != submission_fnames:
                        missing_fnames = template_fnames - submission_fnames
                        extra_fnames = submission_fnames - template_fnames
                        
                        if missing_fnames:
                            ic(f"❌ Missing fnames: {missing_fnames}")
                        if extra_fnames:
                            ic(f"❌ Extra fnames: {extra_fnames}")
                        
                        return False
            
            ic("✓ Submission validation passed")
            return True
            
        except Exception as e:
            ic(f"❌ Validation failed: {e}")
            return False
    
    def fix_submission_format(
        self,
        input_file: str,
        output_file: str,
        template_file: Optional[str] = None
    ) -> str:
        """
        Fix submission file format to match requirements.
        
        Args:
            input_file: Path to input submission file
            output_file: Path to output fixed file
            template_file: Path to template file
            
        Returns:
            Path to fixed submission file
        """
        ic(f"Fixing submission format: {input_file}")
        
        # Load input file
        df = self.file_manager.load_csv(input_file)
        
        # Ensure required columns exist
        if "fname" not in df.columns:
            if "id" in df.columns:
                df["fname"] = df["id"]
            else:
                raise ValueError("No fname or id column found")
        
        if "summary" not in df.columns:
            if "prediction" in df.columns:
                df["summary"] = df["prediction"]
            else:
                raise ValueError("No summary or prediction column found")
        
        # Keep only required columns
        df = df[["fname", "summary"]].copy()
        
        # Fill empty summaries with default
        df["summary"] = df["summary"].fillna("요약을 생성할 수 없습니다.")
        
        # Sort by fname to match expected order
        df = df.sort_values("fname").reset_index(drop=True)
        
        # Create submission file with proper format
        create_submission_file(
            predictions_df=df,
            output_path=output_file,
            template_path=template_file
        )
        
        ic(f"Fixed submission saved: {output_file}")
        return output_file


@click.group()
def cli():
    """Submission Creation CLI."""
    pass


@cli.command()
@click.argument('predictions_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--template-file', type=click.Path(), help='Submission template file')
def from_predictions(predictions_file, output_file, template_file):
    """Create submission from predictions CSV."""
    creator = SubmissionCreator()
    
    submission_path = creator.create_from_predictions(
        predictions_file=predictions_file,
        output_file=output_file,
        template_file=template_file
    )
    
    click.echo(f"Submission created: {submission_path}")


@cli.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--config-name', default='config', help='Configuration name')
@click.option('--test-file', type=click.Path(), help='Test file path')
@click.option('--template-file', type=click.Path(), help='Submission template')
def from_model(checkpoint_path, output_file, config_name, test_file, template_file):
    """Create submission directly from model."""
    creator = SubmissionCreator()
    
    submission_path = creator.create_from_model(
        checkpoint_path=checkpoint_path,
        output_file=output_file,
        config_name=config_name,
        test_file=test_file,
        template_file=template_file
    )
    
    click.echo(f"Submission created: {submission_path}")


@cli.command()
@click.argument('submission_file', type=click.Path(exists=True))
@click.option('--template-file', type=click.Path(), help='Template file for comparison')
def validate(submission_file, template_file):
    """Validate submission file format."""
    creator = SubmissionCreator()
    
    is_valid = creator.validate_submission(
        submission_file=submission_file,
        template_file=template_file
    )
    
    if is_valid:
        click.echo("✓ Submission validation passed")
    else:
        click.echo("❌ Submission validation failed")
        exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--template-file', type=click.Path(), help='Template file')
def fix_format(input_file, output_file, template_file):
    """Fix submission file format."""
    creator = SubmissionCreator()
    
    fixed_path = creator.fix_submission_format(
        input_file=input_file,
        output_file=output_file,
        template_file=template_file
    )
    
    click.echo(f"Fixed submission saved: {fixed_path}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()