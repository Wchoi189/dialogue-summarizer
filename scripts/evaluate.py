#!/usr/bin/env python3
"""
Evaluation script for dialogue summarization using Fire CLI.
Evaluates trained models on validation or test sets with ROUGE metrics.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import click
import pandas as pd
import pytorch_lightning as pl
import torch
from icecream import ic
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.datamodule import DialogueDataModule
from evaluation.evaluator import DialogueEvaluator
from evaluation.metrics import RougeCalculator
from models.kobart_model import KoBARTSummarizationModel
from utils.config_utils import ConfigManager
from utils.file_utils import FileManager
from utils.logging_utils import setup_logging


class DialogueEvaluationRunner:
    """Main evaluation runner for dialogue summarization."""
    
    def __init__(self):
        """Initialize evaluation runner."""
        self.config_manager = ConfigManager()
        self.file_manager = FileManager()
        self.cfg = None
        
    def evaluate(
        self,
        checkpoint_path: str,
        config_name: str = "config",
        config_path: Optional[str] = None,
        split: str = "val",
        output_dir: Optional[Union[str, Path]] = None,
        overrides: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on specified dataset split.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_name: Name of config file
            config_path: Custom config directory path
            split: Dataset split to evaluate (val, test)
            output_dir: Directory to save evaluation results
            overrides: Config overrides
            batch_size: Override batch size for evaluation
            **kwargs: Additional arguments
            
        Returns:
            Evaluation metrics
        """
        ic(f"Starting evaluation on {split} split with checkpoint: {checkpoint_path}")
        
        # Setup configuration
        if config_path:
            self.config_manager = ConfigManager(config_path)
        
        overrides = overrides or []
        
        # Add kwargs as overrides
        for key, value in kwargs.items():
            overrides.append(f"{key}={value}")
        
        # Override batch size if specified
        if batch_size:
            overrides.append(f"dataset.eval_batch_size={batch_size}")
        
        self.cfg = self.config_manager.load_config(
            config_name=config_name,
            overrides=overrides
        )
        assert self.cfg is not None
        
        # Setup logging
        setup_logging(self.cfg)
        
        # Create output directory
        if output_dir is None:
            output_dir = Path(self.cfg.output_dir) / "evaluation"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        ic(f"Results will be saved to: {output_dir}")
        
        # Setup data module
        ic("Setting up data module...")
        datamodule = DialogueDataModule(self.cfg)
        datamodule.setup(stage="test" if split == "test" else "fit")
        
        # Load model from checkpoint
        ic("Loading model from checkpoint...")
        model = KoBARTSummarizationModel.load_from_checkpoint(
            checkpoint_path,
        )
        
        # Setup trainer for evaluation
        trainer = self._setup_trainer(output_dir)
        
        # Run evaluation
        if split == "val":
            ic("Running validation evaluation...")
            results = trainer.validate(model, datamodule=datamodule)
            dataloader = datamodule.val_dataloader()
        elif split == "test":
            ic("Running test evaluation...")
            results = trainer.test(model, datamodule=datamodule)
            dataloader = datamodule.test_dataloader()
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'")
        
        # Get detailed predictions for analysis
        ic("Generating detailed predictions...")
        predictions = trainer.predict(model, dataloader)
        
        # Check if predictions is None
        if predictions is None:
            ic("No predictions returned from trainer.predict; skipping detailed analysis.")
            evaluation_results = {"num_predictions": 0}
        else:
            # Process predictions and save results
            evaluation_results = self._process_predictions(
                predictions, datamodule, split, output_dir
            )
        
        # Combine trainer results with detailed analysis
        if results and isinstance(results[0], dict):
            # Convert any int values to float for consistency
            trainer_results = {k: float(v) if isinstance(v, (int, float)) else v for k, v in results[0].items()}
            evaluation_results = {**evaluation_results, **trainer_results}
        
        # Save evaluation summary
        self._save_evaluation_summary(evaluation_results, output_dir, split)
        
        ic(f"Evaluation complete. Results: {evaluation_results}")
        # Ensure all values are float for type consistency
        evaluation_results = {k: float(v) if isinstance(v, int) else v for k, v in evaluation_results.items()}
        return evaluation_results
    
    def compare_models(
        self,
        checkpoint_paths: List[str],
        model_names: Optional[List[str]] = None,
        config_name: str = "config",
        split: str = "val",
        output_dir: Optional[Union[str, Path]] = None, # MODIFIED
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple model checkpoints.
        
        Args:
            checkpoint_paths: List of checkpoint paths
            model_names: Names for models (optional)
            config_name: Config name
            split: Dataset split
            output_dir: Output directory
            **kwargs: Additional arguments
            
        Returns:
            Comparison results for all models
        """
        ic(f"Comparing {len(checkpoint_paths)} models on {split} split")
        
        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(checkpoint_paths))]
        
        if len(model_names) != len(checkpoint_paths):
            raise ValueError("Number of model names must match number of checkpoints")
        
        # Create comparison output directory
        if output_dir is None:
            output_dir = Path(self.cfg.output_dir if self.cfg else "outputs") / "model_comparison"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_results = {}
        
        # Evaluate each model
        for checkpoint_path, model_name in zip(checkpoint_paths, model_names):
            ic(f"Evaluating {model_name}...")
            
            model_output_dir = output_dir / model_name
            
            try:
                results = self.evaluate(
                    checkpoint_path=checkpoint_path,
                    config_name=config_name,
                    split=split,
                    output_dir=model_output_dir,
                    **kwargs
                )
                comparison_results[model_name] = results
                
            except Exception as e:
                ic(f"Error evaluating {model_name}: {e}")
                comparison_results[model_name] = {"error": str(e)}
        
        # Save comparison summary
        self._save_comparison_summary(comparison_results, output_dir, split)
        
        ic(f"Model comparison complete: {len(comparison_results)} models")
        return comparison_results
    
    def evaluate_submission(
        self,
        submission_file: str,
        reference_file: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Evaluate submission file against references.
        
        Args:
            submission_file: Path to submission CSV
            reference_file: Path to reference CSV (optional)
            output_dir: Output directory for results
            
        Returns:
            Evaluation metrics
        """
        ic(f"Evaluating submission file: {submission_file}")
        
        # Load submission
        submission_df = self.file_manager.load_csv(submission_file)
        ic(f"Loaded submission: {len(submission_df)} predictions")
        
        if reference_file:
            # Load reference data
            reference_df = self.file_manager.load_csv(reference_file)
            ic(f"Loaded references: {len(reference_df)} samples")
            
            # Calculate metrics
            evaluator = DialogueEvaluator()
            metrics = evaluator.evaluate_predictions(
                predictions=submission_df["summary"].tolist(),
                references=reference_df["summary"].tolist()
            )
            
            # Save results
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                results_file = output_dir / "submission_evaluation.json"
                self.file_manager.save_json(metrics, results_file)
                
                ic(f"Submission evaluation saved to: {results_file}")
            
            return metrics
        
        else:
            ic("No reference file provided, only validating submission format")
            
            # Validate submission format
            required_columns = ["fname", "summary"]
            missing_cols = [col for col in required_columns if col not in submission_df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            ic("Submission format validation passed")
            return {"format_valid": True, "num_predictions": len(submission_df)}
    
    def _setup_trainer(self, output_dir: Path) -> pl.Trainer:
        """Setup trainer for evaluation."""
        # Setup logger
        logger = TensorBoardLogger(
            save_dir=output_dir / "logs",
            name="evaluation"
        )

        if self.cfg is None:
            raise ValueError("Configuration (self.cfg) must be loaded before setting up the trainer.")

        trainer = pl.Trainer(
            accelerator=self.cfg.training.accelerator,
            devices=self.cfg.training.devices,
            precision=self.cfg.training.precision,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=self.cfg.training.deterministic,
            benchmark=self.cfg.training.benchmark,
        )

        return trainer
    
    def _process_predictions(
        self,
        predictions: List[List[Any]],
        datamodule: DialogueDataModule,
        split: str,
        output_dir: Path
    ) -> Dict[str, float]:
        """Process and analyze predictions."""
        # Collect all predictions
        all_predictions = []
        all_sample_ids = []
        
        # Each batch_output is a list, possibly of dicts or tuples
        for batch_output in predictions:
            # If batch_output is a dict, use keys; if it's a tuple/list, unpack accordingly
            if isinstance(batch_output, dict):
                all_predictions.extend(batch_output.get("predictions", []))
                all_sample_ids.extend(batch_output.get("sample_ids", []))
            elif isinstance(batch_output, (list, tuple)) and len(batch_output) == 2:
                preds, ids = batch_output
                all_predictions.extend(preds)
                all_sample_ids.extend(ids)
            else:
                # fallback: treat batch_output as predictions only
                all_predictions.extend(batch_output)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            "fname": all_sample_ids,
            "summary": all_predictions
        })
        
        # Save predictions
        predictions_file = output_dir / f"{split}_predictions.csv"
        self.file_manager.save_csv(predictions_df, predictions_file)
        ic(f"Predictions saved to: {predictions_file}")
        
        # Calculate detailed metrics if we have references
        if split == "val":
            # Get reference summaries
            val_data = datamodule.get_sample_data("val", len(predictions_df))
            
            # Create evaluator and calculate metrics
            evaluator = DialogueEvaluator()
            detailed_metrics = evaluator.evaluate_with_analysis(
                predictions=all_predictions,
                references=val_data["summary"].tolist(),
                sample_ids=all_sample_ids
            )
            
            # Save detailed analysis
            analysis_file = output_dir / f"{split}_analysis.json"
            self.file_manager.save_json(detailed_metrics, analysis_file)
            
            return detailed_metrics["overall_metrics"]
        
        return {"num_predictions": len(all_predictions)}
    
    def _save_evaluation_summary(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        split: str
    ) -> None:
        """Save evaluation summary."""
        summary_file = output_dir / f"{split}_evaluation_summary.json"
        self.file_manager.save_json(results, summary_file)
        ic(f"Evaluation summary saved to: {summary_file}")
    
    def _save_comparison_summary(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        output_dir: Path,
        split: str
    ) -> None:
        """Save model comparison summary."""
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        # Save as CSV and JSON
        csv_file = output_dir / f"{split}_model_comparison.csv"
        json_file = output_dir / f"{split}_model_comparison.json"
        
        self.file_manager.save_csv(comparison_df, csv_file)
        self.file_manager.save_json(comparison_results, json_file)
        
        ic(f"Model comparison saved to: {csv_file} and {json_file}")


@click.group()
def cli():
    """Dialogue Summarization Evaluation CLI."""
    pass


@cli.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.option('--config-name', default='config', help='Configuration name')
@click.option('--config-path', type=click.Path(), help='Custom config directory')
@click.option('--split', default='val', type=click.Choice(['val', 'test']), help='Dataset split')
@click.option('--output-dir', type=click.Path(), help='Output directory for results')
@click.option('--batch-size', type=int, help='Override batch size')
def evaluate(checkpoint_path, config_name, config_path, split, output_dir, batch_size):
    """Evaluate model on specified dataset split."""
    runner = DialogueEvaluationRunner()
    
    overrides = []
    if batch_size:
        overrides.append(f"dataset.eval_batch_size={batch_size}")
    
    results = runner.evaluate(
        checkpoint_path=checkpoint_path,
        config_name=config_name,
        config_path=config_path,
        split=split,
        output_dir=output_dir,
        overrides=overrides
    )
    
    click.echo("Evaluation Results:")
    for key, value in results.items():
        click.echo(f"  {key}: {value}")


@cli.command()
@click.argument('checkpoint_paths', nargs=-1, required=True)
@click.option('--model-names', multiple=True, help='Names for models')
@click.option('--config-name', default='config', help='Configuration name')
@click.option('--split', default='val', type=click.Choice(['val', 'test']), help='Dataset split')
@click.option('--output-dir', type=click.Path(), help='Output directory')
def compare(checkpoint_paths, model_names, config_name, split, output_dir):
    """Compare multiple model checkpoints."""
    runner = DialogueEvaluationRunner()
    
    results = runner.compare_models(
        checkpoint_paths=list(checkpoint_paths),
        model_names=list(model_names) if model_names else None,
        config_name=config_name,
        split=split,
        output_dir=output_dir
    )
    
    click.echo("Model Comparison Results:")
    for model_name, metrics in results.items():
        click.echo(f"\n{model_name}:")
        for key, value in metrics.items():
            click.echo(f"  {key}: {value}")


@cli.command()
@click.argument('submission_file', type=click.Path(exists=True))
@click.option('--reference-file', type=click.Path(exists=True), help='Reference CSV file')
@click.option('--output-dir', type=click.Path(), help='Output directory')
def validate_submission(submission_file, reference_file, output_dir):
    """Evaluate submission file against references."""
    runner = DialogueEvaluationRunner()
    
    results = runner.evaluate_submission(
        submission_file=submission_file,
        reference_file=reference_file,
        output_dir=output_dir
    )
    
    click.echo("Submission Evaluation Results:")
    for key, value in results.items():
        click.echo(f"  {key}: {value}")


def main():
    """Main entry point using Click CLI."""
    # Set environment variables
    os.environ["PYTHONHASHSEED"] = "0"
    
    cli()


if __name__ == "__main__":
    main()