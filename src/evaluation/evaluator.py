"""
Comprehensive evaluation pipeline for dialogue summarization.
Provides detailed analysis and metrics calculation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from icecream import ic

from .metrics import RougeCalculator, MultiReferenceRougeCalculator, format_rouge_results

logger = logging.getLogger(__name__)


class DialogueEvaluator:
    """Comprehensive evaluator for dialogue summarization."""
    
    def __init__(self, use_stemmer: bool = False, lang: str = "korean"):
        """
        Initialize evaluator.
        
        Args:
            use_stemmer: Whether to use stemming
            lang: Language for text processing
        """
        self.rouge_calculator = RougeCalculator(use_stemmer=use_stemmer, lang=lang)
        self.multi_ref_calculator = MultiReferenceRougeCalculator(use_stemmer=use_stemmer, lang=lang)
        
        ic("DialogueEvaluator initialized")
    
    def evaluate_predictions(
        self,
        predictions: List[str],
        references: List[str],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate predictions against references.
        
        Args:
            predictions: Predicted summaries
            references: Reference summaries
            sample_ids: Optional sample identifiers
            
        Returns:
            Evaluation metrics
        """
        ic(f"Evaluating {len(predictions)} predictions")
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge_calculator.calculate_rouge(
            predictions=predictions,
            references=references,
            average=True
        )
        
        # Calculate additional metrics
        length_metrics = self._calculate_length_metrics(predictions, references)
        quality_metrics = self._calculate_quality_metrics(predictions, references)
        
        # Combine all metrics
        all_metrics = {**rouge_scores, **length_metrics, **quality_metrics}
        
        ic(f"Evaluation complete: {all_metrics}")
        return all_metrics
    
    def evaluate_with_analysis(
        self,
        predictions: List[str],
        references: List[str],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate with detailed analysis.
        
        Args:
            predictions: Predicted summaries
            references: Reference summaries
            sample_ids: Sample identifiers
            
        Returns:
            Detailed evaluation results
        """
        # Get overall metrics
        overall_metrics = self.evaluate_predictions(predictions, references, sample_ids)
        
        # Per-sample analysis
        sample_analysis = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            sample_id = sample_ids[i] if sample_ids else f"sample_{i}"
            
            # Calculate individual ROUGE scores
            sample_scores = self.rouge_calculator.calculate_rouge([pred], [ref], average=True)
            
            # Length analysis
            pred_len = len(pred.split())
            ref_len = len(ref.split())
            
            sample_data = {
                "sample_id": sample_id,
                "prediction": pred,
                "reference": ref,
                "pred_length": pred_len,
                "ref_length": ref_len,
                "length_ratio": pred_len / ref_len if ref_len > 0 else 0,
                **sample_scores
            }
            
            sample_analysis.append(sample_data)
        
        return {
            "overall_metrics": overall_metrics,
            "sample_analysis": sample_analysis,
            "summary_stats": self._calculate_summary_stats(sample_analysis)
        }
    
    def evaluate_multi_reference(
        self,
        predictions: List[str],
        references: List[List[str]],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate against multiple references per prediction.
        
        Args:
            predictions: Predicted summaries
            references: List of reference lists
            sample_ids: Sample identifiers
            
        Returns:
            Evaluation metrics
        """
        ic(f"Evaluating {len(predictions)} predictions with multiple references")
        
        # Calculate multi-reference ROUGE scores
        rouge_scores = self.multi_ref_calculator.calculate_rouge_multi_ref(
            predictions=predictions,
            references=references,
            average=True
        )
        
        # Additional metrics
        flat_references = [refs[0] for refs in references]  # Use first reference for other metrics
        length_metrics = self._calculate_length_metrics(predictions, flat_references)
        
        all_metrics = {**rouge_scores, **length_metrics}
        
        ic(f"Multi-reference evaluation complete: {all_metrics}")
        return all_metrics
    
    def _calculate_length_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate length-based metrics."""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        length_ratios = [
            pred_len / ref_len if ref_len > 0 else 0
            for pred_len, ref_len in zip(pred_lengths, ref_lengths)
        ]
        
        return {
            "avg_pred_length": np.mean(pred_lengths),
            "avg_ref_length": np.mean(ref_lengths),
            "avg_length_ratio": np.mean(length_ratios),
            "length_ratio_std": np.std(length_ratios)
        }
    
    def _calculate_quality_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate quality-based metrics."""
        # Count empty predictions
        empty_predictions = sum(1 for pred in predictions if not pred.strip())
        
        # Count very short predictions (less than 3 words)
        short_predictions = sum(1 for pred in predictions if len(pred.split()) < 3)
        
        # Count repetitive predictions (more than 50% repeated words)
        repetitive_predictions = sum(1 for pred in predictions if self._is_repetitive(pred))
        
        total = len(predictions)
        
        return {
            "empty_prediction_rate": empty_predictions / total,
            "short_prediction_rate": short_predictions / total,
            "repetitive_prediction_rate": repetitive_predictions / total
        }
    
    def _is_repetitive(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text is repetitive."""
        words = text.split()
        if len(words) < 4:
            return False
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_words / len(words)
        
        return repetition_ratio > threshold
    
    def _calculate_summary_stats(self, sample_analysis: List[Dict]) -> Dict[str, float]:
        """Calculate summary statistics from sample analysis."""
        if not sample_analysis:
            return {}
        
        # Extract metrics
        rouge1_scores = [s["rouge1_f"] for s in sample_analysis]
        rouge2_scores = [s["rouge2_f"] for s in sample_analysis]
        rougeL_scores = [s["rougeL_f"] for s in sample_analysis]
        length_ratios = [s["length_ratio"] for s in sample_analysis]
        
        return {
            "rouge1_std": np.std(rouge1_scores),
            "rouge2_std": np.std(rouge2_scores),
            "rougeL_std": np.std(rougeL_scores),
            "rouge1_min": np.min(rouge1_scores),
            "rouge1_max": np.max(rouge1_scores),
            "length_ratio_median": np.median(length_ratios),
            "num_samples": len(sample_analysis)
        }
    
    def format_evaluation_report(
        self,
        results: Dict[str, Any],
        include_samples: bool = False
    ) -> str:
        """
        Format evaluation results into a readable report.
        
        Args:
            results: Evaluation results from evaluate_with_analysis
            include_samples: Whether to include per-sample details
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("DIALOGUE SUMMARIZATION EVALUATION REPORT")
        lines.append("=" * 50)
        
        # Overall metrics
        overall = results["overall_metrics"]
        lines.append("\nOverall Metrics:")
        lines.append("-" * 20)
        
        # ROUGE scores
        rouge_metrics = ["rouge1_f", "rouge2_f", "rougeL_f", "rouge_f"]
        for metric in rouge_metrics:
            if metric in overall:
                lines.append(f"{metric.upper():12}: {overall[metric]:.4f}")
        
        # Length metrics
        lines.append(f"\nAvg Pred Length: {overall.get('avg_pred_length', 0):.1f}")
        lines.append(f"Avg Ref Length:  {overall.get('avg_ref_length', 0):.1f}")
        lines.append(f"Length Ratio:    {overall.get('avg_length_ratio', 0):.2f}")
        
        # Quality metrics
        if "empty_prediction_rate" in overall:
            lines.append(f"\nEmpty Predictions: {overall['empty_prediction_rate']:.1%}")
            lines.append(f"Short Predictions: {overall['short_prediction_rate']:.1%}")
            lines.append(f"Repetitive Preds:  {overall['repetitive_prediction_rate']:.1%}")
        
        # Summary stats
        if "summary_stats" in results:
            stats = results["summary_stats"]
            lines.append(f"\nScore Variability:")
            lines.append(f"ROUGE-1 Std: {stats.get('rouge1_std', 0):.4f}")
            lines.append(f"ROUGE-2 Std: {stats.get('rouge2_std', 0):.4f}")
            lines.append(f"ROUGE-L Std: {stats.get('rougeL_std', 0):.4f}")
        
        # Sample details
        if include_samples and "sample_analysis" in results:
            lines.append(f"\nPer-Sample Analysis:")
            lines.append("-" * 20)
            
            samples = results["sample_analysis"]
            for sample in samples[:5]:  # Show first 5 samples
                lines.append(f"\nSample: {sample['sample_id']}")
                lines.append(f"  ROUGE-1: {sample['rouge1_f']:.4f}")
                lines.append(f"  ROUGE-2: {sample['rouge2_f']:.4f}")
                lines.append(f"  ROUGE-L: {sample['rougeL_f']:.4f}")
                lines.append(f"  Length:  {sample['pred_length']}/{sample['ref_length']}")
                lines.append(f"  Pred:    {sample['prediction'][:100]}...")
        
        return "\n".join(lines)