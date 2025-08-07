# FILE: src/evaluation/evaluator.py
"""
Core evaluation pipeline for dialogue summarization.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from icecream import ic

from .metrics import calculate_rouge_scores

logger = logging.getLogger(__name__)


class DialogueEvaluator:
    """Core evaluator for dialogue summarization."""
    
    def __init__(self):
        """Initializes the evaluator."""
        ic("DialogueEvaluator initialized")
    
    def evaluate_predictions(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Calculates key metrics for a list of predictions against references.
        """
        ic(f"Evaluating {len(predictions)} predictions")
        
        # Calculate ROUGE scores using the centralized metrics module
        rouge_scores = calculate_rouge_scores(
            predictions=predictions,
            references=references,
            average=True
        )
        
        # Calculate additional quality and length metrics
        length_metrics = self._calculate_length_metrics(predictions, references)
        quality_metrics = self._calculate_quality_metrics(predictions, references)
        
        # Combine all metrics into a single dictionary
        all_metrics = {**rouge_scores, **length_metrics, **quality_metrics}
        ic(f"Evaluation complete: {all_metrics}")
        return all_metrics
    
    # --- Helper methods used by the main evaluation pipeline ---

    def _calculate_length_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculates statistics about prediction and reference lengths."""
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        # Handle division by zero for empty references
        length_ratios = [
            p_len / r_len if r_len > 0 else 0
            for p_len, r_len in zip(pred_lengths, ref_lengths)
        ]
        
        return {
            "avg_pred_length": float(np.mean(pred_lengths)),
            "avg_ref_length": float(np.mean(ref_lengths)),
            "avg_length_ratio": float(np.mean(length_ratios)),
        }
    
    def _calculate_quality_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculates high-level quality metrics."""
        total = len(predictions)
        if total == 0:
            return {}
            
        empty_predictions = sum(1 for pred in predictions if not pred.strip())
        short_predictions = sum(1 for pred in predictions if len(pred.split()) < 3)
        
        return {
            "empty_prediction_rate": empty_predictions / total,
            "short_prediction_rate": short_predictions / total,
        }