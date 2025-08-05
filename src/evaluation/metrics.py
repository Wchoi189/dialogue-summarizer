# FILE: src/evaluation/metrics.py
"""
ROUGE metrics calculation for dialogue summarization evaluation.
"""
import logging
from typing import Dict, List, Union

import numpy as np
from icecream import ic
from rouge_score import rouge_scorer


# Import the centralized post-processing function
from postprocessing.postprocessing import apply_post_processing

logger = logging.getLogger(__name__)

# Use a singleton pattern to initialize the scorer only once
_rouge_scorer_instance = None

def get_rouge_scorer():
    """Initializes and returns a singleton rouge_scorer instance."""
    global _rouge_scorer_instance
    if _rouge_scorer_instance is None:
        ic("Initializing rouge-score package (once)")
        _rouge_scorer_instance = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
    return _rouge_scorer_instance

def calculate_rouge_scores(
    predictions: List[str],
    references: Union[List[str], List[List[str]]],
    average: bool = True,
) -> Union[Dict[str, float], List[Dict]]:
    """
    Calculates ROUGE scores using a centralized post-processor.
    Supports both single and multiple references per prediction.
    """
    if len(predictions) != len(references):
        raise ValueError("Length of predictions and references must match.")

    scorer = get_rouge_scorer()
    
    # Define a standard cleaning config for metrics calculation
    post_cfg = {
        "remove_tokens": ["<s>", "</s>", "<pad>", "<unk>", "<usr>"],
        "korean_specific": {"remove_special_markers": True}
    }

    all_scores = []
    is_multi_ref = isinstance(references[0], list)

    for i in range(len(predictions)):
        pred = apply_post_processing(predictions[i], post_cfg)
        
        if is_multi_ref:
            # Find the best score among multiple references
            best_scores_for_sample = {}
            for ref in references[i]:
                ref = apply_post_processing(ref, post_cfg)
                scores = scorer.score(target=ref, prediction=pred)
                for rouge_type in scores:
                    fmeasure = scores[rouge_type].fmeasure
                    if fmeasure > best_scores_for_sample.get(rouge_type, 0.0):
                        best_scores_for_sample[rouge_type] = fmeasure
            all_scores.append(best_scores_for_sample)
        else:
            # Single reference
            # We need to assert the type for the type checker
            single_ref = references[i]
            assert isinstance(single_ref, str)
            ref = apply_post_processing(single_ref, post_cfg)
            all_scores.append(scorer.score(target=ref, prediction=pred))

    # Aggregate results
    if average:
        agg_scores = {}
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            agg_scores[f'{rouge_type}_f'] = np.mean([s[rouge_type].fmeasure if not is_multi_ref else s[rouge_type] for s in all_scores])
        
        agg_scores['rouge_f'] = (agg_scores['rouge1_f'] + agg_scores['rouge2_f'] + agg_scores['rougeL_f']) / 3
        return agg_scores
    else:
        # For non-averaged, just return the list of score dictionaries
        return all_scores

def format_rouge_results(scores: Dict[str, float], precision: int = 4) -> str:
    """Formats a dictionary of ROUGE F1-scores for display."""
    lines = ["ROUGE Scores:", "-" * 20]
    main_metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'rouge_f']
    for metric in main_metrics:
        if metric in scores:
            lines.append(f"{metric.upper():<10}: {scores[metric]:.{precision}f}")
    return "\n".join(lines)