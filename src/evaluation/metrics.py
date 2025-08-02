"""
ROUGE metrics calculation for dialogue summarization evaluation.
Supports Korean text with proper preprocessing.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from icecream import ic

logger = logging.getLogger(__name__)

# Global rouge scorer instance to avoid repeated initialization
_rouge_scorer_instance = None

class RougeCalculator:
    """ROUGE metrics calculator with Korean text support."""
    
    def __init__(self, use_stemmer: bool = False, lang: str = "korean"):
        """
        Initialize ROUGE calculator.
        
        Args:
            use_stemmer: Whether to use stemming (not applicable for Korean)
            lang: Language for text processing
        """
        global _rouge_scorer_instance
        self.use_stemmer = use_stemmer
        self.lang = lang
        
        # Use singleton pattern to avoid repeated initialization
        if _rouge_scorer_instance is None:
            # Try to import rouge-score package
            try:
                from rouge_score import rouge_scorer
                _rouge_scorer_instance = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'],
                    use_stemmer=use_stemmer
                )
                self.use_rouge_score = True
                ic("Using rouge-score package (initialized once)")
                
            except ImportError:
                logger.warning("rouge-score package not available, using custom implementation")
                _rouge_scorer_instance = None
                self.use_rouge_score = False
        else:
            self.use_rouge_score = True
        
        self.rouge_scorer = _rouge_scorer_instance
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str],
        average: bool = True
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Calculate ROUGE scores for predictions vs references.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            average: Whether to return averaged scores
            
        Returns:
            Dictionary with ROUGE scores
        """
        if len(predictions) != len(references):
            raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")
        
        if self.use_rouge_score:
            return self._calculate_with_rouge_score(predictions, references, average)
        else:
            return self._calculate_custom_rouge(predictions, references, average)
    
    def _calculate_with_rouge_score(
        self,
        predictions: List[str],
        references: List[str],
        average: bool
    ) -> Dict[str, Union[float, List[float]]]:
        """Calculate ROUGE using rouge-score package."""
        scores = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
        }
        
        for pred, ref in zip(predictions, references):
            # Preprocess texts
            pred = self._preprocess_text(pred)
            ref = self._preprocess_text(ref)
            
            # Skip empty texts
            if not pred.strip() or not ref.strip():
                logger.warning("Empty prediction or reference, skipping")
                continue
            
            # Calculate scores
            rouge_scores = self.rouge_scorer.score(ref, pred)
            
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                scores[rouge_type]['precision'].append(rouge_scores[rouge_type].precision)
                scores[rouge_type]['recall'].append(rouge_scores[rouge_type].recall)
                scores[rouge_type]['fmeasure'].append(rouge_scores[rouge_type].fmeasure)
        
        # Format results
        results = {}
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            if average:
                results[f'{rouge_type}_precision'] = np.mean(scores[rouge_type]['precision'])
                results[f'{rouge_type}_recall'] = np.mean(scores[rouge_type]['recall'])
                results[f'{rouge_type}_f'] = np.mean(scores[rouge_type]['fmeasure'])
            else:
                results[f'{rouge_type}_precision'] = scores[rouge_type]['precision']
                results[f'{rouge_type}_recall'] = scores[rouge_type]['recall']
                results[f'{rouge_type}_f'] = scores[rouge_type]['fmeasure']
        
        # Calculate overall F1 score
        if average:
            results['rouge_f'] = (
                results['rouge1_f'] + 
                results['rouge2_f'] + 
                results['rougeL_f']
            ) / 3
        
        return results
    
    def _calculate_custom_rouge(
        self,
        predictions: List[str],
        references: List[str],
        average: bool
    ) -> Dict[str, Union[float, List[float]]]:
        """Custom ROUGE implementation for fallback."""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            pred = self._preprocess_text(pred)
            ref = self._preprocess_text(ref)
            
            if not pred.strip() or not ref.strip():
                continue
            
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # ROUGE-1
            rouge1 = self._calculate_rouge_n(pred_tokens, ref_tokens, n=1)
            rouge1_scores.append(rouge1)
            
            # ROUGE-2
            rouge2 = self._calculate_rouge_n(pred_tokens, ref_tokens, n=2)
            rouge2_scores.append(rouge2)
            
            # ROUGE-L
            rougeL = self._calculate_rouge_l(pred_tokens, ref_tokens)
            rougeL_scores.append(rougeL)
        
        results = {}
        if average:
            results['rouge1_f'] = np.mean([s['fmeasure'] for s in rouge1_scores])
            results['rouge2_f'] = np.mean([s['fmeasure'] for s in rouge2_scores])
            results['rougeL_f'] = np.mean([s['fmeasure'] for s in rougeL_scores])
            results['rouge_f'] = (results['rouge1_f'] + results['rouge2_f'] + results['rougeL_f']) / 3
        else:
            results['rouge1_f'] = [s['fmeasure'] for s in rouge1_scores]
            results['rouge2_f'] = [s['fmeasure'] for s in rouge2_scores]
            results['rougeL_f'] = [s['fmeasure'] for s in rougeL_scores]
        
        return results
    
    def _calculate_rouge_n(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str],
        n: int
    ) -> Dict[str, float]:
        """Calculate ROUGE-N scores."""
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if not ref_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
        
        overlap = len(pred_ngrams & ref_ngrams)
        
        precision = overlap / len(pred_ngrams) if pred_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        
        if precision + recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }
    
    def _calculate_rouge_l(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE-L scores using LCS."""
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        if not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
        
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }
    
    def _get_ngrams(self, tokens: List[str], n: int) -> set:
        """Get n-grams from token list."""
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.add(ngram)
        return ngrams
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for ROUGE calculation."""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic cleaning
        text = text.strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # For Korean text, we might want to do additional preprocessing
        if self.lang == "korean":
            # Remove special tokens that might appear in predictions
            special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<usr>"]
            for token in special_tokens:
                text = text.replace(token, "")
            
            # Clean up any remaining special patterns
            text = re.sub(r'#\w+#', '', text)  # Remove person markers
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class MultiReferenceRougeCalculator(RougeCalculator):
    """ROUGE calculator supporting multiple references per prediction."""
    
    def calculate_rouge_multi_ref(
        self,
        predictions: List[str],
        references: List[List[str]],
        average: bool = True
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Calculate ROUGE scores with multiple references.
        
        Args:
            predictions: List of predicted summaries
            references: List of lists of reference summaries
            average: Whether to return averaged scores
            
        Returns:
            Dictionary with ROUGE scores
        """
        if len(predictions) != len(references):
            raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")
        
        all_scores: Dict[str, List[float]] = {
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': []
        }
        
        for pred, refs in zip(predictions, references):
            # Calculate ROUGE against each reference and take the maximum
            best_scores = {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}
            
            for ref in refs:
                scores = self.calculate_rouge([pred], [ref], average=True)
                
                for metric in ['rouge1_f', 'rouge2_f', 'rougeL_f']:
                    score_value = scores[metric]
                    # Ensure we're working with float values
                    if isinstance(score_value, list):
                        score_value = score_value[0] if score_value else 0.0
                    if score_value > best_scores[metric]:
                        best_scores[metric] = score_value
            
            # Add best scores to overall collection
            for metric in ['rouge1_f', 'rouge2_f', 'rougeL_f']:
                all_scores[metric].append(best_scores[metric])
        
        # Return results
        if average:
            results = {}
            for metric in ['rouge1_f', 'rouge2_f', 'rougeL_f']:
                results[metric] = np.mean(all_scores[metric])
            
            # Calculate overall F1 score
            results['rouge_f'] = (
                results['rouge1_f'] + 
                results['rouge2_f'] + 
                results['rougeL_f']
            ) / 3
            
            return results
        else:
            return all_scores


def calculate_rouge_scores(
    predictions: List[str],
    references: Union[List[str], List[List[str]]],
    average: bool = True,
    use_stemmer: bool = False
) -> Dict[str, Union[float, List[float]]]:
    """
    Convenience function to calculate ROUGE scores.
    
    Args:
        predictions: Predicted summaries
        references: Reference summaries (single or multiple per prediction)
        average: Whether to return averaged scores
        use_stemmer: Whether to use stemming
        
    Returns:
        ROUGE scores
    """
    # Check if we have multiple references
    if isinstance(references[0], list):
        calculator = MultiReferenceRougeCalculator(use_stemmer=use_stemmer)
        # Type cast to satisfy type checker
        from typing import cast
        multi_references = cast(List[List[str]], references)
        return calculator.calculate_rouge_multi_ref(predictions, multi_references, average)
    else:
        calculator = RougeCalculator(use_stemmer=use_stemmer)
        # Type cast to satisfy type checker
        from typing import cast
        single_references = cast(List[str], references)
        return calculator.calculate_rouge(predictions, single_references, average)


def format_rouge_results(scores: Dict[str, float], precision: int = 4) -> str:
    """
    Format ROUGE results for display.
    
    Args:
        scores: ROUGE scores dictionary
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("ROUGE Scores:")
    lines.append("-" * 50)
    
    # Main metrics
    main_metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'rouge_f']
    for metric in main_metrics:
        if metric in scores:
            value = scores[metric]
            lines.append(f"{metric.upper():12}: {value:.{precision}f}")
    
    # Additional metrics if available
    additional_metrics = ['rouge1_precision', 'rouge1_recall', 
                         'rouge2_precision', 'rouge2_recall',
                         'rougeL_precision', 'rougeL_recall']
    
    if any(metric in scores for metric in additional_metrics):
        lines.append("")
        lines.append("Detailed Metrics:")
        lines.append("-" * 50)
        
        for metric in additional_metrics:
            if metric in scores:
                value = scores[metric]
                lines.append(f"{metric:18}: {value:.{precision}f}")
    
    return "\n".join(lines)