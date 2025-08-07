# FILE: src/data/collate.py
"""
Collate function for dialogue summarization datasets.
Handles padding and batching of tokenized data.
"""
from typing import List, Dict
import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

class CollateFunction:
    """Fixed collate function for dialogue data."""
    
    def __init__(self, tokenizer: PreTrainedTokenizerFast, is_inference: bool = False):
        """
        Initialize collate function.
        """
        self.tokenizer = tokenizer
        self.is_inference = is_inference
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        """
        # Separate keys
        tensor_keys = ["input_ids", "attention_mask"]
        if not self.is_inference:
            tensor_keys.append("labels")
        
        # Initialize batch dict
        batched = {}
        
        # Collect sample IDs
        sample_ids = [sample["sample_id"] for sample in batch]
        batched["sample_ids"] = sample_ids
        
        # Pad and batch tensor data
        for key in tensor_keys:
            if key in batch[0]:
                sequences = [sample[key] for sample in batch]
                
                # Ensure all sequences are 1D tensors
                sequences = [seq.squeeze() if seq.dim() > 1 else seq for seq in sequences]
                
                # Determine padding value
                if key == "labels":
                    padding_value = -100
                elif key == "input_ids":
                    padding_value = self.tokenizer.pad_token_id or 0
                else:
                    padding_value = 0
                
                # Pad sequences
                padded = torch.nn.utils.rnn.pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=padding_value
                )
                
                batched[key] = padded
        
        return batched

def create_collate_fn(tokenizer: PreTrainedTokenizerFast, is_inference: bool = False) -> CollateFunction:
    """
    Create collate function for DataLoader.
    """
    return CollateFunction(tokenizer, is_inference)
