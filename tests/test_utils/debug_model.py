#!/usr/bin/env python3
"""
Quick diagnostic script to test tokenization and generation.
Save as scripts/debug_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from data.datamodule import DialogueDataModule
from models.kobart_model import KoBARTSummarizationModel
import torch

def main():
    print("🔍 Debugging Model Performance...")
    
    # Load config
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    # Create model and data
    model = KoBARTSummarizationModel(cfg)
    datamodule = DialogueDataModule(cfg)
    datamodule.setup("fit")
    
    # Get a small sample from training data
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"\n📊 Batch Info:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Sample IDs: {batch['sample_ids'][:3]}")
    
    # Test tokenization
    print(f"\n🔤 Tokenization Test:")
    sample_input = model.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
    sample_target = model.tokenizer.decode(batch['labels'][0], skip_special_tokens=False)
    
    print(f"  Input:  {sample_input[:200]}...")
    print(f"  Target: {sample_target[:100]}...")
    
    # Test generation
    print(f"\n🎯 Generation Test:")
    model.eval()
    with torch.no_grad():
        # Take first sample only
        test_input_ids = batch['input_ids'][:1]
        test_attention_mask = batch['attention_mask'][:1]
        
        # Generate with your current config
        outputs = model.generate(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask
        )
        
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        target_text = model.tokenizer.decode(batch['labels'][0], skip_special_tokens=True)
        
        print(f"  Generated: {generated_text}")
        print(f"  Target:    {target_text}")
        print(f"  Gen length: {len(generated_text.split())} words")
        print(f"  Target length: {len(target_text.split())} words")
        
        # Quick ROUGE calculation
        from evaluation.metrics import RougeCalculator
        rouge_calc = RougeCalculator()
        rouge_scores = rouge_calc.calculate_rouge([generated_text], [target_text])
        print(f"  ROUGE-1: {rouge_scores['rouge1_f']:.4f}")
        print(f"  ROUGE-L: {rouge_scores['rougeL_f']:.4f}")

if __name__ == "__main__":
    main()