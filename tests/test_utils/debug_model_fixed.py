#!/usr/bin/env python3
"""
Fixed diagnostic script to identify label processing issues.
Save as scripts/debug_model_fixed.py
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
    
    # ✅ ANALYZE LABELS BEFORE DECODING
    print(f"\n🔍 Label Analysis:")
    labels_sample = batch['labels'][0]
    print(f"  Labels dtype: {labels_sample.dtype}")
    print(f"  Labels min/max: {labels_sample.min().item()} / {labels_sample.max().item()}")
    print(f"  Vocab size: {len(model.tokenizer)}")
    print(f"  Pad token ID: {model.tokenizer.pad_token_id}")
    
    # Check for problematic values
    unique_values = torch.unique(labels_sample)
    print(f"  Unique label values: {unique_values[:20].tolist()}...")  # Show first 20
    
    # Count -100 (padding) tokens
    padding_count = (labels_sample == -100).sum().item()
    print(f"  Padding tokens (-100): {padding_count}/{len(labels_sample)}")
    
    # Test tokenization on original data
    print(f"\n🔤 Raw Data Test:")
    # Get original data sample
    val_data = datamodule.get_sample_data("train", 3)
    sample_dialogue = val_data.iloc[0]['dialogue']
    sample_summary = val_data.iloc[0]['summary']
    
    print(f"  Raw dialogue: {sample_dialogue[:150]}...")
    print(f"  Raw summary: {sample_summary}")
    
    # Test tokenization manually
    print(f"\n🧪 Manual Tokenization Test:")
    dialogue_tokens = model.tokenizer(sample_dialogue, max_length=512, truncation=True, padding=False)
    summary_tokens = model.tokenizer(sample_summary, max_length=128, truncation=True, padding=False)
    
    print(f"  Dialogue tokens: {len(dialogue_tokens['input_ids'])}")
    print(f"  Summary tokens: {len(summary_tokens['input_ids'])}")
    print(f"  Summary token IDs: {summary_tokens['input_ids'][:20]}...")
    
    # Try to decode summary tokens
    try:
        decoded_summary = model.tokenizer.decode(summary_tokens['input_ids'], skip_special_tokens=False)
        print(f"  Decoded summary: {decoded_summary}")
    except Exception as e:
        print(f"  ❌ Decoding failed: {e}")
    
    # ✅ TEST GENERATION WITH SAFE DECODING
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
        
        # SAFE label decoding - remove -100 tokens first
        safe_labels = labels_sample.clone()
        safe_labels[safe_labels == -100] = model.tokenizer.pad_token_id
        
        try:
            target_text = model.tokenizer.decode(safe_labels, skip_special_tokens=True)
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
            
        except Exception as e:
            print(f"  ❌ Target decoding failed: {e}")
            print(f"  Generated: {generated_text}")

if __name__ == "__main__":
    main()