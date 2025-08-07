#!/usr/bin/env python3
"""
Quick verification that CollateFunction is fixed.
Save as scripts/verify_collate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from data.datamodule import DialogueDataModule
import torch

def main():
    print("🔧 Verifying CollateFunction Fix...")
    
    # Load config and create datamodule
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    datamodule = DialogueDataModule(cfg)
    datamodule.setup("fit")
    
    # Get one batch
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"\n📊 Batch Analysis:")
    print(f"  Labels shape: {batch['labels'].shape}")
    
    # Check for -100 tokens
    labels_flat = batch['labels'].flatten()
    num_minus_100 = (labels_flat == -100).sum().item()
    total_tokens = labels_flat.numel()
    
    print(f"  Total label tokens: {total_tokens}")
    print(f"  -100 padding tokens: {num_minus_100}")
    print(f"  Padding ratio: {num_minus_100/total_tokens:.2%}")
    
    if num_minus_100 > 0:
        print("  ✅ SUCCESS: -100 padding tokens found!")
        print("  🎉 CollateFunction is working correctly!")
        
        # Show sample
        sample_labels = batch['labels'][0]
        non_padding = sample_labels[sample_labels != -100]
        padding_count = (sample_labels == -100).sum().item()
        
        print(f"\n📝 Sample Analysis:")
        print(f"  Sample length: {len(sample_labels)}")
        print(f"  Content tokens: {len(non_padding)}")
        print(f"  Padding tokens: {padding_count}")
        print(f"  Non-padding tokens: {non_padding[:10].tolist()}...")
        
    else:
        print("  ❌ FAILURE: No -100 padding tokens found!")
        print("  🔧 CollateFunction is still broken!")
        
        # Show what we got instead
        unique_values = torch.unique(labels_flat)
        print(f"  Found tokens: {unique_values[:20].tolist()}...")

if __name__ == "__main__":
    main()