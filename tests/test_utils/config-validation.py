#!/usr/bin/env python3
"""Test for configuration duplicates"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager

def test_no_duplicates():
    print("🔍 Testing for Configuration Duplicates...")
    
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    # Check where special tokens are defined
    special_token_locations = []
    
    # Check dataset config
    if "dataset" in cfg and "preprocessing" in cfg.dataset:
        if "special_tokens" in cfg.dataset.preprocessing:
            tokens = cfg.dataset.preprocessing.special_tokens
            special_token_locations.append(f"dataset.preprocessing.special_tokens: {tokens}")
    
    # Check model config
    if "model" in cfg and "tokenizer" in cfg.model:
        if "additional_special_tokens" in cfg.model.tokenizer:
            tokens = cfg.model.tokenizer.additional_special_tokens
            special_token_locations.append(f"model.tokenizer.additional_special_tokens: {tokens}")
    
    # Check global preprocessing config
    if "preprocessing" in cfg:
        if "special_tokens" in cfg.preprocessing:
            tokens = cfg.preprocessing.special_tokens
            special_token_locations.append(f"preprocessing.special_tokens: {tokens}")
    
    print(f"\n📍 Special tokens found in {len(special_token_locations)} locations:")
    for location in special_token_locations:
        print(f"  - {location}")
    
    if len(special_token_locations) == 1:
        print("✅ No duplicates found - single source of truth!")
        return True
    elif len(special_token_locations) > 1:
        print("❌ Duplicates found - multiple definitions exist!")
        print("   This can cause conflicts and inconsistent behavior.")
        return False
    else:
        print("⚠️  No special tokens found anywhere!")
        return False

if __name__ == "__main__":
    success = test_no_duplicates()
    if not success:
        sys.exit(1)