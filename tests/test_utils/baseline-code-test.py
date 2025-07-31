#!/usr/bin/env python3
"""Test baseline configuration settings"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from models.kobart_model import KoBARTSummarizationModel
import torch
# --- Configuration (Should be inside a main function) ---
# Suppress informational messages from the transformers library
from transformers import logging
logging.set_verbosity_error()
def test_baseline_config():
    print("🧪 Testing Baseline Configuration...")
    
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    # Verify key baseline settings
    print(f"✓ Model: {cfg.model.model_name_or_path}")
    print(f"✓ Learning rate: {cfg.training.optimizer.lr}")
    print(f"✓ Batch size: {cfg.training.batch_size}")
    print(f"✓ Max epochs: {cfg.training.max_epochs}")
    print(f"✓ Generation beams: {cfg.training.generation.num_beams}")
    print(f"✓ Max input length: {cfg.dataset.preprocessing.get('max_input_length')}")
    print(f"✓ Max target length: {cfg.dataset.preprocessing.get('max_target_length')}")
    
    # Test model creation
    print("\n🔧 Testing model creation...")
    model = KoBARTSummarizationModel(cfg)
    
    # Test generation with baseline settings
    print("\n🎯 Testing generation...")
    test_input = "#Person1#: 안녕하세요, 의사선생님. #Person2#: 네, 어떻게 오셨나요?"
    
    inputs = model.tokenizer(test_input, return_tensors='pt', max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    generated = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_input}")
    print(f"Generated: {generated}")
    
    # Apply postprocessing
    if hasattr(model, '_apply_post_processing'):
        cleaned = model._apply_post_processing(generated)
        print(f"After postprocessing: {cleaned}")
    
    print("\n✅ Baseline configuration test complete!")

if __name__ == "__main__":
    test_baseline_config()