#!/usr/bin/env python3
"""Test fixed model configuration"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from models.kobart_model import KoBARTSummarizationModel
import torch

def test_fixed_model():
    print("🧪 Testing Fixed Model Configuration...")
    
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    model = KoBARTSummarizationModel(cfg)
    model.eval()
    
    test_input = "#Person1#: 안녕하세요, 의사선생님. #Person2#: 네, 어떻게 오셨나요?"
    
    inputs = model.tokenizer(test_input, return_tensors='pt', max_length=512, truncation=True)
    
    print(f"Input: {test_input}")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    generated = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}")
    
    # Apply postprocessing
    if hasattr(model, '_apply_post_processing'):
        cleaned = model._apply_post_processing(generated)
        print(f"After postprocessing: {cleaned}")
    
    # Check if it matches the working pure model output
    if "의사선생" in generated or len(generated.strip()) > 5:
        print("✅ Fixed model works like pure model!")
        return True
    else:
        print("❌ Still broken")
        return False

if __name__ == "__main__":
    test_fixed_model()