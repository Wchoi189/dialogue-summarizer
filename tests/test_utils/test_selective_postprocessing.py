#!/usr/bin/env python3
"""Test selective post-processing"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from models.kobart_model import KoBARTSummarizationModel

def test_selective_postprocessing():
    print("🧪 Testing Selective Post-Processing...")
    
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    model = KoBARTSummarizationModel(cfg)
    
    # Test cases
    test_cases = [
        "<usr> #Person1#: 안녕하세요, 의사선생님. #Person2#: 네, 어떻게 오셨나요?</s>",
        "#Person1#: 안녕하세요:::: #Person2#: 네????",
        "<usr>의사선생님.    : 네, 어떻게 오셨나요?</s>",
        "#Person2#는 숨쉬기 어려워합니다<pad><pad>",
    ]
    
    for i, test_text in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Input:  '{test_text}'")
        
        cleaned = model._apply_post_processing(test_text)
        print(f"  Output: '{cleaned}'")
        
        # Check results
        if "<usr>" in cleaned or "</s>" in cleaned or "<pad>" in cleaned:
            print("  ❌ Still contains unwanted tokens")
        elif "#Person" not in test_text or "#Person" in cleaned:
            print("  ✅ Correctly preserved #Person# tokens")
        else:
            print("  ❌ Incorrectly removed #Person# tokens")

if __name__ == "__main__":
    test_selective_postprocessing()