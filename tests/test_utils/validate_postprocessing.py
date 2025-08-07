#!/usr/bin/env python3
"""Test postprocessing configuration"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.config_utils import ConfigManager
from models.kobart_model import KoBARTSummarizationModel

def test_postprocessing():
    print("🧪 Testing Postprocessing Configuration...")
    
    configs_to_test = ["default", "aggressive", "minimal"]
    
    for config_name in configs_to_test:
        print(f"\n--- Testing {config_name} postprocessing ---")
        
        try:
            config_manager = ConfigManager()
            cfg = config_manager.load_config(
                config_name="config",
                overrides=[f"postprocessing={config_name}"]  # ✅ Changed
            )
            
            model = KoBARTSummarizationModel(cfg)
            
            test_text = "<usr>#Person1#: 안녕하세요<s>의사선생님</s><pad><pad>"
            cleaned = model._apply_post_processing(test_text)
            
            print(f"  Input:  '{test_text}'")
            print(f"  Output: '{cleaned}'")
            
            if "<usr>" in cleaned or "<pad>" in cleaned:
                print(f"  ❌ Still contains unwanted tokens")
            else:
                print(f"  ✅ Cleaned successfully")
        
        except Exception as e:
            print(f"  ❌ Error with {config_name}: {e}")

if __name__ == "__main__":
    test_postprocessing()
    print("✅ Post-processing tests completed") 