#!/usr/bin/env python3
"""Comprehensive postprocessing test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from models.kobart_model import KoBARTSummarizationModel

def test_complete_postprocessing_setup():
    print("🔧 Testing Complete Postprocessing Setup...")
    
    # Test 1: Default config loads correctly
    print("\n1. Testing default config loading...")
    try:
        config_manager = ConfigManager()
        cfg = config_manager.load_config("config")
        
        if "postprocessing" in cfg:
            print("✅ Postprocessing config loaded successfully")
            print(f"   Remove tokens: {cfg.postprocessing.get('remove_tokens', [])}")
        else:
            print("❌ Postprocessing config not found in main config")
            return False
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False
    
    # Test 2: Model can use postprocessing
    print("\n2. Testing model postprocessing integration...")
    try:
        model = KoBARTSummarizationModel(cfg)
        
        # Test problematic input
        test_input = "<usr>안녕하세요 #Person1# 의사선생님<s></s><pad>"
        result = model._apply_post_processing(test_input)
        
        print(f"   Input:  '{test_input}'")
        print(f"   Output: '{result}'")
        
        # Check if unwanted tokens are removed
        unwanted_found = any(token in result for token in ["<usr>", "<s>", "<pad>"])
        if unwanted_found:
            print("❌ Unwanted tokens still present")
            return False
        else:
            print("✅ Unwanted tokens removed successfully")
    except Exception as e:
        print(f"❌ Error in model postprocessing: {e}")
        return False
    
    # Test 3: Different profiles work
    print("\n3. Testing different postprocessing profiles...")
    profiles = ["default", "aggressive", "minimal"]
    
    for profile in profiles:
        try:
            cfg_profile = config_manager.load_config(
                config_name="config",
                overrides=[f"postprocessing={profile}"]
            )
            
            if "postprocessing" in cfg_profile:
                print(f"✅ Profile '{profile}' loaded successfully")
            else:
                print(f"❌ Profile '{profile}' not loaded")
                return False
        except Exception as e:
            print(f"❌ Error loading profile '{profile}': {e}")
            return False
    
    print("\n🎉 All postprocessing tests passed!")
    return True

if __name__ == "__main__":
    success = test_complete_postprocessing_setup()
    if not success:
        sys.exit(1)