#!/usr/bin/env python3
"""Debug postprocessing corruption"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from models.kobart_model import KoBARTSummarizationModel
from transformers import logging
logging.set_verbosity_error()

def debug_postprocessing_corruption():
    print("🔍 Debugging Postprocessing Corruption...")
    
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    model = KoBARTSummarizationModel(cfg)
    
    # Test with actual target text
    original_target = "#Person2# 는 숨쉬기 어려워합니다. 의사는 #Person2# 에게 증상을 확인하고, 천식 검사를 위해 폐 전문의에게 가볼 것을 권합니다."
    
    print(f"Original target: '{original_target}'")
    
    # Check if postprocessing config exists
    if "postprocessing" in cfg:
        print(f"Postprocessing config found: {cfg.postprocessing}")
        processed = model._apply_post_processing(original_target)
        print(f"After postprocessing: '{processed}'")
        
        if "#Person2#" not in processed:
            print("❌ CRITICAL: Postprocessing is removing #Person2# tokens!")
            print("This corrupts the training targets!")
            return False
    else:
        print("❌ No postprocessing config found")
        return False
    
    return True

if __name__ == "__main__":
    debug_postprocessing_corruption()