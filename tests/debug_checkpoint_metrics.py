#!/usr/bin/env python3
"""Debug checkpoint metric names"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager

def debug_checkpoint_metrics():
    print("🔍 Debugging Checkpoint Metrics...")
    
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    print(f"Current monitor setting: {cfg.training.monitor}")
    print(f"Current mode: {cfg.training.mode}")
    
    # Check what the actual metric names should be
    print("\nExpected metric names from logs:")
    print("- val/rouge1_f")
    print("- val/rouge2_f") 
    print("- val/rougeL_f")
    print("- val/rouge_f")
    
    print(f"\nCheckpoint filename pattern:")
    print("best-{epoch:02d}-{val/rouge_f:.5f}")
    print("                    ↑")
    print("This should match the monitor setting!")

if __name__ == "__main__":
    debug_checkpoint_metrics()