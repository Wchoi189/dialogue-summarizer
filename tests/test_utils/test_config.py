#!/usr/bin/env python3
"""
Test script to verify the fixed configuration works before training
"""

import sys
import torch
from omegaconf import OmegaConf
from src.models.kobart_model import KoBARTSummarizationModel

def test_fixed_configuration():
    print("=== TESTING FIXED CONFIGURATION ===")
    
    try:
        # Load the updated config
        with open('configs/config.yaml', 'r') as f:
            cfg = OmegaConf.load(f)
        
        print(f"Model: {cfg.model.model_name_or_path}")
        print(f"Generation max_length: {cfg.training.generation.max_length}")
        print(f"Generation min_length: {cfg.training.generation.min_length}")
        print(f"Repetition penalty: {cfg.training.generation.repetition_penalty}")
        
        # Initialize model (this will test tokenizer and special tokens)
        model = KoBARTSummarizationModel(cfg)
        
        print(f"✅ Model initialized successfully")
        print(f"Vocab size: {len(model.tokenizer)}")
        
        # Test a simple generation
        test_input = "#Person1#: 안녕하세요, 의사선생님. #Person2#: 네, 안녕하세요."
        inputs = model.tokenizer(test_input, return_tensors='pt', max_length=256, truncation=True)
        
        # Use the generation config from training
        gen_config = cfg.training.generation
        with torch.no_grad():
            outputs = model.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=gen_config.max_length,
                min_length=gen_config.min_length,
                num_beams=gen_config.num_beams,
                no_repeat_ngram_size=gen_config.no_repeat_ngram_size,
                early_stopping=gen_config.early_stopping,
                do_sample=gen_config.do_sample,
                repetition_penalty=gen_config.repetition_penalty,
                length_penalty=gen_config.length_penalty
            )
        
        generated = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Test generation successful")
        print(f"Generated: {repr(generated)}")
        print(f"Length: {len(generated)} chars")
        
        print("\n=== CONFIGURATION READY FOR TRAINING ===")
        print("You can now start fresh training with: python scripts/train.py")
        
    except Exception as e:
        print(f"❌ Error in configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_fixed_configuration()
