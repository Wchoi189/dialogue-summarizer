#!/usr/bin/env python3
"""
Deep debug to trace the entire data pipeline step by step.
Save as scripts/debug_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config_utils import ConfigManager
from data.preprocessing import create_preprocessor
from models.kobart_model import KoBARTSummarizationModel
import torch

def main():
    print("🔍 Deep Pipeline Debug...")
    
    # Load config
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    # Create model and preprocessor
    model = KoBARTSummarizationModel(cfg)
    preprocessor = create_preprocessor(cfg)
    
    print(f"\n📊 Tokenizer Info:")
    print(f"  Vocab size: {len(model.tokenizer)}")
    print(f"  Pad token: '{model.tokenizer.pad_token}' (ID: {model.tokenizer.pad_token_id})")
    print(f"  EOS token: '{model.tokenizer.eos_token}' (ID: {model.tokenizer.eos_token_id})")
    print(f"  BOS token: '{model.tokenizer.bos_token}' (ID: {model.tokenizer.bos_token_id})")
    
    # Test with sample data
    sample_dialogue = "#Person1#: 안녕하세요, 스미스씨. 저는 호킨스 의사입니다. 오늘 왜 오셨나요?\n#Person2#: 건강검진을 받는 것이 좋을 것 같아서요."
    sample_summary = "스미스씨가 건강검진을 받고 있고, 호킨스 의사는 매년 건강검진을 받는 것을 권장합니다."
    
    print(f"\n🧪 Step-by-Step Tokenization:")
    print(f"Raw dialogue: {sample_dialogue}")
    print(f"Raw summary: {sample_summary}")
    
    # Test preprocessing steps
    print(f"\n1️⃣ Preprocessing Step:")
    cleaned_dialogue = preprocessor.preprocess_dialogue(sample_dialogue)
    cleaned_summary = preprocessor.preprocess_summary(sample_summary)
    print(f"Cleaned dialogue: {cleaned_dialogue}")
    print(f"Cleaned summary: {cleaned_summary}")
    
    # Test manual tokenization
    print(f"\n2️⃣ Manual Tokenization:")
    dialogue_tokens = model.tokenizer(
        cleaned_dialogue,
        max_length=512,
        truncation=True,
        padding=False,
        add_special_tokens=True,
        return_tensors=None
    )
    
    summary_tokens = model.tokenizer(
        cleaned_summary,
        max_length=128,
        truncation=True,
        padding=False,
        add_special_tokens=True,
        return_tensors=None
    )
    
    print(f"Dialogue token IDs: {dialogue_tokens['input_ids'][:10]}...")
    print(f"Summary token IDs: {summary_tokens['input_ids'][:10]}...")
    print(f"Dialogue length: {len(dialogue_tokens['input_ids'])}")
    print(f"Summary length: {len(summary_tokens['input_ids'])}")
    
    # Check for out-of-vocab tokens
    max_dialogue_token = max(dialogue_tokens['input_ids'])
    max_summary_token = max(summary_tokens['input_ids'])
    print(f"Max dialogue token: {max_dialogue_token} (vocab: {len(model.tokenizer)})")
    print(f"Max summary token: {max_summary_token} (vocab: {len(model.tokenizer)})")
    
    if max_dialogue_token >= len(model.tokenizer):
        print(f"❌ DIALOGUE: Out-of-vocab token detected!")
    if max_summary_token >= len(model.tokenizer):
        print(f"❌ SUMMARY: Out-of-vocab token detected!")
    
    # Test preprocessor prepare_inputs
    print(f"\n3️⃣ Preprocessor prepare_inputs:")
    try:
        prepared_inputs = preprocessor.prepare_inputs(
            dialogue=sample_dialogue,
            summary=sample_summary,
            is_inference=False
        )
        
        print(f"Prepared keys: {list(prepared_inputs.keys())}")
        for key, tensor in prepared_inputs.items():
            if hasattr(tensor, 'shape'):
                print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                if key == 'labels':
                    print(f"    Labels min/max: {tensor.min().item()}/{tensor.max().item()}")
                    print(f"    Labels sample: {tensor.squeeze()[:10].tolist()}")
            else:
                print(f"  {key}: {type(tensor)}")
    except Exception as e:
        print(f"❌ prepare_inputs failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test decoding
    print(f"\n4️⃣ Decoding Test:")
    try:
        # Test dialogue decoding
        decoded_dialogue = model.tokenizer.decode(dialogue_tokens['input_ids'], skip_special_tokens=False)
        print(f"Decoded dialogue: {decoded_dialogue}")
        
        # Test summary decoding
        decoded_summary = model.tokenizer.decode(summary_tokens['input_ids'], skip_special_tokens=False)
        print(f"Decoded summary: {decoded_summary}")
        
    except Exception as e:
        print(f"❌ Decoding failed: {e}")
    
    # Test with padded sequences
    print(f"\n5️⃣ Padding Test:")
    
    # Manually pad summary to 128 length with pad_token_id
    padded_summary_ids = summary_tokens['input_ids'] + [model.tokenizer.pad_token_id] * (128 - len(summary_tokens['input_ids']))
    print(f"Padded summary length: {len(padded_summary_ids)}")
    print(f"Padded summary sample: {padded_summary_ids[:10]}")
    print(f"Padded summary end: {padded_summary_ids[-10:]}")
    
    # Now convert to labels format (-100 for padding)
    labels = []
    for token_id in padded_summary_ids:
        if token_id == model.tokenizer.pad_token_id:
            labels.append(-100)
        else:
            labels.append(token_id)
    
    print(f"Labels sample: {labels[:10]}")
    print(f"Labels end: {labels[-10:]}")
    print(f"Number of -100 tokens: {labels.count(-100)}")

if __name__ == "__main__":
    main()