#!/usr/bin/env python3
"""Test if base model can do basic Korean generation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

def test_base_model_sanity():
    print("🧪 Testing Base Model Sanity...")
    
    # Test with clean base model
    model_name = "gogamza/kobart-base-v2"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Simple Korean test
    test_input = "안녕하세요. 오늘 날씨가 좋습니다."
    
    print(f"Input: {test_input}")
    
    # Tokenize
    inputs = tokenizer(test_input, return_tensors='pt', max_length=128, truncation=True)
    
    # Generate with simple settings
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=50,
            min_length=5,
            num_beams=1,  # Greedy
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}")
    
    # Check if output makes sense
    if "선거" in generated or "카메라" in generated or "트럼프" in generated:
        print("❌ Base model is also generating nonsense!")
        return False
    elif len(generated.strip()) < 3:
        print("❌ Base model generating too short text")
        return False
    else:
        print("✅ Base model seems to work")
        return True

if __name__ == "__main__":
    test_base_model_sanity()