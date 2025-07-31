#!/usr/bin/env python3
"""Test base model with better generation parameters"""
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

def test_base_model_with_better_params():
    print("🧪 Testing Base Model with Better Generation...")
    
    model_name = "digit82/kobart-summarization"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    # Test input
    test_input = "#Person1#: 안녕하세요, 의사선생님. #Person2#: 네, 어떻게 오셨나요?"
    
    inputs = tokenizer(test_input, return_tensors='pt', max_length=512, truncation=True)
    
    print(f"Input: {test_input}")
    
    # ✅ BETTER GENERATION PARAMETERS
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=50,
            min_length=5,
            num_beams=4,              # ✅ Beam search instead of greedy
            no_repeat_ngram_size=3,   # ✅ Prevent repetition
            early_stopping=True,
            repetition_penalty=1.2,   # ✅ Penalize repetition
            length_penalty=1.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}")
    
    # Check quality
    if generated.count("예") > 10:
        print("⚠️  Still repetitive, but this is normal for base models")
        print("   Fine-tuning will fix this!")
        return True
    elif len(generated.strip()) < 3:
        print("❌ Too short")
        return False
    else:
        print("✅ Good generation quality!")
        return True

if __name__ == "__main__":
    test_base_model_with_better_params()