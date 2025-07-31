#!/usr/bin/env python3
"""Test pure base model without modifications"""
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

def test_pure_base_model():
    print("🧪 Testing Pure Base Model...")
    
    model_name = "digit82/kobart-summarization"
    
    # Load pure model without any modifications
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Add special tokens (minimal)
    special_tokens = ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    # Test input
    test_input = "#Person1#: 안녕하세요, 의사선생님. #Person2#: 네, 어떻게 오셨나요?"
    
    inputs = tokenizer(test_input, return_tensors='pt', max_length=512, truncation=True)
    
    print(f"Input: {test_input}")
    
    # Generate with EXACT baseline settings
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}")
    
    # Check if it makes sense
    if any(word in generated for word in ["2017", "PGA", "컨소시엄", "유럽재생"]):
        print("❌ Pure base model is also broken!")
        return False
    elif len(generated.strip()) < 5:
        print("⚠️  Very short output")
        return False
    else:
        print("✅ Pure base model works!")
        return True

if __name__ == "__main__":
    test_pure_base_model()