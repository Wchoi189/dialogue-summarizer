#!/usr/bin/env python3
"""Find a working KoBART model"""
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

def test_models():
    models = [
        "gogamza/kobart-base-v2",
        "hyunwoongko/kobart", 
        "ainize/kobart-news",
        "facebook/bart-base"  # English fallback
    ]
    
    test_input = "#Person1#: 안녕하세요, 의사선생님. #Person2#: 네, 어떻게 오셨나요?"
    
    for model_name in models:
        print(f"\n🧪 Testing {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
            
            # Add special tokens
            special_tokens = ['#Person1#', '#Person2#', '#Person3#']
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            model.resize_token_embeddings(len(tokenizer))
            
            inputs = tokenizer(test_input, return_tensors='pt', max_length=256, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=120,
                    num_beams=3,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  Generated: {generated}")
            
            # Quality check
            if "국민일보" in generated or "SBS" in generated or "현대" in generated:
                print("  ❌ Generates news-like text")
            elif len(generated.strip()) < 3:
                print("  ⚠️  Too short")
            else:
                print("  ✅ Looks reasonable - USE THIS MODEL!")
                return model_name
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return None

if __name__ == "__main__":
    working_model = test_models()
    if working_model:
        print(f"\n🎉 Use this model: {working_model}")
    else:
        print("\n😞 No working model found")