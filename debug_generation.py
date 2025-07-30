#!/usr/bin/env python3
"""
Debug script to test the current model's generation quality
"""

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

def test_model_generation():
    print("=== TESTING KoBARTSummarizationModel GENERATION ===")
    
    # Load the original pretrained model
    tokenizer = AutoTokenizer.from_pretrained('digit82/kobart-summarization')
    model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
    
    # Add special tokens
    special_tokens = ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Added {num_added} special tokens")
    
    # Resize model embeddings
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to {len(tokenizer)}")
    
    # Test dialogue from your data
    test_dialogue = '''#Person1#: 안녕하세요, 스미스씨. 저는 호킨스 의사입니다. 오늘 왜 오셨나요?
#Person2#: 건강검진을 받는 것이 좋을 것 같아서요.
#Person1#: 그렇군요, 당신은 5년 동안 건강검진을 받지 않았습니다. 매년 받아야 합니다.
#Person2#: 알고 있습니다. 하지만 아무 문제가 없다면 왜 의사를 만나러 가야 하나요?
#Person1#: 심각한 질병을 피하는 가장 좋은 방법은 이를 조기에 발견하는 것입니다. 그러니 당신의 건강을 위해 최소한 매년 한 번은 오세요.
#Person2#: 알겠습니다.'''
    
    expected_summary = "스미스씨가 건강검진을 받고 있고, 호킨스 의사는 매년 건강검진을 받는 것을 권장합니다."
    
    # Tokenize input
    inputs = tokenizer(
        test_dialogue,
        max_length=512,
        truncation=True,
        padding=False,
        return_tensors='pt'
    )
    
    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")
    
    # Test different generation configs
    generation_configs = [
        {
            "name": "Current (fixed)",
            "max_length": 150,
            "min_length": 20,
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "repetition_penalty": 1.3,
            "length_penalty": 1.5,
            "do_sample": False
        },
        {
            "name": "Conservative",
            "max_length": 100,
            "min_length": 10,
            "num_beams": 2,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "repetition_penalty": 1.2,
            "length_penalty": 1.0,
            "do_sample": False
        },
        {
            "name": "Sampling",
            "max_length": 100,
            "min_length": 10,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3
        }
    ]
    
    print("\n=== GENERATION TESTS ===")
    for config in generation_configs:
        name = config.pop("name")
        print(f"\n{name}:")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **config
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generated: {repr(generated_text)}")
        print(f"  Length: {len(generated_text)} chars")
    
    print(f"\nExpected: {repr(expected_summary)}")
    print(f"Expected length: {len(expected_summary)} chars")

if __name__ == "__main__":
    test_model_generation()
