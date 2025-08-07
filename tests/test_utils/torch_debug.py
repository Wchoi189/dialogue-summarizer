#!/usr/bin/env python3
"""Debug torch.compile issues"""
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

def debug_torch_compile():
    print("🔧 Debugging torch.compile...")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Compile available: {hasattr(torch, 'compile')}")
    
    if not hasattr(torch, 'compile'):
        print("❌ torch.compile not available - upgrade PyTorch to 2.0+")
        return False
    
    # Test with simple model first
    print("\n🧪 Testing with simple model...")
    try:
        model_name = "digit82/kobart-summarization"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        
        print("✅ Model loaded successfully")
        
        # Try compilation
        print("🔧 Attempting compilation...")
        compiled_model = torch.compile(model, mode="default")
        print("✅ Compilation successful!")
        
        # Test generation
        print("🎯 Testing generation...")
        test_input = "안녕하세요"
        inputs = tokenizer(test_input, return_tensors='pt')
        
        with torch.no_grad():
            outputs = compiled_model.generate(
                input_ids=inputs['input_ids'],
                max_length=20,
                num_beams=1,  # Simple generation
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated}")
        print("✅ Compiled model generation works!")
        
        return True
        
    except Exception as e:
        print(f"❌ torch.compile test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_torch_compile()