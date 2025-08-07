#!/usr/bin/env python3
"""
Quick diagnostic script to analyze current model performance issues.
"""

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
import pandas as pd
import sys
from pathlib import Path

def load_test_data(data_path="/home/wb2x/workspace/dialogue-summarizer/data"):
    """Load a few test samples for diagnosis."""
    train_path = Path(data_path) / "train.csv"
    if not train_path.exists():
        print(f"❌ Data file not found: {train_path}")
        return None
    
    df = pd.read_csv(train_path)
    print(f"✅ Loaded {len(df)} training samples")
    return df.head(5)  # Just first 5 for testing

def analyze_baseline_model():
    """Analyze the baseline KoBART model performance."""
    print("🔍 ANALYZING BASELINE MODEL PERFORMANCE")
    print("="*60)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "digit82/kobart-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    print(f"✅ Model loaded: {model_name}")
    print(f"✅ Tokenizer vocab size: {len(tokenizer)}")
    
    # Load test data
    test_df = load_test_data()
    if test_df is None:
        return
    
    print("\n📝 TESTING MODEL ON SAMPLE DIALOGUES")
    print("-" * 60)
    
    for idx, row in test_df.iterrows():
        print(f"\n--- SAMPLE {idx + 1} ---")
        dialogue = row['dialogue']
        true_summary = row['summary']
        
        print(f"Dialogue ({len(dialogue)} chars):")
        print(f"  {dialogue[:200]}...")
        
        print(f"True Summary ({len(true_summary)} chars):")
        print(f"  {true_summary}")
        
        # Test with different generation configs
        configs = [
            {"max_length": 50, "num_beams": 4, "name": "Current Config"},
            {"max_length": 120, "num_beams": 4, "name": "Extended Length"},
            {"max_length": 80, "num_beams": 6, "name": "More Beams"},
        ]
        
        for config in configs:
            try:
                # Tokenize input
                inputs = tokenizer(
                    dialogue,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=config["max_length"],
                        num_beams=config["num_beams"],
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                
                # Decode
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                print(f"Generated ({config['name']}):")
                print(f"  {generated}")
                
                # Quick quality check
                if len(generated.strip()) < 10:
                    print("  ⚠️  WARNING: Very short output")
                if generated == dialogue[:len(generated)]:
                    print("  ⚠️  WARNING: Copying input")
                
            except Exception as e:
                print(f"  ❌ Error with {config['name']}: {e}")
        
        print("-" * 40)
        if idx >= 2:  # Only test first 3 samples
            break

def analyze_tokenization_issues():
    """Analyze potential tokenization problems."""
    print("\n🔤 TOKENIZATION ANALYSIS")
    print("="*60)
    
    model_name = "digit82/kobart-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test Korean dialogue patterns
    test_texts = [
        "#Person1# 안녕하세요. 어떻게 지내세요?",
        "#Person2# 저는 잘 지내고 있습니다. 감사합니다.",
        "#Person1# 오늘 날씨가 좋네요. #Person2# 네, 정말 좋습니다.",
        "요약: 두 사람이 인사를 나누고 날씨에 대해 이야기했습니다."
    ]
    
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"IDs: {token_ids}")
        
        # Check for special tokens
        special_tokens = [t for t in tokens if '#' in t]
        if special_tokens:
            print(f"Special tokens found: {special_tokens}")

def check_config_compatibility():
    """Check if current config matches model requirements."""
    print("\n⚙️  CONFIG COMPATIBILITY CHECK")
    print("="*60)
    
    # Load your current config values
    current_config = {
        "max_input_length": 512,
        "max_target_length": 60,  # This looks too short!
        "generation_max_length": 50,  # This is definitely too short!
        "learning_rate": 3e-6,
        "batch_size": 32,
    }
    
    # Recommended values based on research
    recommended_config = {
        "max_input_length": 512,
        "max_target_length": 120,  # Korean summaries need more space
        "generation_max_length": 100,  # Allow longer outputs
        "learning_rate": 5e-5,  # Higher for better convergence
        "batch_size": 16,  # Smaller for better gradients
    }
    
    print("Current vs Recommended Configuration:")
    print("-" * 40)
    
    for key in current_config:
        current = current_config[key]
        recommended = recommended_config[key]
        status = "✅" if current == recommended else "⚠️"
        
        print(f"{status} {key:20}: {current:10} → {recommended}")
    
    print("\n🔧 CRITICAL FIXES NEEDED:")
    
    fixes = []
    if current_config["max_target_length"] < 100:
        fixes.append("📏 Increase max_target_length to 120+ for Korean summaries")
    
    if current_config["generation_max_length"] < 80:
        fixes.append("🎯 Increase generation max_length to 100+")
    
    if current_config["learning_rate"] < 1e-5:
        fixes.append("📈 Increase learning rate to 5e-5 for better convergence")
    
    for fix in fixes:
        print(f"  {fix}")

def main():
    """Run all diagnostic checks."""
    print("🚀 KOREAN DIALOGUE SUMMARIZATION MODEL DIAGNOSTIC")
    print("="*70)
    
    try:
        analyze_baseline_model()
        analyze_tokenization_issues()
        check_config_compatibility()
        
        print("\n🎯 IMMEDIATE ACTION ITEMS:")
        print("="*40)
        print("1. 📏 Fix generation lengths (max_length: 100+)")
        print("2. 📈 Increase learning rate (5e-5)")
        print("3. 🔤 Check Korean text preprocessing")
        print("4. 📊 Run full EDA on Korean dialogue patterns")
        print("5. 🔄 Retrain with corrected configuration")
        
        print("\n💡 NEXT STEPS:")
        print("1. Run the EDA script first: python korean_dialogue_eda.py")
        print("2. Update configs based on diagnostics")
        print("3. Test with corrected generation parameters")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        print("Make sure you're in the project directory and data files exist.")

if __name__ == "__main__":
    main()