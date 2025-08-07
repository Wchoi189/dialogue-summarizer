#!/usr/bin/env python3
"""Debug tokenizer issues"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.config_utils import ConfigManager
from models.kobart_model import KoBARTSummarizationModel

def debug_tokenizer():
    print("🔍 Debugging Tokenizer Issues...")
    
    config_manager = ConfigManager()
    cfg = config_manager.load_config("config")
    
    model = KoBARTSummarizationModel(cfg)
    tokenizer = model.tokenizer
    
    print(f"\n📊 Tokenizer Info:")
    print(f"  Model name: {cfg.model.model_name_or_path}")
    print(f"  Tokenizer class: {type(tokenizer).__name__}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    print(f"\n🔤 Special Tokens:")
    print(f"  BOS: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"  EOS: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  UNK: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    
    # Check for <usr> token
    if '<usr>' in tokenizer.get_vocab():
        usr_id = tokenizer.convert_tokens_to_ids('<usr>')
        print(f"  ❌ FOUND <usr> token with ID: {usr_id}")
    else:
        print(f"  ✅ No <usr> token found")
    
    # Test encoding/decoding
    test_text = "#Person1#: 안녕하세요. #Person2#: 네, 안녕하세요."
    print(f"\n🧪 Test Encoding/Decoding:")
    print(f"  Input: {test_text}")
    
    # Encode
    encoded = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"  Encoded IDs: {encoded[:10]}...")
    
    # Decode with different methods
    decoded_with_special = tokenizer.decode(encoded, skip_special_tokens=False)
    decoded_without_special = tokenizer.decode(encoded, skip_special_tokens=True)
    
    print(f"  Decoded (with special): '{decoded_with_special}'")
    print(f"  Decoded (without special): '{decoded_without_special}'")
    
    # Check if <usr> appears in decoding
    if '<usr>' in decoded_with_special or '<usr>' in decoded_without_special:
        print(f"  ❌ <usr> token found in decoded text!")
        return False
    else:
        print(f"  ✅ No <usr> token in decoded text")
        return True

if __name__ == "__main__":
    debug_tokenizer()