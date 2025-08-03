# FILE: src/postprocessing/postprocessing.py
import re
from typing import Dict

def _remove_repetitive_phrases(text: str) -> str:
    """Removes repetitive sentences from generated text, keeping the first occurrence."""
    # Split by common sentence endings to handle multiple sentences
    sentences = re.split(r'([.!?])', text)
    
    # Group sentences with their punctuation
    phrases = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
               for i in range(0, len(sentences), 2)]

    unique_phrases = []
    seen = set()
    for phrase in phrases:
        cleaned_phrase = phrase.strip()
        if cleaned_phrase and cleaned_phrase not in seen:
            unique_phrases.append(cleaned_phrase)
            seen.add(cleaned_phrase)
            
    return ' '.join(unique_phrases).strip()

def _fix_incomplete_sentences(text: str) -> str:
    """Removes a trailing sentence fragment if it's incomplete."""
    text = text.strip()
    if not text or text.endswith((".", "!", "?")):
        return text
    
    # Find the last sentence-ending punctuation mark
    last_punc_pos = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    
    # If a punctuation mark exists and there's text after it, trim to that point
    if last_punc_pos != -1:
        return text[:last_punc_pos+1].strip()
        
    # If no punctuation exists at all, the text is a single fragment; return as is.
    return text

def _swap_tokens_back(text: str, post_cfg: Dict) -> str:
    """Replaces mapped names back to their original #Person# tokens."""
    token_swap_cfg = post_cfg.get("token_swapping", {"enable": False})
    if token_swap_cfg.get("enable"):
        token_map = token_swap_cfg.get("token_map", {})
        # Create a reverse mapping
        reverse_map = {v: k for k, v in token_map.items()}
        for replacement, original in reverse_map.items():
            text = text.replace(replacement, original)
    return text

def apply_post_processing(text: str, post_cfg: Dict) -> str:
    """
    Applies a full pipeline of post-processing steps based on a configuration dictionary.
    """
    # 1. Remove special tokens defined in the config
    remove_tokens = post_cfg.get("remove_tokens", [])
    for token in remove_tokens:
        text = text.replace(token, "")

    # 2. Basic text cleaning
    text_cleaning = post_cfg.get("text_cleaning", {})
    if text_cleaning.get("strip_whitespace", True):
        text = text.strip()
    if text_cleaning.get("normalize_whitespace", True):
        text = re.sub(r'\s+', ' ', text)
    
    # 3. Korean-specific cleaning
    korean_cfg = post_cfg.get("korean_specific", {})
    if korean_cfg.get("remove_special_markers", True):
        # This regex removes markers like #Address# but keeps #Person1#, #Person2#, etc.
        # BEFORE:
        # text = re.sub(r'#(?!Person\d+#)\w+#', '', text)

        # AFTER (✅ Use this simpler regex to remove ALL #word# tokens):
        text = re.sub(r'#\w+#', '', text)        
        text = ' '.join(text.split())

    # 4. Advanced cleaning using helper functions
    advanced_cfg = post_cfg.get("advanced", {})
    if advanced_cfg.get("remove_repetitive_phrases", True):
        text = _remove_repetitive_phrases(text)
    if advanced_cfg.get("fix_incomplete_sentences", True):
        text = _fix_incomplete_sentences(text)

    # ✅ ADD THIS AS THE FINAL STEP BEFORE THE LENGTH CHECK
    text = _swap_tokens_back(text, post_cfg)

    # 5. Final check for minimum length
    min_length = post_cfg.get("advanced", {}).get("min_length", 5)
    if len(text.strip()) < min_length:
        return "요약을 생성할 수 없습니다."
        
    return text.strip()