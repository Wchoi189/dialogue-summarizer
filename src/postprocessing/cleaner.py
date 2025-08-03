# FILE: src/postprocessing/cleaner.py
import re
from typing import Dict, List, Optional

class TextPostProcessor:
    """Centralized text post-processing with configurable options."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
    
    def process(self, text: str, stage: Optional[str] = None) -> str:
        """
        Main entry point for text post-processing.
        
        Args:
            text: Input text to process
            stage: Optional stage identifier for stage-specific config
            
        Returns:
            Processed text
        """
        if not text or not text.strip():
            return ""
        
        # Get configuration for this stage
        post_cfg = self._get_stage_config(stage)
        
        # Apply processing steps in order
        text = self._remove_tokens(text, post_cfg)
        text = self._clean_text(text, post_cfg)
        text = self._apply_korean_specific(text, post_cfg)
        text = self._apply_advanced_cleaning(text, post_cfg)
        text = self._validate_output(text, post_cfg)
        
        return text
    
    def _get_stage_config(self, stage: Optional[str]) -> Dict:
        """Get configuration for specific stage or default."""
        if not self.config:
            return self._get_fallback_config()
        
        if stage and stage in self.config:
            return self.config[stage]
        
        return self.config.get("default", self._get_fallback_config())
    
    def _get_fallback_config(self) -> Dict:
        """Fallback configuration when no config is provided."""
        return {
            "remove_tokens": ["<s>", "</s>", "<pad>", "<usr>"],
            "text_cleaning": {
                "strip_whitespace": True,
                "normalize_whitespace": True,
                "remove_empty_lines": True,
                "remove_repetitive_phrases": True
            },
            "advanced": {
                "fix_incomplete_sentences": True,
                "min_length": 5
            },
            "korean_specific": {
                "remove_special_markers": False
            }
        }
    
    def _remove_tokens(self, text: str, config: Dict) -> str:
        """Remove unwanted tokens from text."""
        remove_tokens = config.get("remove_tokens", [])
        
        # Always ensure <usr> is removed
        if "<usr>" not in remove_tokens:
            remove_tokens = list(remove_tokens) + ["<usr>"]
        
        for token in remove_tokens:
            text = text.replace(token, "")
        
        return text
    
    def _clean_text(self, text: str, config: Dict) -> str:
        """Apply basic text cleaning operations."""
        text_cleaning = config.get("text_cleaning", {})
        
        # Strip whitespace
        if text_cleaning.get("strip_whitespace", True):
            text = text.strip()
        
        # Normalize whitespace
        if text_cleaning.get("normalize_whitespace", True):
            text = re.sub(r'\s+', ' ', text)
        
        # Remove empty lines
        if text_cleaning.get("remove_empty_lines", True):
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = ' '.join(lines)
        
        # Remove repetitive phrases
        if text_cleaning.get("remove_repetitive_phrases", True):
            text = self._remove_repetitive_phrases(text)
        
        return text
    
    def _apply_korean_specific(self, text: str, config: Dict) -> str:
        """Apply Korean-specific cleaning rules."""
        korean_cfg = config.get("korean_specific", {})
        
        # Remove special markers (but preserve #Person# tokens)
        if korean_cfg.get("remove_special_markers", False):
            text = re.sub(r'#(?!Person\d+#)\w+#', '', text)
            text = ' '.join(text.split())
        
        return text
    
    def _apply_advanced_cleaning(self, text: str, config: Dict) -> str:
        """Apply advanced cleaning operations."""
        advanced_cfg = config.get("advanced", {})
        
        # Fix incomplete sentences
        if advanced_cfg.get("fix_incomplete_sentences", True):
            text = self._fix_incomplete_sentences(text, advanced_cfg)
        
        return text
    
    def _validate_output(self, text: str, config: Dict) -> str:
        """Validate final output and apply fallbacks."""
        advanced_cfg = config.get("advanced", {})
        min_length = advanced_cfg.get("min_length", 5)
        fallback_message = advanced_cfg.get("fallback_message", "요약을 생성할 수 없습니다.")
        
        if len(text.strip()) < min_length:
            return fallback_message
        
        return text.strip()
    
    def _remove_repetitive_phrases(self, text: str) -> str:
        """Remove repetitive sentences from generated text."""
        sentences = text.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        return '. '.join(unique_sentences).strip()
    
    def _fix_incomplete_sentences(self, text: str, advanced_cfg: Dict) -> str:
        """Fix incomplete sentences by removing the trailing incomplete one."""
        text = text.strip()
        if not text:
            return text
        
        # Check if text already ends properly
        if text.endswith((".", "!", "?")):
            return text
        
        # Remove the last incomplete sentence fragment
        sentences = re.split(r'[.!?]', text)
        if len(sentences) > 1:
            complete_sentences = sentences[:-1]
            if complete_sentences:
                # Reconstruct with original punctuation
                result = ""
                parts = re.split(r'([.!?])', text)
                for i in range(0, len(parts) - 2, 2):
                    result += parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
                return result.strip()
        
        return text


# Convenience functions for backward compatibility
def apply_comprehensive_post_processing(text: str, post_cfg: Dict) -> str:
    """
    Backward compatibility function.
    
    Args:
        text: Input text
        post_cfg: Post-processing configuration
        
    Returns:
        Processed text
    """
    processor = TextPostProcessor({"default": post_cfg})
    return processor.process(text)


def create_post_processor(config: Optional[Dict] = None) -> TextPostProcessor:
    """
    Factory function to create a post-processor instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        TextPostProcessor instance
    """
    return TextPostProcessor(config)


# Example usage and configuration
DEFAULT_CONFIG = {
    "training": {
        "remove_tokens": ["<s>", "</s>", "<pad>", "<usr>"],
        "text_cleaning": {
            "strip_whitespace": True,
            "normalize_whitespace": True,
            "remove_empty_lines": True,
            "remove_repetitive_phrases": True
        },
        "korean_specific": {
            "remove_special_markers": False
        },
        "advanced": {
            "fix_incomplete_sentences": True,
            "min_length": 5,
            "fallback_message": "요약을 생성할 수 없습니다."
        }
    },
    "validation": {
        "remove_tokens": ["<s>", "</s>", "<pad>", "<usr>"],
        "text_cleaning": {
            "strip_whitespace": True,
            "normalize_whitespace": True,
            "remove_empty_lines": True,
            "remove_repetitive_phrases": True
        },
        "korean_specific": {
            "remove_special_markers": False
        },
        "advanced": {
            "fix_incomplete_sentences": True,
            "min_length": 3,  # More lenient for validation
            "fallback_message": "검증 실패"
        }
    },
    "inference": {
        "remove_tokens": ["<s>", "</s>", "<pad>", "<usr>"],
        "text_cleaning": {
            "strip_whitespace": True,
            "normalize_whitespace": True,
            "remove_empty_lines": True,
            "remove_repetitive_phrases": True
        },
        "korean_specific": {
            "remove_special_markers": False
        },
        "advanced": {
            "fix_incomplete_sentences": True,
            "min_length": 5,
            "fallback_message": "요약을 생성할 수 없습니다."
        }
    }
}