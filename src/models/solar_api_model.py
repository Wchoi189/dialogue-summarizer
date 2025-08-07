# src/models/solar_api_model.py
"""
Solar API model integration for dialogue summarization.
Use this as an ensemble member or fallback.
"""

import logging
import time
from typing import List, Optional
import requests
from omegaconf import DictConfig
from icecream import ic

logger = logging.getLogger(__name__)

class SolarAPIClient:
    """Client for Solar API dialogue summarization."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.api_cfg = cfg.get("solar_api", {})
        self.api_key = self.api_cfg.get("api_key")
        self.base_url = self.api_cfg.get("base_url", "https://api.upstage.ai/v1/solar")
        self.model = self.api_cfg.get("model", "solar-1-mini-chat")
        
        # Rate limiting
        self.requests_per_minute = self.api_cfg.get("rate_limit", 60)
        self.request_delay = 60.0 / self.requests_per_minute
        
        ic(f"SolarAPIClient initialized: model={self.model}")
    
    def create_prompt(self, dialogue: str, topic: Optional[str] = None) -> str:
        """Create optimized prompt for Solar API."""
        base_prompt = """다음 대화를 간결하고 정확하게 요약해주세요. 주요 내용과 결론을 포함하되, #Person1#, #Person2# 등의 화자 표시는 그대로 유지해주세요.

대화:
{dialogue}

요약:"""
        
        if topic:
            base_prompt = f"주제: {topic}\n\n" + base_prompt
            
        return base_prompt.format(dialogue=dialogue)
    
    def summarize_single(self, dialogue: str, topic: Optional[str] = None) -> str:
        """Summarize a single dialogue using Solar API."""
        prompt = self.create_prompt(dialogue, topic)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 150,
            "temperature": 0.1,  # Low temperature for consistency
            "top_p": 0.9
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            summary = result["choices"][0]["message"]["content"].strip()
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            return summary
            
        except Exception as e:
            logger.error(f"Solar API request failed: {e}")
            return "요약을 생성할 수 없습니다."
    
    def summarize_batch(self, dialogues: List[str], topics: Optional[List[Optional[str]]] = None) -> List[str]:
        """Summarize batch of dialogues with progress tracking."""
        summaries = []
        topics = topics or [None for _ in range(len(dialogues))]
        
        from tqdm import tqdm
        for i, (dialogue, topic) in enumerate(tqdm(zip(dialogues, topics), total=len(dialogues), desc="Solar API")):
            summary = self.summarize_single(dialogue, topic)
            summaries.append(summary)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                ic(f"Processed {i + 1}/{len(dialogues)} dialogues")
        
        return summaries


# Ensemble strategy
class EnsemblePredictor:
    """Ensemble KoBART + Solar API for better performance."""
    
    def __init__(self, kobart_model, solar_client, cfg: DictConfig):
        self.kobart_model = kobart_model
        self.solar_client = solar_client
        self.cfg = cfg
        self.ensemble_cfg = cfg.get("ensemble", {})
        
    def predict_ensemble(self, dialogues: List[str], topics: Optional[List[str]] = None) -> List[str]:
        """Generate ensemble predictions."""
        # Get KoBART predictions
        ic("Generating KoBART predictions...")
        kobart_preds = self.kobart_model.predict_batch(dialogues)
        
        # Get Solar API predictions for subset (due to cost)
        sample_size = min(len(dialogues), self.ensemble_cfg.get("solar_sample_size", 50))
        ic(f"Generating Solar predictions for {sample_size} samples...")
        
        solar_dialogues = dialogues[:sample_size]
        solar_topics = topics[:sample_size] if topics else None
        solar_preds = self.solar_client.summarize_batch(solar_dialogues, solar_topics)
        
        # Simple ensemble: use Solar for first N, KoBART for rest
        ensemble_preds = solar_preds + kobart_preds[sample_size:]
        
        ic(f"Ensemble complete: {len(ensemble_preds)} predictions")
        return ensemble_preds