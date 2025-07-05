import os
import torch
from typing import Dict, Optional
from transformers import pipeline

class SentimentAnalyzer:
    """ML-based sentiment analyzer using DistilBERT or fine-tuned models."""
    
    def __init__(self) -> None:
        self.pipeline = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the ML model from local path or download DistilBERT."""
        model_path = os.getenv("MODEL_PATH", "./model")
        
        if os.path.exists(model_path) and os.listdir(model_path):
            source = model_path
        else:
            # Use a smaller, faster model for demo
            source = "distilbert-base-uncased-finetuned-sst-2-english"
        
        self.pipeline = pipeline(
            "text-classification",
            model=source,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True
        )
    
    def predict(self, text: str) -> Dict[str, float]:
        results = self.pipeline(text)
        scores = results[0] if isinstance(results[0], list) else results
        
        best = max(scores, key=lambda x: x['score'])
        label = self._map_label(best['label'])
        
        return {"label": label, "score": round(best['score'], 4)}
    
    def _map_label(self, label: str) -> str:
        if label.upper() in ['POSITIVE', 'LABEL_1', '1']:
            return "positive"
        return "negative"

_analyzer: Optional[SentimentAnalyzer] = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
