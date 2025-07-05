import os
from typing import Dict, Optional

class SentimentAnalyzer:
    """Simple rule-based sentiment analyzer using word lists."""
    
    def __init__(self) -> None:
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic', 
            'wonderful', 'perfect', 'love', 'like', 'best', 'brilliant',
            'outstanding', 'superb', 'marvelous', 'incredible', 'fabulous'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disgusting',
            'pathetic', 'useless', 'disappointing', 'poor', 'annoying', 'stupid',
            'ridiculous', 'frustrating', 'waste', 'regret'
        }
    
    def predict(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count > negative_count:
            return {"label": "positive", "score": 0.85}
        elif negative_count > positive_count:
            return {"label": "negative", "score": 0.85}
        else:
            # Default to positive with lower confidence
            return {"label": "positive", "score": 0.55}

_analyzer: Optional[SentimentAnalyzer] = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
