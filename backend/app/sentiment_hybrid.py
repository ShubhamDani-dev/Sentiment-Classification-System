import os
import torch
from typing import Dict, Optional
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, use_ml_model: bool = False):
        self.use_ml_model = use_ml_model
        self.pipeline = None
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic', 
            'wonderful', 'perfect', 'love', 'like', 'best', 'brilliant',
            'outstanding', 'superb', 'marvelous', 'incredible', 'fabulous',
            'happy', 'joy', 'pleased', 'satisfied', 'delighted'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disgusting',
            'pathetic', 'useless', 'disappointing', 'poor', 'annoying', 'stupid',
            'ridiculous', 'frustrating', 'waste', 'regret', 'angry', 'sad',
            'upset', 'disappointed', 'broken', 'failed'
        }
        
        if self.use_ml_model:
            self._load_ml_model()
    
    def _load_ml_model(self):
        """Load the ML model (DistilBERT or fine-tuned model)"""
        try:
            model_path = os.getenv("MODEL_PATH", "./model")
            
            if os.path.exists(model_path) and os.listdir(model_path):
                logger.info(f"Loading fine-tuned model from {model_path}")
                source = model_path
            else:
                logger.info("Loading DistilBERT sentiment model")
                source = "distilbert-base-uncased-finetuned-sst-2-english"
            
            self.pipeline = pipeline(
                "text-classification",
                model=source,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            logger.info("ML model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            logger.info("Falling back to rule-based sentiment analysis")
            self.use_ml_model = False
    
    def predict(self, text: str) -> Dict[str, float]:
        if self.use_ml_model and self.pipeline:
            return self._predict_ml(text)
        else:
            return self._predict_rules(text)
    
    def _predict_ml(self, text: str) -> Dict[str, float]:
        """Use ML model for prediction"""
        try:
            results = self.pipeline(text)
            scores = results[0] if isinstance(results[0], list) else results
            
            best = max(scores, key=lambda x: x['score'])
            label = self._map_ml_label(best['label'])
            
            return {"label": label, "score": round(best['score'], 4)}
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._predict_rules(text)
    
    def _predict_rules(self, text: str) -> Dict[str, float]:
        """Use rule-based prediction"""
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if positive_count > negative_count:
            # More confident if more sentiment words found
            confidence = min(0.95, 0.7 + (total_sentiment_words * 0.1))
            return {"label": "positive", "score": round(confidence, 2)}
        elif negative_count > positive_count:
            confidence = min(0.95, 0.7 + (total_sentiment_words * 0.1))
            return {"label": "negative", "score": round(confidence, 2)}
        else:
            # Neutral case - slight positive bias
            return {"label": "positive", "score": 0.55}
    
    def _map_ml_label(self, label: str) -> str:
        """Map ML model labels to our standard format"""
        if label.upper() in ['POSITIVE', 'LABEL_1', '1']:
            return "positive"
        return "negative"
    
    def switch_to_ml_model(self):
        """Switch to ML model if not already using it"""
        if not self.use_ml_model:
            self.use_ml_model = True
            self._load_ml_model()

_analyzer: Optional[SentimentAnalyzer] = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        # Start with rule-based for fast startup
        use_ml = os.getenv("USE_ML_MODEL", "false").lower() == "true"
        _analyzer = SentimentAnalyzer(use_ml_model=use_ml)
    return _analyzer

def enable_ml_model():
    """Enable ML model for better accuracy"""
    analyzer = get_sentiment_analyzer()
    analyzer.switch_to_ml_model()
