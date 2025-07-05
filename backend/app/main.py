import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import PredictionRequest, PredictionResponse
from .sentiment import get_sentiment_analyzer, enable_ml_model

app = FastAPI(title="Sentiment Analysis API", description="Tells you if text is positive or negative")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model when the app starts
@app.on_event("startup")
async def startup_event():
    print("Loading sentiment model...")
    get_sentiment_analyzer()
    print("Model loaded successfully!")

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    analyzer = get_sentiment_analyzer()
    result = analyzer.predict(request.text)
    
    return PredictionResponse(
        label=result["label"],
        score=result["score"]
    )

@app.post("/enable-ml-model")
async def enable_ml_model_endpoint():
    """Switch to using the ML model for better accuracy"""
    try:
        enable_ml_model()
        return {"message": "ML model enabled successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable ML model: {str(e)}")

@app.post("/debug-predict")
async def debug_predict(request: PredictionRequest):
    """Debug version that shows word counts"""
    analyzer = get_sentiment_analyzer()
    text_lower = request.text.lower()
    # Apply same word processing as in the actual prediction
    words = [re.sub(r'[^\w]', '', word) for word in text_lower.split()]
    words = [word for word in words if word]
    
    positive_count = sum(1 for word in words if word in analyzer.positive_words)
    negative_count = sum(1 for word in words if word in analyzer.negative_words)
    
    positive_words_found = [word for word in words if word in analyzer.positive_words]
    negative_words_found = [word for word in words if word in analyzer.negative_words]
    
    result = analyzer.predict(request.text)
    
    return {
        "text": request.text,
        "original_words": request.text.lower().split(),
        "processed_words": words,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_words_found": positive_words_found,
        "negative_words_found": negative_words_found,
        "prediction": result
    }

@app.get("/model-status")
async def get_model_status():
    """Check which model is currently running"""
    analyzer = get_sentiment_analyzer()
    return {
        "using_ml_model": analyzer.use_ml_model,
        "ml_model_loaded": analyzer.pipeline is not None,
        "model_type": "ML (DistilBERT)" if analyzer.use_ml_model else "Simple rules"
    }

@app.get("/health")
async def health_check():
    """Health check - just tells you if the API is alive"""
    return {"status": "healthy"}

