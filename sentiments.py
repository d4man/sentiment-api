# sentiments.py
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def predict_sentiment(text: str) -> dict:
    result = sentiment_pipeline(text)[0]
    return {
        "label": result['label'],
        "score": round(result['score'], 5)
    }
