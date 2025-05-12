# app.py

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Sentiment Analysis with SHAP Explainability")

# Load model + tokenizer
MODEL_PATH = "./fine_tuned_tweet_eval"
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(
    torch.device("cuda" if device == 0 else "cpu")
)
pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=device,
)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(req: TextRequest):
    try:
        result = pipe(req.text)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/global")
async def explain_global():
    """
    Returns the global SHAP summary bar chart as PNG.
    """
    file_path = "outputs/shap/global_summary_bar.png"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Global SHAP explanation not found")
    return FileResponse(file_path, media_type="image/png")
