from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import uvicorn

app = FastAPI()
sentiment_pipeline = pipeline("sentiment-analysis")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/form", response_class=HTMLResponse)
async def handle_form(request: Request, text: str = Form(...)):
    result = sentiment_pipeline(text)[0]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": {
            "label": result['label'],
            "score": round(result['score'], 4)
        }
    })

@app.post("/predict")
async def predict_sentiment(payload: dict):
    text = payload.get("text", "")
    result = sentiment_pipeline(text)[0]
    return {"label": result['label'], "score": round(result['score'], 4)}
