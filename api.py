"""
LENS NLP Suite — FastAPI Backend
Exposes 4 NLP models as REST API endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import os

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LENS NLP Suite API",
    description="REST API for Spam Detection, Language Detection, Food Sentiment Analysis, and News Classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename: str):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {filename}")
    return joblib.load(path)

try:
    spam_model     = load_model("spam_classifier.pkl")
    language_model = load_model("lang_det.pkl")
    news_model     = load_model("news_cat.pkl")
    review_model   = load_model("review.pkl")
except FileNotFoundError as e:
    raise RuntimeError(str(e))

# ── Request / Response Schemas ────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class BulkTextInput(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    input: str
    prediction: str

class BulkPredictionResponse(BaseModel):
    results: List[PredictionResponse]

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "✅ LENS NLP API is running",
        "version": "1.0.0",
        "models_loaded": ["spam_classifier", "language_detector", "news_categoriser", "review_sentiment"],
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. SPAM CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/spam/predict", response_model=PredictionResponse, tags=["Spam Classifier"])
def predict_spam(body: TextInput):
    """
    Classify a single message as **Spam** or **Not Spam**.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    pred = spam_model.predict([body.text])[0]
    label = "Spam" if pred == 0 else "Not Spam"
    return PredictionResponse(input=body.text, prediction=label)


@app.post("/spam/predict/bulk", response_model=BulkPredictionResponse, tags=["Spam Classifier"])
def predict_spam_bulk(body: BulkTextInput):
    """
    Classify a **list of messages** as Spam or Not Spam.
    """
    if not body.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")
    preds = spam_model.predict(body.texts)
    results = [
        PredictionResponse(input=t, prediction="Spam" if p == 0 else "Not Spam")
        for t, p in zip(body.texts, preds)
    ]
    return BulkPredictionResponse(results=results)


# ══════════════════════════════════════════════════════════════════════════════
# 2. LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/language/predict", response_model=PredictionResponse, tags=["Language Detection"])
def predict_language(body: TextInput):
    """
    Detect the **language** of the given text.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    pred = language_model.predict([body.text])[0]
    return PredictionResponse(input=body.text, prediction=str(pred))


@app.post("/language/predict/bulk", response_model=BulkPredictionResponse, tags=["Language Detection"])
def predict_language_bulk(body: BulkTextInput):
    """
    Detect languages for a **list of texts**.
    """
    if not body.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")
    preds = language_model.predict(body.texts)
    results = [
        PredictionResponse(input=t, prediction=str(p))
        for t, p in zip(body.texts, preds)
    ]
    return BulkPredictionResponse(results=results)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FOOD REVIEW SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/sentiment/predict", response_model=PredictionResponse, tags=["Food Review Sentiment"])
def predict_sentiment(body: TextInput):
    """
    Classify a food review as **Positive** or **Negative**.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    pred = review_model.predict([body.text])[0]
    label = "Positive" if str(pred).lower() in ["1", "positive", "pos"] else "Negative"
    return PredictionResponse(input=body.text, prediction=label)


@app.post("/sentiment/predict/bulk", response_model=BulkPredictionResponse, tags=["Food Review Sentiment"])
def predict_sentiment_bulk(body: BulkTextInput):
    """
    Classify a **list of food reviews** as Positive or Negative.
    """
    if not body.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")
    preds = review_model.predict(body.texts)
    results = [
        PredictionResponse(
            input=t,
            prediction="Positive" if str(p).lower() in ["1", "positive", "pos"] else "Negative"
        )
        for t, p in zip(body.texts, preds)
    ]
    return BulkPredictionResponse(results=results)


# ══════════════════════════════════════════════════════════════════════════════
# 4. NEWS CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/news/predict", response_model=PredictionResponse, tags=["News Classification"])
def predict_news(body: TextInput):
    """
    Categorise a news headline or article into a **topic category**.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    pred = news_model.predict([body.text])[0]
    return PredictionResponse(input=body.text, prediction=str(pred))


@app.post("/news/predict/bulk", response_model=BulkPredictionResponse, tags=["News Classification"])
def predict_news_bulk(body: BulkTextInput):
    """
    Categorise a **list of news texts** into topic categories.
    """
    if not body.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")
    preds = news_model.predict(body.texts)
    results = [
        PredictionResponse(input=t, prediction=str(p))
        for t, p in zip(body.texts, preds)
    ]
    return BulkPredictionResponse(results=results)
