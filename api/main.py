"""
api/main.py  –  FastAPI service for the fine-tuned fake-news detector.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import torch, numpy as np, pathlib, logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── FastAPI app exists IMMEDIATELY ─────────────────────────────
app = FastAPI(title="Fake-News Detector", version="1.0")

# global vars that will be set on startup
tok = model = device = None
label_id2str = {0: "real", 1: "fake"}

# ── load model once at startup ─────────────────────────────────
MODEL_DIR = pathlib.Path(__file__).parent / "model" / "fakenews_roberta"

@app.on_event("startup")
def load_model():
    global tok, model, device
    if not MODEL_DIR.exists():
        logging.error("❌ Model directory %s not found", MODEL_DIR)
        raise RuntimeError(
            f"Model folder {MODEL_DIR} missing. "
            "Did Step 2 finish and save the model?"
        )

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device).eval()
    logging.info("✅ Loaded model from %s onto %s", MODEL_DIR, device)

# ── Pydantic schemas ───────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str


class BatchPredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    label: Literal["real", "fake"]
    prob: float

# ── helper ─────────────────────────────────────────────────────
def _predict_logits(texts: List[str]) -> np.ndarray:
    with torch.no_grad():
        inputs = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits.cpu().numpy()
    return logits

# ── routes ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, detail="text must be non-empty")
    logits = _predict_logits([req.text])[0]
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred = int(probs.argmax())
    return {"label": label_id2str[pred], "prob": round(float(probs[pred]), 4)}


@app.post("/batch_predict", response_model=List[PredictResponse])
def batch_predict(req: BatchPredictRequest):
    if not req.texts:
        raise HTTPException(400, detail="texts must contain ≥1 items")
    logits = _predict_logits(req.texts)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(-1)
    return [
        {"label": label_id2str[int(p)], "prob": round(float(probs[i, p]), 4)}
        for i, p in enumerate(preds)
    ]
