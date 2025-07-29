"""
api/main.py – FastAPI service for the fine‑tuned fake‑news detector.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from uuid import uuid4
from collections import deque
import torch, numpy as np, pathlib, logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ───────────────────────────────────────────────────────────────
app = FastAPI(title="Fake‑News Detector", version="1.2")

MODEL_DIR = pathlib.Path(__file__).parent / "model" / "fakenews_roberta"
label_id2str = {0: "real", 1: "fake"}

tok = model = device = None

# keep last 1 000 requests for live‑stats & explanations
# each item:  {id, text, pred, prob}
_history: deque[dict] = deque(maxlen=1_000)

# ───────────────────────────── model init
@app.on_event("startup")
def load_model():
    global tok, model, device
    if not MODEL_DIR.exists():
        logging.error("❌ Model directory %s not found", MODEL_DIR)
        raise RuntimeError(
            f"Model folder {MODEL_DIR} missing. "
            "Did Step 2 finish and save the model?"
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


# ───────────────────────────── Pydantic
class PredictRequest(BaseModel):
    text: str


class BatchPredictRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    id: str
    label: Literal["real", "fake"]
    prob: float


class ExplainResponse(BaseModel):
    words: List[str]
    contrib: List[float]  # signed weights


# ───────────────────────────── helpers
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


def _explain(text: str, k: int = 8) -> tuple[list[str], list[float]]:
    """
    Cheap explanation: take token logit‑difference between classes.
    Positive → pushes towards 'real', negative towards 'fake'.
    """
    tokens = tok.tokenize(text)[: k * 4]  # avoid super‑long texts
    if not tokens:
        return [], []

    pieces = [tok.convert_tokens_to_string([t]) for t in tokens]
    scores = []
    for piece in pieces:
        masked_text = text.replace(piece, tok.mask_token, 1)
        logits = _predict_logits([masked_text])[0]
        diff = float(logits[0] - logits[1])  # real – fake
        scores.append(diff)

    # pick top‑k by absolute contribution
    tops = sorted(zip(pieces, scores), key=lambda t: -abs(t[1]))[:k]
    words, contrib = map(list, zip(*tops))
    return words, contrib


# ───────────────────────────── routes
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, detail="text must be non‑empty")

    logits = _predict_logits([req.text])[0]
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred = int(probs.argmax())

    _id = uuid4().hex
    _history.append(
        {
            "id": _id,
            "text": req.text,
            "pred": pred,                     # ← store the *class*, not just prob
            "prob": float(probs[pred]),
        }
    )

    return {
        "id": _id,
        "label": label_id2str[pred],
        "prob": round(float(probs[pred]), 4),
    }


@app.post("/batch_predict", response_model=List[PredictResponse])
def batch_predict(req: BatchPredictRequest):
    if not req.texts:
        raise HTTPException(400, detail="texts must contain ≥1 items")

    logits = _predict_logits(req.texts)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(-1)

    out = []
    for text, pred, prob_vec in zip(req.texts, preds, probs):
        _id = uuid4().hex
        _history.append(
            {
                "id": _id,
                "text": text,
                "pred": int(pred),
                "prob": float(prob_vec[pred]),
            }
        )
        out.append(
            {
                "id": _id,
                "label": label_id2str[int(pred)],
                "prob": round(float(prob_vec[pred]), 4),
            }
        )
    return out


@app.get("/stats")
def stats():
    """
    Returns live statistics for the last 1 000 requests:
      { n, real, fake, avg }
    """
    n = len(_history)
    if n == 0:
        return {"n": 0}

    preds = np.array([h["pred"] for h in _history])
    probs = np.array([h["prob"] for h in _history])

    real = int((preds == 0).sum())
    fake = n - real
    return {
        "n": n,
        "real": real,
        "fake": fake,
        "avg": round(float(probs.mean()), 4),
    }


@app.get("/explain/{item_id}", response_model=ExplainResponse)
def explain(item_id: str):
    hit = next((h for h in _history if h["id"] == item_id), None)
    if not hit:
        raise HTTPException(404, "id not found (too old?)")

    words, contrib = _explain(hit["text"])
    return {"words": words, "contrib": contrib}
