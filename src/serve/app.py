import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "models/final")
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    state["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    state["device"] = device
    # warm up CUDA kernels so the first real request isn't slow
    warm = state["tok"]("warmup", return_tensors="pt").to(device)
    with torch.inference_mode():
        state["model"](**warm)
    yield
    state.clear()


app = FastAPI(title="Ticket Classifier", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    needs_review: bool
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "loading",
            "device": state.get("device"), "model_dir": MODEL_DIR}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    tok, model, device = state["tok"], state["model"], state["device"]

    inputs = tok(req.text, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

    idx = int(probs.argmax())
    conf = float(probs[idx])
    return PredictResponse(
        label=model.config.id2label[idx],
        confidence=round(conf, 4),
        needs_review=conf < THRESHOLD,
        latency_ms=round((time.perf_counter() - start) * 1000, 2))


Instrumentator().instrument(app).expose(app)
