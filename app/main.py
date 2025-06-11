from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Ensure src/ is on the Python path to import preprocessing utilities
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocess.preprocessing import get_output_feature_names

app = FastAPI(title="Opioid Abuse Prediction API")

# Load configuration and artifacts once at startup
with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG: Dict = yaml.safe_load(f)

RAW_FEATURES = CONFIG.get("raw_features", [])
ENGINEERED = CONFIG.get("features", {}).get("engineered", [])
ARTIFACTS = CONFIG.get("artifacts", {})

MODEL_PATH = PROJECT_ROOT / ARTIFACTS.get("model_path", "models/model.pkl")
PREPROC_PATH = PROJECT_ROOT / ARTIFACTS.get(
    "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
)

with open(PREPROC_PATH, "rb") as f:
    PIPELINE = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

FEATURE_NAMES = get_output_feature_names(
    preprocessor=PIPELINE, input_features=RAW_FEATURES, config=CONFIG
)
SELECTED_INDICES = [FEATURE_NAMES.index(f) for f in ENGINEERED if f in FEATURE_NAMES]


class PredictionRequest(BaseModel):
    """Input schema for model inference."""

    rx_ds: int = Field(..., example=330)
    A: int = Field(..., example=0)
    B: int = Field(..., example=0)
    C: int = Field(..., example=0)
    D: int = Field(..., example=1)
    E: int = Field(..., example=1)
    F: int = Field(..., example=1)
    H: int = Field(..., example=0)
    I: int = Field(..., example=1)
    J: int = Field(..., example=1)
    K: int = Field(..., example=0)
    L: int = Field(..., example=1)
    M: int = Field(..., example=1)
    N: int = Field(..., example=1)
    R: int = Field(..., example=0)
    S: int = Field(..., example=0)
    T: int = Field(..., example=0)
    V: int = Field(..., example=0)


@app.get("/")
async def read_root() -> Dict[str, str]:
    """Simple greeting for sanity check."""
    return {"message": "Welcome to the opioid abuse prediction API"}


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint used in tests."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: PredictionRequest) -> Dict[str, Optional[float]]:
    """Return model prediction for a single sample."""
    data = request.dict()
    df = pd.DataFrame([data])
    df.rename(columns={"rx_ds": "rx ds"}, inplace=True)
    X_raw = df[RAW_FEATURES]
    X_proc = PIPELINE.transform(X_raw)
    if SELECTED_INDICES:
        X_proc = X_proc[:, SELECTED_INDICES]
    pred = MODEL.predict(X_proc)[0]
    proba: Optional[float] = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(X_proc)[:, 1][0])
    return {"prediction": float(pred), "prediction_proba": proba}

