from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import yaml
from pathlib import Path
import pandas as pd
from preprocess.preprocessing import get_output_feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh)

PIPELINE_PATH = PROJECT_ROOT / CONFIG.get("artifacts", {}).get(
    "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
)
MODEL_PATH = PROJECT_ROOT / CONFIG.get("artifacts", {}).get(
    "model_path", "models/model.pkl"
)

with PIPELINE_PATH.open("rb") as fh:
    PIPELINE = pickle.load(fh)

with MODEL_PATH.open("rb") as fh:
    MODEL = pickle.load(fh)

RAW_FEATURES = CONFIG.get("raw_features", [])
ENGINEERED = CONFIG.get("features", {}).get("engineered", [])

app = FastAPI()


class PredictionInput(BaseModel):
    rx_ds: int = Field(..., alias="rx ds")
    A: int
    B: int
    C: int
    D: int
    E: int
    F: int
    H: int
    I: int
    J: int
    K: int
    L: int
    M: int
    N: int
    R: int
    S: int
    T: int
    V: int

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "rx_ds": 100,
                "A": 0,
                "B": 1,
                "C": 0,
                "D": 0,
                "E": 1,
                "F": 0,
                "H": 0,
                "I": 0,
                "J": 1,
                "K": 0,
                "L": 0,
                "M": 0,
                "N": 1,
                "R": 0,
                "S": 0,
                "T": 0,
                "V": 0,
            }
        }


@app.get("/")
def root():
    return {"message": "Welcome to the opioid abuse prediction API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput):
    data = payload.dict(by_alias=True)
    df = pd.DataFrame([data])
    X_raw = df[RAW_FEATURES]
    X_proc = PIPELINE.transform(X_raw)
    if ENGINEERED:
        feature_names = get_output_feature_names(
            preprocessor=PIPELINE,
            input_features=RAW_FEATURES,
            config=CONFIG,
        )
        indices = [feature_names.index(f) for f in ENGINEERED if f in feature_names]
        X_proc = X_proc[:, indices]
    pred = MODEL.predict(X_proc)[0]
    proba = MODEL.predict_proba(X_proc)[0, 1] if hasattr(MODEL, "predict_proba") else None
    return {"prediction": int(pred), "probability": float(proba) if proba is not None else None}
