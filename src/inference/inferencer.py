"""
inferencer.py

Batch inference entry point.

Usage
-----
python -m src.inference.inferencer `
    data/inference/new_data.csv config.yaml data/inference/output_predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import os
from pathlib import Path
import tempfile

import pandas as pd
import yaml

from preprocess.preprocessing import get_output_feature_names


logger = logging.getLogger(__name__)

# Resolve project root to allow relative artifact paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# helper to load pickled artefacts
def _load_pickle(path: str, label: str):
    """Safely load a pickled artefact, with a descriptive error if missing"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with p.open("rb") as fh:
        return pickle.load(fh)


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def run_inference(
    input_csv: str, config_yaml: str, output_csv: str, run: "wandb.sdk.wandb_run.Run" | None = None
) -> None:
    """
    Run batch inference:
    1. Load config, preprocessing pipeline, and trained model
    2. Validate that required **raw_features** exist in the input CSV
    3. Transform features via the pipeline
    4. Optionally keep only the engineered subset used during training
    5. Generate predictions and save to CSV
    """
    _setup_logging()

    # ── 1. Load config and artefacts ──────────────────────────────────────
    with open(config_yaml, "r", encoding="utf-8") as fh:
        config: Dict = yaml.safe_load(fh)

    if run is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_art = run.use_artifact("model:latest")
            model_dir = model_art.download(root=tmp_dir)
            model = _load_pickle(
                os.path.join(model_dir, "model.pkl"),
                "model",
            )

            pp_art = run.use_artifact("preprocessing_pipeline:latest")
            pp_dir = pp_art.download(root=tmp_dir)
            pipeline = _load_pickle(
                os.path.join(pp_dir, "preprocessing_pipeline.pkl"),
                "preprocessing pipeline",
            )
    else:
        pp_path = _resolve(
            config.get("artifacts", {}).get(
                "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
            )
        )
        model_path = _resolve(
            config.get("artifacts", {}).get("model_path", "models/model.pkl")
        )

        logger.info("Loading preprocessing pipeline: %s", pp_path)
        pipeline = _load_pickle(str(pp_path), "preprocessing pipeline")

        logger.info("Loading trained model: %s", model_path)
        model = _load_pickle(str(model_path), "model")

    # ── 2. Read raw data and basic validation ─────────────────────────────
    logger.info("Reading input CSV: %s", input_csv)
    input_df: pd.DataFrame = pd.read_csv(input_csv)
    logger.info("Input shape: %s", input_df.shape)

    raw_features: List[str] = config.get("raw_features", [])
    missing = [c for c in raw_features if c not in input_df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)

    X_raw = input_df[raw_features]

    # ── 3. Transform via the *same* preprocessing pipeline ────────────────
    logger.info("Applying preprocessing pipeline to input data")
    X_proc = pipeline.transform(X_raw)

    # ── 4. Keep only engineered features that were used in training ───────
    engineered = config.get("features", {}).get("engineered", [])
    if engineered:
        feature_names = get_output_feature_names(
            preprocessor=pipeline,
            input_features=raw_features,
            config=config,
        )
        selected = [f for f in engineered if f in feature_names]
        if not selected:
            logger.error(
                "None of the engineered features are present after transform"
            )
            sys.exit(1)
        indices = [feature_names.index(f) for f in selected]
        X_proc = X_proc[:, indices]

    # ── 5. Generate predictions ───────────────────────────────────────────
    logger.info("Generating predictions")
    input_df["prediction"] = model.predict(X_proc)
    if hasattr(model, "predict_proba"):
        input_df["prediction_proba"] = model.predict_proba(X_proc)[:, 1]

    # ── 6. Save results ───────────────────────────────────────────────────
    logger.info("Writing predictions to %s", output_csv)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    input_df.to_csv(output_csv, index=False)
    logger.info("Inference complete")


# CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch inference on a CSV file")
    parser.add_argument("input_csv", help="Path to raw input CSV")
    parser.add_argument("config_yaml", help="Path to config.yaml")
    parser.add_argument("output_csv", help="Destination for predictions CSV")
    args = parser.parse_args()

    run_inference(args.input_csv, args.config_yaml, args.output_csv)


def run_inference_df(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Run inference on an in-memory DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input data containing the columns listed under ``raw_features`` in
        ``config``.
    config : Dict
        Parsed YAML configuration providing artifact paths and feature lists.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with ``prediction`` and ``prediction_proba`` columns
        appended.

    Notes
    -----
    This helper mirrors :func:`run_inference` but avoids any file I/O. It is
    intended for use from API endpoints where data are already loaded in memory
    and predictions should be returned directly.
    """

    pp_path = _resolve(
        config.get("artifacts", {}).get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
        )
    )
    model_path = _resolve(
        config.get("artifacts", {}).get("model_path", "models/model.pkl")
    )

    pipeline = _load_pickle(str(pp_path), "preprocessing pipeline")
    model = _load_pickle(str(model_path), "model")

    raw_features: list[str] = config.get("raw_features", [])
    missing = [c for c in raw_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X_raw = df[raw_features]
    X_proc = pipeline.transform(X_raw)

    engineered = config.get("features", {}).get("engineered", [])
    if engineered:
        feature_names = get_output_feature_names(
            preprocessor=pipeline,
            input_features=raw_features,
            config=config,
        )
        selected = [f for f in engineered if f in feature_names]
        if not selected:
            raise ValueError(
                "None of the engineered features are present after transform"
            )
        indices = [feature_names.index(f) for f in selected]
        X_proc = X_proc[:, indices]

    result_df = df.copy()
    result_df["prediction"] = model.predict(X_proc)
    if hasattr(model, "predict_proba"):
        result_df["prediction_proba"] = model.predict_proba(X_proc)[:, 1]

    return result_df


if __name__ == "__main__":
    main()

