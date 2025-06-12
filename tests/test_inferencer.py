"""
Comprehensive pytest suite for inferencer.py

- All file I/O uses pytest's tmp_path for temp isolation
- Pipelines are always fitted before saving
- Covers: happy path, engineered feature logic, missing features, logging, model without predict_proba, input errors
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression

from inference import inferencer

# ---- Utility functions for temp file I/O ----


def write_pickle(obj, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def write_yaml(obj, path):
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)


def write_csv(df, path):
    df.to_csv(path, index=False)

# ---- Fixtures for common objects ----


@pytest.fixture
def toy_df():
    """Minimal DataFrame for inference tests (2 features)."""
    return pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 2, 2, 1]})


@pytest.fixture
def toy_model():
    """Simple trained classifier (LogisticRegression)."""
    X = np.array([[0, 1], [1, 2], [0, 2], [1, 1]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)
    return model


@pytest.fixture
def toy_pipeline(toy_df):
    """Fitted pipeline for feature passthrough."""
    col_transform = ColumnTransformer(
        [('pass', FunctionTransformer(), ['f1', 'f2'])])
    pipe = Pipeline([('col_transform', col_transform)])
    pipe.fit(toy_df)  # Fit on toy data!
    return pipe


@pytest.fixture
def toy_config(tmp_path):
    """Minimal config dict for inference."""
    return {
        "artifacts": {
            "model_path": str(tmp_path / "model.pkl"),
            "preprocessing_pipeline": str(tmp_path / "preproc.pkl"),
        },
        "features": {"engineered": ["f1", "f2"]},
        "raw_features": ["f1", "f2"]
    }

# ---- Test Cases ----


def test_run_inference_happy_path(tmp_path, toy_config, toy_df, toy_model, toy_pipeline):
    """Full inference run writes expected predictions."""
    # Save pipeline and model artifacts
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    # Write config
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    # Write input CSV
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    # Output path
    output_path = tmp_path / "out.csv"
    # Run inference
    inferencer.run_inference(
        str(input_path), str(config_path), str(output_path))
    # Check output file created and has predictions
    df_pred = pd.read_csv(output_path)
    assert "prediction" in df_pred.columns
    assert len(df_pred) == len(toy_df)


def test_inference_invalid_model_path(tmp_path, toy_config, toy_df, toy_pipeline):
    """Fails with FileNotFoundError if model file is missing."""
    pp_path = tmp_path / "preproc.pkl"
    write_pickle(toy_pipeline, pp_path)
    # No model.pkl written!
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(tmp_path / "missing_model.pkl")
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    with pytest.raises(FileNotFoundError):
        inferencer.run_inference(
            str(input_path), str(config_path), str(output_path))


def test_engineered_features_subset(tmp_path, toy_df):
    """If engineered subset is configured, output is correct shape."""
    # Train model only on 'f1'
    X = toy_df[["f1"]].values
    y = [0, 1, 0, 1]
    model = LogisticRegression()
    model.fit(X, y)

    # Pipeline also only for 'f1'
    col_transform = ColumnTransformer(
        [('pass', FunctionTransformer(), ['f1'])])
    pipe = Pipeline([('col_transform', col_transform)])
    pipe.fit(toy_df)

    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(pipe, pp_path)
    write_pickle(model, model_path)
    config = {
        "raw_features": ["f1", "f2"],
        "features": {"engineered": ["f1"]},
        "artifacts": {
            "preprocessing_pipeline": str(pp_path),
            "model_path": str(model_path),
        }
    }
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    inferencer.run_inference(
        str(input_path), str(config_path), str(output_path))
    out_df = pd.read_csv(output_path)
    assert len(out_df) == len(toy_df)
    assert "prediction" in out_df.columns


def test_engineered_features_missing_exit(tmp_path, toy_df, toy_model, toy_pipeline):
    """If none of the engineered features are present, triggers sys.exit."""
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    config = {
        "raw_features": ["f1", "f2"],
        "features": {"engineered": ["something_not_present"]},
        "artifacts": {
            "preprocessing_pipeline": str(pp_path),
            "model_path": str(model_path),
        }
    }
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    with pytest.raises(SystemExit):
        inferencer.run_inference(
            str(input_path), str(config_path), str(output_path))


def test_run_inference_logs(monkeypatch, tmp_path, toy_config, toy_df, toy_model, toy_pipeline, caplog):
    """Logger emits info-level logs for all steps."""
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    with caplog.at_level("INFO"):
        inferencer.run_inference(
            str(input_path), str(config_path), str(output_path))
        # At least 3 main steps: loading, preprocessing, inference
        assert any(
            "Loading preprocessing pipeline" in msg for msg in caplog.messages)
        assert any(
            "Applying preprocessing pipeline" in msg for msg in caplog.messages)


def test_inference_handles_model_without_predict_proba(tmp_path, toy_config, toy_df):
    """Inference does not fail if model lacks predict_proba."""
    from sklearn.svm import LinearSVC
    X = toy_df[["f1", "f2"]].values
    y = np.array([0, 1, 0, 1])
    model = LinearSVC()
    model.fit(X, y)
    col_transform = ColumnTransformer(
        [('pass', FunctionTransformer(), ['f1', 'f2'])])
    pipe = Pipeline([('col_transform', col_transform)])
    pipe.fit(toy_df)
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(pipe, pp_path)
    write_pickle(model, model_path)
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    inferencer.run_inference(
        str(input_path), str(config_path), str(output_path))
    out_df = pd.read_csv(output_path)
    assert "prediction" in out_df.columns
    assert len(out_df) == len(toy_df)


def test_inference_fails_on_unfitted_pipeline(tmp_path, toy_config, toy_df, toy_model):
    """Fails with NotFittedError if pipeline was not fitted before pickling."""
    # This test ensures best practice: always fit pipeline before saving
    col_transform = ColumnTransformer(
        [('pass', FunctionTransformer(), ['f1', 'f2'])])
    pipe = Pipeline([('col_transform', col_transform)])  # Not fitted
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(pipe, pp_path)
    write_pickle(toy_model, model_path)
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        inferencer.run_inference(
            str(input_path), str(config_path), str(output_path))


def test_run_inference_df_happy_path(tmp_path, toy_config, toy_df, toy_model, toy_pipeline):
    """run_inference_df returns predictions without touching disk."""
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    out_df = inferencer.run_inference_df(toy_df, config)
    assert "prediction" in out_df.columns
    assert len(out_df) == len(toy_df)


def test_run_inference_df_missing_columns(tmp_path, toy_config, toy_df, toy_pipeline, toy_model):
    """ValueError if required raw features are missing."""
    pp_path = tmp_path / "pp.pkl"
    model_path = tmp_path / "m.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    bad_df = toy_df.drop(columns=["f2"])
    with pytest.raises(ValueError):
        inferencer.run_inference_df(bad_df, config)


def test_run_inference_df_engineered_subset(tmp_path, toy_df):
    """Subset of engineered features is respected."""
    X = toy_df[["f1"]].values
    y = [0, 1, 0, 1]
    model = LogisticRegression()
    model.fit(X, y)
    col_transform = ColumnTransformer(
        [('pass', FunctionTransformer(), ['f1'])])
    pipe = Pipeline([('col_transform', col_transform)])
    pipe.fit(toy_df)
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(pipe, pp_path)
    write_pickle(model, model_path)
    config = {
        "raw_features": ["f1", "f2"],
        "features": {"engineered": ["f1"]},
        "artifacts": {
            "preprocessing_pipeline": str(pp_path),
            "model_path": str(model_path),
        }
    }
    out_df = inferencer.run_inference_df(toy_df, config)
    assert len(out_df) == len(toy_df)
    assert "prediction" in out_df.columns


def test_run_inference_df_engineered_features_missing(tmp_path, toy_df, toy_model, toy_pipeline):
    """Raise if configured engineered features are absent after preprocessing."""
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    config = {
        "raw_features": ["f1", "f2"],
        "features": {"engineered": ["not_present"]},
        "artifacts": {
            "preprocessing_pipeline": str(pp_path),
            "model_path": str(model_path),
        }
    }
    with pytest.raises(ValueError):
        inferencer.run_inference_df(toy_df, config)
