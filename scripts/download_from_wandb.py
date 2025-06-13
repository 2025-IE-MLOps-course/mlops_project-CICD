import os
import shutil
from pathlib import Path
import wandb


def download_artifacts(dest_dir: str | Path = "models") -> None:
    """Download latest model and preprocessing pipeline from Weights & Biases.

    Authentication uses the WANDB_API_KEY environment variable. The project and
    entity are read from WANDB_PROJECT and WANDB_ENTITY.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    model_file = dest / "model.pkl"
    pipeline_file = dest / "preprocessing_pipeline.pkl"
    if model_file.exists() and pipeline_file.exists():
        print("Model artifacts already exist. Skipping download.")
        return

    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")
    api_key = os.environ.get("WANDB_API_KEY")
    if not all([project, entity, api_key]):
        raise EnvironmentError(
            "WANDB_PROJECT, WANDB_ENTITY, and WANDB_API_KEY must be set"
        )

    wandb.login(key=api_key)
    run = wandb.init(project=project, entity=entity, job_type="fetch-model", reinit=True)

    model_art = run.use_artifact("model:latest")
    model_dir = Path(model_art.download())
    for name in ["model.pkl", "preprocessing_pipeline.pkl"]:
        src = model_dir / name
        if src.exists():
            shutil.copy(src, dest / name)

    # Also try explicit preprocessing_pipeline artifact if present
    try:
        pp_art = run.use_artifact("preprocessing_pipeline:latest")
        pp_dir = Path(pp_art.download())
        src = pp_dir / "preprocessing_pipeline.pkl"
        if src.exists():
            shutil.copy(src, pipeline_file)
    except wandb.errors.CommError:
        pass

    run.finish()


if __name__ == "__main__":
    download_artifacts()
