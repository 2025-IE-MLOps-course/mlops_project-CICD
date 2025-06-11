"""
inference/run.py

MLflow-compatible batch inference step with Hydra config and W&B logging.
Uses the trained model and preprocessing pipeline to generate predictions
for new data. Logs prediction artifact, input hash/schema, duration,
prediction probability stats, and sample table to W&B.
"""

import sys
import logging
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import copy

import hydra
import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import yaml
from data_validation.data_validator import validate_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference.inferencer import run_inference

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inference")


def df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for the inference MLflow step."""
    config_path = PROJECT_ROOT / "config.yaml"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"inference_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="inference",
            name=run_name,
            config=cfg_dict,
            tags=["inference"],
        )
        logger.info("Started WandB run: %s", run_name)

        input_path = PROJECT_ROOT / cfg.inference.input_csv
        output_path = PROJECT_ROOT / cfg.inference.output_csv

        # Prefer W&B artifact for input data to preserve lineage
        try:
            in_art = run.use_artifact("predictions_input:latest")
            art_dir = Path(in_art.download())
            csv_files = list(art_dir.glob("*.csv"))
            if csv_files:
                input_path = csv_files[0]
                logger.info("Using predictions_input artifact: %s", input_path)
        except Exception:
            logger.warning(
                "predictions_input artifact not found; falling back to %s",
                input_path,
            )

        # Log input data hash and schema, and validate before inference
        if input_path.is_file():
            in_df = pd.read_csv(input_path)
            # ----- Data validation -----
            val_cfg = copy.deepcopy(cfg_dict)
            val_report = (
                PROJECT_ROOT / "logs" / f"inference_validation_{run.id[:8]}.json"
            )
            val_cfg.setdefault("data_validation", {})["report_path"] = str(val_report)
            validate_data(in_df, config=val_cfg)
            if val_report.is_file():
                art = wandb.Artifact("inference_validation_report", type="report")
                art.add_file(str(val_report))
                run.log_artifact(art, aliases=["latest"])
                with open(val_report) as f:
                    rep = json.load(f)
                wandb.summary.update({
                    "inference_validation_result": rep.get("result", "unknown"),
                    "inference_validation_errors": len(rep.get("errors", [])),
                    "inference_validation_warnings": len(rep.get("warnings", [])),
                })

            wandb.summary["input_data_hash"] = df_hash(in_df)
            input_schema = {col: str(dtype) for col, dtype in in_df.dtypes.items()}
            wandb.summary["input_data_schema"] = input_schema

            schema_path = (
                PROJECT_ROOT / "artifacts" / f"infer_input_schema_{run.id[:8]}.json"
            )
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            with open(schema_path, "w") as f:
                json.dump(input_schema, f, indent=2)
            art = wandb.Artifact("inference_input_schema", type="schema")
            art.add_file(str(schema_path))
            run.log_artifact(art, aliases=["latest"])

            # Optionally log the full input CSV for lineage
            if cfg.data_load.get("log_artifacts", True):
                in_art = wandb.Artifact("predictions_input", type="predictions_input")
                in_art.add_file(str(input_path))
                run.log_artifact(in_art, aliases=["latest"])

        # Save resolved config for reproducibility
        temp_cfg = PROJECT_ROOT / "artifacts" / f"infer_cfg_{run.id[:8]}.yaml"
        temp_cfg.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_cfg, "w") as f:
            yaml.safe_dump(cfg_dict, f)

        # Track inference duration
        t0 = time.time()
        run_inference(str(input_path), str(temp_cfg), str(output_path), run=run)
        duration = time.time() - t0
        wandb.summary["inference_duration_seconds"] = duration

        # Log predictions as artifact and sample table
        if output_path.is_file():
            out_df = pd.read_csv(output_path)

            # Log the entire table (since it's small)
            wandb.log({"prediction_table": wandb.Table(dataframe=out_df)})

            wandb.summary["n_predictions"] = len(out_df)
            wandb.summary["prediction_columns"] = list(out_df.columns)

            if "prediction_proba" in out_df.columns:
                wandb.summary["prediction_proba_mean"] = float(
                    out_df["prediction_proba"].mean()
                )
                wandb.summary["prediction_proba_min"] = float(
                    out_df["prediction_proba"].min()
                )
                wandb.summary["prediction_proba_max"] = float(
                    out_df["prediction_proba"].max()
                )

            # Log predictions as artifact
            if cfg.data_load.get("log_artifacts", True):
                artifact = wandb.Artifact("predictions", type="predictions")
                artifact.add_file(str(output_path))
                run.log_artifact(artifact, aliases=["latest"])
                logger.info("Logged predictions artifact to WandB")

    except Exception as e:
        logger.exception("Failed during inference step")
        if run is not None:
            run.alert(title="Inference Step Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
