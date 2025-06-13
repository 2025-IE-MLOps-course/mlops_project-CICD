
# MLOps Pipeline for Opioid Abuse Disorder Prediction


This repository provides a modular, **production-quality** MLOps pipeline for binary classification of opioid abuse disorder, built as part of an academic project in the fundamentals of MLOps. The codebase is designed to bridge the gap between research prototypes (Jupyter notebooks) and scalable, maintainable machine learning systems in production.

---

## 🚦 Project Status

**Phase 1: Modularization, Testing, and Best Practices**
- Jupyter notebook translated into well-documented, test-driven Python modules
 - End-to-end pipeline: data ingestion, validation, model training (with integrated feature engineering and preprocessing), evaluation, and batch inference
- Robust configuration via `config.yaml` and reproducibility through explicit artifact management
- Extensive unit testing with pytest
- Strict adherence to software engineering and MLOps best practices

**Phase 2: Hydra, MLflow, and W&B Integration**
- All pipeline steps now execute as MLflow runs
- Dynamic configuration managed with Hydra
- Metrics and artifacts automatically logged to Weights & Biases

**Phase 3: CI/CD and FastAPI Serving**
- GitHub Actions workflow runs tests for Python 3.10 and 3.11
- FastAPI application (`app/main.py`) exposes prediction and health endpoints
  and now includes a `/predict_batch` route for validating and scoring multiple
  records in one request

---

## 📁 Repository Structure

```text
.
├── README.md
├── config.yaml                  # Central pipeline configuration
├── environment.yml              # Reproducible conda environment
├── data/                        # All project data (raw, splits, processed, inference)
├── models/                      # Serialized models, metrics, and preprocessing artifacts
├── logs/                        # Logging and validation artifacts
├── notebooks/                   # Source Jupyter notebook
├── app/                         # FastAPI application for online serving
├── .github/workflows/           # GitHub Actions CI pipeline
├── src/
│   ├── data_load/               # Data ingestion utilities
│   ├── data_validation/         # Config-driven schema and data validation
│   ├── evaluation/              # Model evaluation (metrics, confusion matrix, etc.)
│   ├── inference/               # Batch inference script
│   ├── model/                   # Training step with feature engineering & preprocessing
│   ├── features/                # Feature engineering utilities (used within model step)
│   └── preprocess/              # Preprocessing utilities (used within model step)
├── tests/                       # Unit and integration tests (pytest)
```

Artifact paths in `config.yaml` such as `data/splits`, `data/processed`, and
`models/` are resolved relative to this project root. Generated metrics,
preprocessing pipelines, and trained models will be saved under these
directories.

---

## 🔬 Problem Description

The pipeline predicts **opioid abuse disorder** based on anonymized claims data, engineered features, and diagnostic group flags. It applies rigorous validation and modular design suitable for research, teaching, and real-world deployment.

### Data Dictionary

| Feature        | Description                                                               |
|----------------|---------------------------------------------------------------------------|
| OD             | Opioid abuse disorder flag (target)                                       |
| Low_inc        | Low income indicator                                                      |
| Surgery        | Major surgery in 2 years                                                  |
| rx ds          | Days supply of opioid drugs over 2 years                                  |
| A-V            | ICD-10 disease chapter flags (see full mapping in the code base)          |

*Full dictionary in project root.*

---

## 🛠️ Pipeline Modules

### 0. Pipeline Orchestration (`src/main.py`)
- Single entry point that orchestrates the entire MLOps workflow
- Supports configurable pipeline stages: data validation, model training, and optional batch inference
- Integrates all modular components with robust logging and error handling
- Enables reproducible, scriptable runs for both research and production

### 1. Data Loading (`src/data_load/data_loader.py`)
- Loads data from CSV/Excel as specified in `config.yaml`
- Loads secrets from `.env` (for secure environments)
- Robust error handling and logging

### 2. Data Validation (`src/data_validation/data_validator.py`)
- Validates schema, column types, ranges, missing values
- Configurable strictness: `raise` or `warn` on errors
- Outputs detailed validation reports (JSON) and logs an HTML summary to W&B

### 3. Model Training, Feature Engineering & Preprocessing (`src/model/model.py`)
- Performs all feature engineering and preprocessing within the training step
- Loads the `validated_data` artifact from the previous step
- Data split (train/valid/test) with stratification
- Model registry (easily swap DecisionTree, LogisticRegression, RandomForest)
- Evaluation: accuracy, precision, recall, specificity, F1, ROC AUC, etc.
- Logs raw and processed splits to W&B and saves metrics, models, and preprocessing pipeline as artifacts

### 4. Evaluation (`src/evaluation/run.py`)
- Loads the `model` and `processed_data` artifacts
- Reconstructs processed train/valid/test splits for metrics
- Logs dataset hash, schema, and sample rows to W&B
- Generates confusion matrix, ROC and PR curves when applicable
- Saves metrics JSON as an artifact

### 5. Batch Inference (`src/inference/inferencer.py`)
- Loads preprocessing and model artifacts
- Transforms new data, predicts outcomes, exports CSV
- Provides `run_inference_df` helper used by the `/predict_batch` API endpoint
  to apply the same preprocessing and validation in memory

### 6. Unit Testing (`tests/`)
- Full pytest suite for each module
- Sample/mock data provided for CI/CD

---

## ⚙️ Configuration and Reproducibility

- **config.yaml**: All settings for data, model, features, metrics, and paths
- **environment.yml**: Reproducible Conda environment (Python, scikit-learn, pandas, PyYAML, etc.)
- **Artifacts**: All intermediate and final artifacts (preprocessing, models, metrics) are versioned and stored

---

## 🚀 Quickstart

**Environment setup:**
```bash
conda env create -f environment.yml
conda activate mlops_project
./setup.sh  # install all dependencies for testing
dvc pull     # download project data
wandb login  # authenticate with Weights & Biases
cp .env.example .env  # create local credentials file
# Edit `.env` and set WANDB_PROJECT and WANDB_ENTITY
```

**Run end-to-end pipeline:**
```bash
# run via python
python main.py main.steps=all
# or using MLflow
mlflow run . -P steps=all
# run the model step with Hydra overrides
python src/model/run.py model.decision_tree.params.max_depth=6 model.decision_tree.params.min_samples_split=3
mlflow run src/model -P hydra_options='model.decision_tree.params.max_depth=6 model.decision_tree.params.min_samples_split=3'
```
To override hyperparameters in the `model` step, pass them through the
`hydra_options` parameter when launching the run with MLflow. For example:

```bash
mlflow run src/model -P hydra_options="model.decision_tree.params.max_depth=8 model.decision_tree.params.min_samples_split=3"
```
All steps log metrics and artifacts to W&B by default.

Include `inference` in `main.steps` to generate predictions. For training only,
omit it (e.g., `main.steps=data_load,data_validation,model,evaluation`).

**Run inference from any server (after cloning repo and installing dependencies):**
```bash
python -m src.main --stage infer --config config.yaml --input_csv data/inference/new_data.csv --output_csv data/inference/output_predictions.csv
```

**Serve the model via FastAPI:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Call the running API (batch example):**
```bash
python scripts/call_api.py --url http://localhost:8000/predict_batch --input data/inference/new_data.csv
```
The `/predict` endpoint remains available for single-record requests.

**Run tests:**
```bash
pytest
```

## 🐳 Docker Deployment

Build the Docker image and run the API locally:
```bash
docker build -t opioid-api .
docker run -p 8000:8000 opioid-api
```
Before building or running, set the following environment variables so the
container can download the latest model from Weights & Biases:

```bash
export WANDB_PROJECT=<your-project>
export WANDB_ENTITY=<your-entity>
export WANDB_API_KEY=<your-api-key>
```

The server respects the `PORT` environment variable (default `8000`),
making it compatible with platforms such as Render. See `render.yaml`
for a minimal deployment configuration.

---

## 📈 Next Steps

- **CI/CD Enhancements:** Automate Docker image builds and publishing

---

## 📚 Academic and Teaching Notes

- **Best Practices:** Each module demonstrates academic best practices in code quality, modularity, and reproducibility, suitable for both teaching and production environments
- **Extensibility:** All logic is driven by config and easily extensible for advanced topics in MLOps
- **Assessment:** Project is fully test-driven, with sample data and clear structure for student/peer evaluation

---

## 👩‍💻 Authors and Acknowledgments

- Project led by Prof. Ivan Diaz
- Part of MLOps (IE University)
- Inspired by open-source MLOps community and real-world healthcare analytics use cases

---

## 📜 License

This project is for academic and educational purposes. See the [MIT License](LICENSE) for details.

---

**For questions or collaboration, open an issue or contact [maintainer email].**
