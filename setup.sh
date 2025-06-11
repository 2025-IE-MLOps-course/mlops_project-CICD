#!/usr/bin/env bash
set -e

# Install python dependencies listed in environment.yml
pip install \
    pandas==2.2.3 \
    numpy==2.2.6 \
    openpyxl==3.1.5 \
    pyyaml==6.0.2 \
    python-dotenv==1.1.0 \
    pytest==8.3.5 \
    pytest-dotenv==0.5.2 \
    scikit-learn==1.6.1 \
    scipy==1.15.2 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    dvc==3.59.2 \
    dvc-s3==3.2.0 \
    awscli==2.27.27 \
    pytest-cov==6.1.1 \
    black==25.1.0 \
    flake8==7.2.0 \
    mlflow-skinny==2.22.0 \
    wandb==0.19.11

# Add src to PYTHONPATH for this session
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

echo "Environment ready. PYTHONPATH set to include src/"
