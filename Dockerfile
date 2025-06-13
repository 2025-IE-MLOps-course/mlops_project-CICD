FROM python:3.10-slim

WORKDIR /code
ENV PYTHONPATH=/code/src

# install dependencies first to leverage layer caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# copy application code
COPY app ./app
COPY src ./src
COPY config.yaml ./
COPY main.py ./
COPY MLproject ./
COPY scripts ./scripts

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "python scripts/download_from_wandb.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
