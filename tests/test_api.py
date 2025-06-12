import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_batch():
    payload = {
        "rx ds": 100,
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
    resp = client.post("/predict_batch", json=[payload, payload])
    assert resp.status_code == 200
    result = resp.json()
    assert isinstance(result, list)
    assert len(result) == 2
    assert set(result[0].keys()) == {"prediction", "probability"}
