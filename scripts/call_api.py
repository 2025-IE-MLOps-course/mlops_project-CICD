"""Simple helper to POST sample data to the FastAPI service."""

from __future__ import annotations

import json
import requests

SAMPLE_PAYLOAD = {
    "rx_ds": 330,
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 1,
    "E": 1,
    "F": 1,
    "H": 0,
    "I": 1,
    "J": 1,
    "K": 0,
    "L": 1,
    "M": 1,
    "N": 1,
    "R": 0,
    "S": 0,
    "T": 0,
    "V": 0,
}

if __name__ == "__main__":
    resp = requests.post("http://localhost:8000/predict", json=SAMPLE_PAYLOAD)
    print("Status:", resp.status_code)
    try:
        print("Response:", json.dumps(resp.json(), indent=2))
    except ValueError:
        print("Invalid JSON response")
