import numpy as np
from fastapi.testclient import TestClient

from src.api import app, SEQ_LEN, feats

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_dogru_boyut():
    window = np.random.rand(SEQ_LEN, len(feats)).tolist()
    r = client.post("/predict", json={"window": window})
    assert r.status_code == 200
    assert r.json()["rul"] >= 0


def test_predict_yanlis_boyut():
    window = np.random.rand(10, len(feats)).tolist()
    r = client.post("/predict", json={"window": window})
    assert r.status_code == 422