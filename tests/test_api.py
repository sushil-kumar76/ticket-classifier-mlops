from fastapi.testclient import TestClient
from src.serve.app import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_known_intent():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": "My card has not arrived yet"})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] == "card_arrival"
        assert body["confidence"] > 0.5
        assert body["needs_review"] is False


def test_predict_low_confidence_flagged():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": "asdfgh qwerty zzz"})
        assert r.json()["needs_review"] is True


def test_predict_rejects_empty_text():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": ""})
        assert r.status_code == 422
