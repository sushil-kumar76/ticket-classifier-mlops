"""Contract tests: run against any model, real or stub."""
from fastapi.testclient import TestClient

from src.serve.app import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_response_schema():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": "My card has not arrived"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"label", "confidence", "needs_review", "latency_ms"}
        assert 0.0 <= body["confidence"] <= 1.0
        assert isinstance(body["needs_review"], bool)

def test_predict_rejects_empty_text():
    with TestClient(app) as client:
        assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_oversized_text():
    with TestClient(app) as client:
        assert client.post("/predict", json={"text": "a" * 1001}).status_code == 422


def test_metrics_endpoint_exposed():
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "http_request_duration_seconds" in r.text
