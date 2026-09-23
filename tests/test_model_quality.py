"""Accuracy tests: skipped unless the real trained model is present."""
import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    os.getenv("STUB_MODEL") == "1",
    reason="requires the real trained model, not the CI stub")

from src.serve.app import app  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("My card has not arrived yet", "card_arrival"),
    ("I was charged twice for my card payment", "transaction_charged_twice"),
])
def test_known_intents(text, expected):
    with TestClient(app) as client:
        body = client.post("/predict", json={"text": text}).json()
        assert body["label"] == expected
        assert body["confidence"] > 0.5
        assert body["needs_review"] is False


def test_gibberish_is_flagged_for_review():
    with TestClient(app) as client:
        body = client.post("/predict", json={"text": "asdfgh qwerty zzz"}).json()
        assert body["needs_review"] is True
