"""
Integration tests for src/serving/api.py
Uses TestClient — no real server needed, no real model weights needed.
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.serving.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_rgb_bytes() -> bytes:
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _make_depth_bytes() -> bytes:
    arr = (np.random.rand(224, 224) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_has_model_loaded(self, client):
        r = client.get("/health")
        assert "model_loaded" in r.json()

    def test_health_has_device(self, client):
        r = client.get("/health")
        assert "device" in r.json()


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type(self, client):
        r = client.get("/metrics")
        assert "text/plain" in r.headers["content-type"]


class TestPredictEndpoint:

    def test_predict_returns_200(self, client):
        r = client.post(
            "/predict",
            files={
                "rgb_file":   ("rgb.png",   _make_rgb_bytes(),   "image/png"),
                "depth_file": ("depth.png", _make_depth_bytes(), "image/png"),
            },
        )
        assert r.status_code == 200

    def test_predict_has_detections_key(self, client):
        r = client.post(
            "/predict",
            files={
                "rgb_file":   ("rgb.png",   _make_rgb_bytes(),   "image/png"),
                "depth_file": ("depth.png", _make_depth_bytes(), "image/png"),
            },
        )
        assert "detections" in r.json()

    def test_predict_has_depth_quality(self, client):
        r = client.post(
            "/predict",
            files={
                "rgb_file":   ("rgb.png",   _make_rgb_bytes(),   "image/png"),
                "depth_file": ("depth.png", _make_depth_bytes(), "image/png"),
            },
        )
        data = r.json()
        assert "depth_quality" in data
        assert 0.0 <= data["depth_quality"] <= 1.0

    def test_predict_has_latency_ms(self, client):
        r = client.post(
            "/predict",
            files={
                "rgb_file":   ("rgb.png",   _make_rgb_bytes(),   "image/png"),
                "depth_file": ("depth.png", _make_depth_bytes(), "image/png"),
            },
        )
        assert "latency_ms" in r.json()


class TestFeedbackEndpoint:

    def test_feedback_returns_200(self, client):
        payload = {
            "image_hash":           "abc123",
            "original_prediction":  {"detections": []},
            "correction":           {"detections": [], "notes": "test"},
        }
        r = client.post("/feedback", json=payload)
        assert r.status_code == 200

    def test_feedback_returns_queued(self, client):
        payload = {
            "image_hash":           "abc123",
            "original_prediction":  {"detections": []},
            "correction":           {"detections": []},
        }
        r = client.post("/feedback", json=payload)
        assert r.json()["status"] == "queued"

    def test_queue_size_returns_200(self, client):
        r = client.get("/feedback/queue-size")
        assert r.status_code == 200