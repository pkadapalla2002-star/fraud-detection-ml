"""
Unit tests for the Fraud Detection API.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api import app

client = TestClient(app)

VALID_KEY = "dash-key-123"
BAD_KEY   = "not-a-real-key"

# ── Sample transactions ────────────────────────────────────────────────────────
LEGIT_TXN = {
    "Time": 0, "Amount": 2.69,
    "V1": 1.19, "V2": 0.27, "V3": 0.17, "V4": 0.45,
    "V5": 0.06, "V6": -0.08, "V7": -0.08, "V8": 0.09,
    "V9": -0.26, "V10": -0.17, "V11": 1.61, "V12": 1.07,
    "V13": 0.49, "V14": -0.14, "V15": 0.64, "V16": 0.46,
    "V17": -0.11, "V18": -0.18, "V19": -0.15, "V20": -0.07,
    "V21": -0.23, "V22": -0.64, "V23": 0.10, "V24": -0.34,
    "V25": 0.17, "V26": 0.13, "V27": -0.01, "V28": 0.01,
}

FRAUD_TXN = {
    "Time": 406, "Amount": 2000.0,
    "V1": -3.0, "V2": 2.5, "V3": -4.0, "V4": 1.2,
    "V5": 0.0, "V6": 0.0, "V7": -2.5, "V8": 0.0,
    "V9": 0.0, "V10": -3.5, "V11": 0.0, "V12": -4.0,
    "V13": 0.0, "V14": -5.0, "V15": 0.0, "V16": 0.0,
    "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
    "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0,
}


# ── /health ────────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_no_auth_required(self):
        """Health endpoint must be open — used by load balancers."""
        r = client.get("/health")
        assert r.status_code == 200


# ── /model/info ────────────────────────────────────────────────────────────────
class TestModelInfo:
    def test_requires_auth(self):
        r = client.get("/model/info")
        assert r.status_code == 401

    def test_rejects_bad_key(self):
        r = client.get("/model/info", headers={"X-API-Key": BAD_KEY})
        assert r.status_code == 401

    def test_returns_metrics_with_valid_key(self):
        r = client.get("/model/info", headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 200
        data = r.json()
        assert "roc_auc" in data
        assert data["roc_auc"] > 0.9

    def test_includes_top_shap_features(self):
        r = client.get("/model/info", headers={"X-API-Key": VALID_KEY})
        assert "top_shap_features" in r.json()


# ── /predict ───────────────────────────────────────────────────────────────────
class TestPredict:
    def test_requires_auth(self):
        r = client.post("/predict", json=LEGIT_TXN)
        assert r.status_code == 401

    def test_rejects_bad_key(self):
        r = client.post("/predict", json=LEGIT_TXN, headers={"X-API-Key": BAD_KEY})
        assert r.status_code == 401

    def test_response_shape(self):
        r = client.post("/predict", json=LEGIT_TXN, headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 200
        data = r.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert "explanation" in data

    def test_probability_in_range(self):
        r = client.post("/predict", json=LEGIT_TXN, headers={"X-API-Key": VALID_KEY})
        prob = r.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_level_valid(self):
        r = client.post("/predict", json=LEGIT_TXN, headers={"X-API-Key": VALID_KEY})
        assert r.json()["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_suspicious_transaction_flagged(self):
        """High-anomaly transaction should be flagged as fraud."""
        r = client.post("/predict", json=FRAUD_TXN, headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 200
        assert r.json()["is_fraud"] is True
        assert r.json()["fraud_probability"] > 0.9

    def test_explanation_has_top_features(self):
        """Explanation should contain at least 3 SHAP features."""
        r = client.post("/predict", json=FRAUD_TXN, headers={"X-API-Key": VALID_KEY})
        assert len(r.json()["explanation"]) >= 3

    def test_missing_field_uses_default(self):
        """V fields are optional and default to 0.0."""
        minimal = {"Time": 0, "Amount": 10.0}
        r = client.post("/predict", json=minimal, headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 200


# ── /predict/batch ─────────────────────────────────────────────────────────────
class TestPredictBatch:
    def test_requires_auth(self):
        r = client.post("/predict/batch", json={"transactions": [LEGIT_TXN]})
        assert r.status_code == 401

    def test_batch_response_shape(self):
        payload = {"transactions": [LEGIT_TXN, FRAUD_TXN]}
        r = client.post("/predict/batch", json=payload, headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert "fraud_count" in data
        assert len(data["results"]) == 2

    def test_batch_detects_fraud(self):
        payload = {"transactions": [LEGIT_TXN, FRAUD_TXN]}
        r = client.post("/predict/batch", json=payload, headers={"X-API-Key": VALID_KEY})
        assert r.json()["fraud_count"] >= 1

    def test_batch_limit_enforced(self):
        """Batches over 1000 transactions should be rejected."""
        big_batch = {"transactions": [LEGIT_TXN] * 1001}
        r = client.post("/predict/batch", json=big_batch, headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 400

    def test_single_item_batch(self):
        payload = {"transactions": [LEGIT_TXN]}
        r = client.post("/predict/batch", json=payload, headers={"X-API-Key": VALID_KEY})
        assert r.json()["total"] == 1
