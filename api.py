"""
Fraud Detection REST API — FastAPI
Endpoints:
  POST /predict        — single transaction prediction
  POST /predict/batch  — batch prediction from JSON list
  GET  /health         — health check
  GET  /model/info     — model metadata & metrics
"""

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import joblib
import json
import numpy as np
import pandas as pd
import shap
import os

load_dotenv()

# ── API Key Auth ──────────────────────────────────────────────────────────────
_raw_keys = os.getenv("API_KEYS", "")
# Parse "name:key,name:key" format → set of valid keys
VALID_API_KEYS: set[str] = {
    part.split(":")[1] for part in _raw_keys.split(",") if ":" in part
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(api_key_header)) -> str:
    if not key or key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "XGBoost-based credit card fraud detection with SHAP explainability.\n\n"
        "All endpoints except `/health` require an `X-API-Key` header.\n\n"
        "**Demo keys:** `dash-key-123` · `analyst-key-456` · `admin-key-789`"
    ),
    version="1.0.0",
)

ARTIFACTS = "model_artifacts"

# ── Load artifacts once at startup ────────────────────────────────────────────
model = joblib.load(f"{ARTIFACTS}/xgb_model.pkl")
scaler = joblib.load(f"{ARTIFACTS}/scaler.pkl")
explainer = joblib.load(f"{ARTIFACTS}/shap_explainer.pkl")

with open(f"{ARTIFACTS}/metrics.json") as f:
    metrics = json.load(f)

FEATURE_NAMES = metrics["feature_names"]


# ── Request / Response schemas ────────────────────────────────────────────────
class Transaction(BaseModel):
    Time: float = Field(..., example=0.0, description="Seconds since first transaction")
    Amount: float = Field(..., example=149.62, description="Transaction amount in USD")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    explanation: dict


class BatchRequest(BaseModel):
    transactions: list[Transaction]


class BatchResponse(BaseModel):
    total: int
    fraud_count: int
    results: list[PredictionResponse]


# ── Helper ────────────────────────────────────────────────────────────────────
def preprocess(txn: Transaction) -> pd.DataFrame:
    amount_scaled = scaler.fit_transform([[txn.Amount]])[0][0]
    time_scaled = scaler.fit_transform([[txn.Time]])[0][0]
    row = {f"V{i}": getattr(txn, f"V{i}") for i in range(1, 29)}
    row["Amount_scaled"] = amount_scaled
    row["Time_scaled"] = time_scaled
    return pd.DataFrame([row])[FEATURE_NAMES]


def risk_label(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    elif prob < 0.85:
        return "HIGH"
    return "CRITICAL"


def get_shap_explanation(input_df: pd.DataFrame) -> dict:
    shap_vals = explainer(input_df)
    values = shap_vals.values[0]
    top_idx = np.argsort(np.abs(values))[::-1][:5]
    return {
        FEATURE_NAMES[i]: round(float(values[i]), 4)
        for i in top_idx
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "version": "1.0.0"}


@app.get("/model/info")
def model_info(api_key: str = Depends(require_api_key)):
    return {
        "roc_auc": metrics["roc_auc"],
        "avg_precision": metrics["avg_precision"],
        "test_size": metrics["test_size"],
        "fraud_count_in_test": metrics["fraud_count"],
        "features": FEATURE_NAMES,
        "top_shap_features": dict(
            sorted(metrics["shap_global_importance"].items(),
                   key=lambda x: x[1], reverse=True)[:10]
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction, api_key: str = Depends(require_api_key)):
    input_df = preprocess(txn)
    prob = float(model.predict_proba(input_df)[0][1])
    explanation = get_shap_explanation(input_df)
    return PredictionResponse(
        is_fraud=prob >= 0.5,
        fraud_probability=round(prob, 4),
        risk_level=risk_label(prob),
        explanation=explanation,
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest, api_key: str = Depends(require_api_key)):
    if len(req.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 transactions per batch.")
    results = []
    for txn in req.transactions:
        input_df = preprocess(txn)
        prob = float(model.predict_proba(input_df)[0][1])
        explanation = get_shap_explanation(input_df)
        results.append(PredictionResponse(
            is_fraud=prob >= 0.5,
            fraud_probability=round(prob, 4),
            risk_level=risk_label(prob),
            explanation=explanation,
        ))
    fraud_count = sum(1 for r in results if r.is_fraud)
    return BatchResponse(total=len(results), fraud_count=fraud_count, results=results)
