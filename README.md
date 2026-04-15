# Credit Card Fraud Detection — ML Project

An end-to-end machine learning system for detecting fraudulent credit card transactions, with a REST API and interactive dashboard.

---

## Project Structure

```
fraud_detection/
├── train_model.py        # Model training pipeline
├── api.py                # FastAPI REST API
├── app.py                # Streamlit dashboard
├── model_artifacts/
│   ├── xgb_model.pkl     # Trained XGBoost model
│   ├── scaler.pkl        # StandardScaler for Amount & Time
│   ├── shap_explainer.pkl# SHAP TreeExplainer
│   └── metrics.json      # Evaluation metrics + SHAP global importance
└── README.md
```

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions | 492 fraud (0.17%)
- **Features:** `Time`, `Amount`, `V1–V28` (PCA-anonymized), `Class` (0=legit, 1=fraud)

---

## Quickstart

### 1. Install dependencies
```bash
pip install pandas scikit-learn xgboost streamlit plotly imbalanced-learn shap fastapi uvicorn joblib
brew install libomp  # macOS only, required for XGBoost
```

### 2. Train the model
```bash
cd fraud_detection
python train_model.py
```

### 3. Start the dashboard
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 4. Start the REST API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs
```

---

## ML Pipeline

### Problem
Extreme class imbalance — only 0.17% of transactions are fraudulent. A naive model that predicts "legit" for everything achieves 99.83% accuracy but is completely useless.

### Solution

```
Raw CSV
  → StandardScaler (normalize Amount & Time)
  → Train/Test Split (80/20, stratified)
  → SMOTE (oversample fraud in training set)
  → XGBoost (200 trees, depth=6, lr=0.05)
  → SHAP Explainer (TreeExplainer on background sample)
  → Evaluation + Artifact Export
```

### Why SMOTE?
Synthetic Minority Oversampling Technique generates new synthetic fraud examples by interpolating between existing ones, rather than simply duplicating rows. This prevents the model from ignoring the minority class.

### Why XGBoost?
Gradient boosting builds 200 trees sequentially — each tree corrects the mistakes of the previous one. It handles imbalanced tabular data well, trains fast, and is the industry standard for structured datasets.

### Why SHAP?
Finance regulations (e.g. SR 11-7, GDPR Article 22) require models to explain their decisions. SHAP assigns each feature a contribution score for every individual prediction, enabling human-readable explanations like: *"This transaction was flagged because V14 was unusually low and the amount was 10x the user's average."*

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.9757 |
| Average Precision | 0.8390 |
| Fraud Recall | 88% |
| False Positive Rate | ~0.1% |

---

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata & top SHAP features |
| POST | `/predict` | Single transaction prediction |
| POST | `/predict/batch` | Batch prediction (up to 1000 transactions) |

### Example — Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 0, "Amount": 2000.0, "V1": -3.0, "V14": -5.0, ...}'
```

**Response:**
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9988,
  "risk_level": "CRITICAL",
  "explanation": {
    "V14": 2.31,
    "V12": 1.05,
    "V7": 0.70,
    "V10": 0.64,
    "V4": 0.53
  }
}
```

Interactive API docs available at: `http://localhost:8000/docs`

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | KPI cards, confusion matrix, class distribution, classification report |
| **Model Performance** | ROC curve, Precision-Recall curve, TP/FP/FN/TN breakdown |
| **Feature Analysis** | Feature importances, amount distributions, fraud over time |
| **Explainability (SHAP)** | Global SHAP importance + per-transaction waterfall explanation |
| **Live Predictor** | Manual transaction input or CSV batch upload with SHAP explanation |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, scikit-learn |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Model | XGBoost |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Serialization | joblib |

---

## What Makes This Industry-Ready

- **Explainability:** Every prediction comes with SHAP values — satisfies regulatory requirements
- **REST API:** Model served as a microservice, consumable by any downstream system
- **Risk levels:** LOW / MEDIUM / HIGH / CRITICAL tiering based on probability thresholds
- **Batch prediction:** Process up to 1000 transactions per API call
- **Separation of concerns:** Training, serving, and visualization are fully decoupled
