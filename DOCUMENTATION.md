# Fraud Detection ML Project — Full Documentation

Author: Raj Davande
GitHub: https://github.com/pkadapalla2002-star/fraud-detection-ml

---

## Table of Contents

1. Project Overview
2. Dataset
3. Project Structure
4. Step-by-Step: What Was Built and Why
   - Step 1: Data Acquisition
   - Step 2: Model Training Pipeline
   - Step 3: Streamlit Dashboard
   - Step 4: SHAP Explainability
   - Step 5: FastAPI REST API
   - Step 6: API Authentication
   - Step 7: Cost-Benefit Analysis
   - Step 8: Unit Tests
5. How to Run the Project
6. API Reference
7. Dashboard Pages
8. Model Performance
9. Key Concepts Explained
10. What Makes This Industry-Ready

---

## 1. Project Overview

This is an end-to-end machine learning system that detects fraudulent credit card transactions.
It is not just a notebook — it is a fully engineered system with a trained model, a REST API,
an interactive dashboard, explainability, authentication, and tests.

The goal was to build something that reflects how fraud detection actually works in a real bank
or fintech company — not just a model that scores well on accuracy, but a system that can be
reasoned about, explained, consumed by other services, and trusted by a business team.

---

## 2. Dataset

- Name:    Credit Card Fraud Detection
- Source:  Kaggle (mlg-ulb/creditcardfraud)
- File:    creditcard.csv (144MB — not committed to GitHub)
- Rows:    284,807 transactions
- Fraud:   492 cases (0.17% of all transactions)

Columns:
- Time     — seconds elapsed since the first transaction in the dataset
- Amount   — transaction amount in USD
- V1–V28   — anonymized features produced by PCA (to protect cardholder privacy)
- Class    — label: 0 = legitimate, 1 = fraudulent

The dataset is extremely imbalanced. Only 1 in every 578 transactions is fraud.
This is the central challenge the entire pipeline is designed to handle.

---

## 3. Project Structure

```
fraud_detection/
│
├── train_model.py              # Training pipeline — run once to generate artifacts
├── api.py                      # FastAPI REST API (port 8000)
├── app.py                      # Streamlit dashboard (port 8501)
│
├── model_artifacts/
│   ├── xgb_model.pkl           # Trained XGBoost model
│   ├── scaler.pkl              # StandardScaler for Amount and Time
│   ├── shap_explainer.pkl      # SHAP TreeExplainer
│   └── metrics.json            # All evaluation metrics + SHAP global importance
│
├── tests/
│   ├── __init__.py
│   └── test_api.py             # 19 unit tests for the API
│
├── .streamlit/
│   └── config.toml             # Dashboard theme (dark mode)
│
├── .env                        # API keys — never committed to GitHub
├── .gitignore                  # Excludes creditcard.csv and .env
├── requirements.txt            # All Python dependencies
├── README.md                   # Short project summary for GitHub
└── DOCUMENTATION.md            # This file
```

---

## 4. Step-by-Step: What Was Built and Why

---

### Step 1: Data Acquisition

What happened:
    Downloaded the dataset directly from Kaggle using the Kaggle CLI and API token.

Why:
    The IEEE-CIS Fraud Detection competition dataset required accepting competition rules
    on Kaggle's website before downloading — an unnecessary friction. The Credit Card Fraud
    dataset is a standard open dataset with no such gate, making it immediately usable.

Command used:
    kaggle datasets download -d mlg-ulb/creditcardfraud -p /Downloads/ --unzip

Result:
    creditcard.csv — 284,807 rows, 31 columns, 144MB.

---

### Step 2: Model Training Pipeline (train_model.py)

What happened:
    Built a full training pipeline that reads the raw CSV, preprocesses it, handles class
    imbalance, trains an XGBoost classifier, evaluates it, and saves all artifacts to disk.

#### 2a. Preprocessing

    The Amount and Time columns are on very different scales to the V1–V28 PCA features.
    Left unscaled, they can mislead the model. StandardScaler was applied to both,
    transforming them to have mean=0 and standard deviation=1.

    Amount and Time are then renamed to Amount_scaled and Time_scaled and the originals
    are dropped. The V1–V28 features are already scaled from PCA, so no action needed.

#### 2b. Train/Test Split

    80% of data used for training, 20% held out for testing.
    Stratified split ensures both sets have the same fraud ratio (0.17%).
    Random seed fixed at 42 for reproducibility.

#### 2c. SMOTE (Synthetic Minority Oversampling Technique)

    Problem:
        With only 492 fraud cases vs 227,845 legit cases in training, a model can achieve
        99.83% accuracy by predicting "legit" for everything. This is useless for fraud detection.

    Solution:
        SMOTE generates synthetic fraud examples by interpolating between existing fraud cases.
        It picks a real fraud transaction, finds its nearest fraud neighbors, and creates new
        points along the lines between them. The result is a balanced training set.

    Why not just duplicate fraud rows?
        Duplication causes the model to memorize specific fraud cases rather than learning
        the general pattern. SMOTE creates variation, leading to better generalization.

    Applied only to training data — never to test data. Applying it to test data would
    contaminate the evaluation with artificial samples and give falsely optimistic results.

#### 2d. XGBoost Classifier

    XGBoost (Extreme Gradient Boosting) trains 200 decision trees sequentially.
    Each tree focuses on the mistakes made by all previous trees combined.
    The final prediction is a weighted vote across all 200 trees.

    Why XGBoost:
    - Handles tabular data better than neural networks in most benchmarks
    - Robust to feature scale differences
    - Built-in regularization prevents overfitting
    - Fast training even on 280K+ rows
    - Industry standard for structured data in finance

    Parameters used:
        n_estimators=200    — number of trees
        max_depth=6         — maximum depth per tree (controls complexity)
        learning_rate=0.05  — how much each tree contributes (lower = more robust)
        n_jobs=-1           — use all CPU cores

#### 2e. SHAP Explainer

    After training, a SHAP TreeExplainer is created using a background sample of 200
    training rows. The explainer is used to compute SHAP values — a way of measuring
    how much each feature contributed to a specific prediction.

    The explainer is saved as shap_explainer.pkl so the dashboard and API can load it
    without retraining.

    A global SHAP importance summary (average |SHAP value| per feature across 500 test
    samples) is computed and saved into metrics.json.

#### 2f. Saved Artifacts

    xgb_model.pkl       — the trained model
    scaler.pkl          — the fitted StandardScaler
    shap_explainer.pkl  — the SHAP TreeExplainer
    metrics.json        — ROC-AUC, avg precision, confusion matrix, classification report,
                          ROC curve points, PR curve points, feature names,
                          SHAP global importance

---

### Step 3: Streamlit Dashboard (app.py)

What happened:
    Built an interactive multi-page dashboard using Streamlit and Plotly.

Why Streamlit:
    Streamlit converts Python scripts into web apps with no HTML/CSS/JavaScript required.
    It is the standard tool for ML dashboards in the industry — interviewers and stakeholders
    can understand and use it immediately.

Why Plotly:
    Plotly charts are interactive (hover, zoom, filter) unlike static matplotlib charts.
    Interactive charts are expected in any professional data product.

Pages built:
    1. Overview           — KPIs, confusion matrix, class distribution, classification report
    2. Model Performance  — ROC curve, Precision-Recall curve, TP/FP/FN/TN detail
    3. Feature Analysis   — Feature importances, amount distributions, fraud over time
    4. Explainability     — SHAP global importance + per-transaction waterfall chart
    5. Cost-Benefit       — Business threshold analysis (see Step 7)
    6. Live Predictor     — Enter any transaction manually, get prediction + SHAP chart

All artifacts are loaded once at startup using @st.cache_resource (model, scaler, explainer)
and @st.cache_data (metrics JSON). This means subsequent page loads are instant — nothing
is reloaded from disk on every interaction.

The raw CSV data loader was made resilient to multiple file paths so the dashboard works
both locally (where the CSV is in ../creditcard.csv) and on any cloud deployment where
it might be placed differently or not available at all.

---

### Step 4: SHAP Explainability

What happened:
    Added a dedicated Explainability page to the dashboard and integrated SHAP explanations
    into the Live Predictor page.

Why this matters in finance:
    Regulations such as the EU's GDPR Article 22 and the US Federal Reserve's SR 11-7
    guidance require financial institutions to explain automated decisions to customers
    and auditors. "The model said so" is not acceptable. SHAP provides a mathematically
    rigorous way to attribute each prediction to specific input features.

What SHAP values mean:
    Each feature gets a SHAP value for a specific prediction.
    - Positive SHAP value = this feature pushed the prediction toward FRAUD
    - Negative SHAP value = this feature pushed the prediction toward LEGIT
    - Magnitude = how strongly it pushed

    Example: "V14 contributed +2.31 toward fraud, V3 contributed -0.85 toward legit"

Global vs local explanations:
    - Global: average |SHAP| across many predictions — tells you which features matter most
              overall. More reliable than raw feature importance from the model.
    - Local: SHAP values for one specific transaction — tells you exactly why THIS transaction
             was flagged. This is what you show to a compliance officer or customer.

Dashboard implementation:
    - Explainability page: loads a random transaction from the dataset, runs SHAP,
      shows a waterfall bar chart with red bars (→ fraud) and green bars (→ legit)
    - Live Predictor: after every manual prediction, automatically shows the SHAP
      waterfall for that transaction

---

### Step 5: FastAPI REST API (api.py)

What happened:
    Wrapped the model in a REST API using FastAPI, served by Uvicorn on port 8000.

Why an API:
    The Streamlit dashboard is for humans. The API is for machines.
    In a real bank, a transaction processing system would call this API in real time
    as each transaction happens — the ML model is just one service in a larger pipeline.
    Without an API, the model is a dead end.

Why FastAPI:
    - Automatically generates interactive Swagger documentation at /docs
    - Built-in data validation via Pydantic (wrong input types = informative error, not crash)
    - Async-capable for high-throughput scenarios
    - Industry standard for Python ML APIs

Endpoints:
    GET  /health           — health check, no auth required
                             used by load balancers and monitoring systems
    GET  /model/info       — model metadata, top SHAP features, performance metrics
    POST /predict          — single transaction → fraud probability + risk level + explanation
    POST /predict/batch    — up to 1000 transactions at once → full results + fraud count

Response structure for /predict:
    {
        "is_fraud": true/false,
        "fraud_probability": 0.0–1.0,
        "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
        "explanation": { "V14": 2.31, "V12": 1.05, ... }
    }

Risk levels:
    0.0–0.30  → LOW
    0.30–0.60 → MEDIUM
    0.60–0.85 → HIGH
    0.85–1.0  → CRITICAL

    These thresholds were chosen based on typical fraud operations workflows where
    LOW/MEDIUM = monitor, HIGH = flag for review, CRITICAL = auto-block.

---

### Step 6: API Authentication

What happened:
    Added API key authentication to all endpoints except /health.
    Keys are stored in a .env file that is never committed to GitHub.

Why authentication:
    An unauthenticated ML API is a security liability. Anyone could:
    - Query the model to probe for weaknesses (model inversion attacks)
    - Use it without authorisation
    - Overwhelm it with requests (denial of service)

    In any real financial system, every internal service-to-service call is authenticated.

How it works:
    1. .env file contains API keys in "name:key" format
    2. The API reads these at startup into a set of valid keys
    3. Every protected endpoint uses FastAPI's Security dependency to check
       the X-API-Key header against the valid keys set
    4. Invalid or missing key → 401 Unauthorized immediately, no model is ever called

    The keys are kept in a set (not a list) so lookup is O(1) regardless of how many
    keys exist — important for high-traffic APIs.

Demo keys (in .env):
    dash-key-123       — for the dashboard
    analyst-key-456    — for analysts
    admin-key-789      — for admin use

Why .env is not committed:
    If API keys are committed to a public GitHub repository, they are permanently exposed
    even if later deleted (git history retains them). The .gitignore explicitly excludes
    .env to prevent this.

---

### Step 7: Cost-Benefit Analysis Page

What happened:
    Added a business-facing dashboard page that converts model metrics into dollar figures
    and lets users tune the decision threshold based on real costs.

Why this is important:
    Finance professionals do not think in ROC-AUC. They think in money.
    A model that catches 88% of fraud sounds good — but is it worth it if it also
    blocks 500 legitimate customers a day, each costing $5 in support calls?
    That depends entirely on the average fraud value, which this page makes explicit.

    Additionally, the default threshold of 0.5 is almost never optimal for imbalanced
    datasets. The cost-benefit page shows the user the threshold that maximises
    net dollar benefit given their specific cost inputs.

What the page does:
    1. User inputs three business parameters:
       - Average fraud transaction value (default: $122 — actual dataset average)
       - Cost per false positive (default: $5 — customer service + churn risk)
       - Cost per false negative (default: $122 — full fraud amount lost)

    2. Computes across every threshold point from the saved PR curve:
       - Fraud caught (TP) and missed (FN)
       - False alarms (FP)
       - Dollar value of fraud saved = TP × fraud amount
       - Dollar cost of false alarms = FP × false positive cost
       - Net benefit = fraud saved − false alarm cost

    3. Displays:
       - Net benefit chart across all thresholds, with optimal threshold marked in gold
       - KPI row at the optimal threshold: fraud caught, false alarms, net benefit, vs baseline
       - Precision-Recall curve with the optimal point highlighted
       - Scenario comparison table: Conservative vs Optimal vs Aggressive strategies

    4. Shows the key insight: at threshold 0.5, the model leaves money on the table.
       Lowering the threshold catches more fraud and increases net benefit — but only
       up to the point where false alarm costs outweigh additional fraud caught.

---

### Step 8: Unit Tests (tests/test_api.py)

What happened:
    Wrote 19 unit tests for the API using pytest and FastAPI's built-in TestClient.

Why tests:
    In a finance codebase, untested code is not production-ready code.
    Tests serve three purposes:
    1. Prove the code works correctly right now
    2. Catch regressions when code changes in the future
    3. Signal to any interviewer or engineering team that you write professional code

    A portfolio project without tests is a toy. With tests it becomes a system.

Test structure:
    TestHealth (2 tests)
        - Returns 200 status
        - Does not require authentication

    TestModelInfo (4 tests)
        - Blocked without API key
        - Blocked with wrong API key
        - Returns metrics including ROC-AUC > 0.9 with valid key
        - Includes SHAP features in response

    TestPredict (8 tests)
        - Blocked without API key
        - Blocked with wrong key
        - Response contains all required fields
        - Fraud probability is between 0 and 1
        - Risk level is one of the four valid values
        - A known high-anomaly transaction is flagged as fraud with >0.9 probability
        - Explanation contains at least 3 SHAP features
        - Missing V fields default to 0.0 without error

    TestPredictBatch (5 tests)
        - Blocked without API key
        - Response has correct total count and results array
        - Fraud is detected in a batch containing a known fraud transaction
        - Batches over 1000 transactions are rejected with 400 status
        - Single-item batch works correctly

All 19 tests pass. Run time: ~1.3 seconds.

---

## 5. How to Run the Project

### Prerequisites
    Python 3.10+
    brew install libomp    (macOS only — required for XGBoost)

### Install dependencies
    pip install -r requirements.txt

### Train the model (run once)
    cd fraud_detection
    python train_model.py

    This generates all files in model_artifacts/ and takes about 2–3 minutes.

### Start the dashboard
    streamlit run app.py
    Open: http://localhost:8501

### Start the API
    uvicorn api:app --host 0.0.0.0 --port 8000
    Open: http://localhost:8000/docs

### Run the tests
    pytest tests/ -v

---

## 6. API Reference

### Authentication
    All endpoints except /health require the header:
    X-API-Key: <your-key>

    Available demo keys: dash-key-123 | analyst-key-456 | admin-key-789

### GET /health
    No authentication required.
    Response: { "status": "ok", "model": "XGBoost", "version": "1.0.0" }

### GET /model/info
    Requires authentication.
    Response: ROC-AUC, avg precision, test size, fraud count, feature list, top SHAP features.

### POST /predict
    Requires authentication.
    Body: Transaction object with Time, Amount, V1–V28 (V fields optional, default 0.0)
    Response:
        {
            "is_fraud": true,
            "fraud_probability": 0.9988,
            "risk_level": "CRITICAL",
            "explanation": {
                "V14": 2.31,
                "V12": 1.05,
                "V7": 0.70
            }
        }

### POST /predict/batch
    Requires authentication.
    Body: { "transactions": [ <Transaction>, ... ] }   (max 1000)
    Response:
        {
            "total": 2,
            "fraud_count": 1,
            "results": [ <PredictionResponse>, ... ]
        }

### Example curl call
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -H "X-API-Key: dash-key-123" \
      -d '{"Time": 0, "Amount": 299.99, "V1": -2.5, "V14": -4.0}'

---

## 7. Dashboard Pages

    Page 1 — Overview
        KPI cards: ROC-AUC, Avg Precision, Fraud Detected, False Positives
        Class distribution pie chart
        Confusion matrix heatmap
        Classification report table

    Page 2 — Model Performance
        ROC Curve with AUC score
        Precision-Recall Curve with average precision score
        Detailed TP / FP / FN / TN metric cards

    Page 3 — Feature Analysis
        Top 15 feature importances (from XGBoost)
        Transaction amount distribution: legit vs fraud side by side
        Fraud vs legit volume over time
        (Requires local creditcard.csv — skipped on cloud deployment)

    Page 4 — Explainability (SHAP)
        Global SHAP importance bar chart (top 15 features)
        Load random transaction button
        Per-transaction SHAP waterfall chart
        Shows prediction, actual label, fraud probability, and correct/wrong indicator

    Page 5 — Cost-Benefit Analysis
        Business parameter inputs (fraud amount, FP cost, FN cost)
        Net dollar benefit across all thresholds
        Optimal threshold highlighted
        KPIs at optimal threshold
        Precision-Recall curve with optimal point
        Scenario comparison table

    Page 6 — Live Predictor
        Manual transaction input (all 30 features)
        Load random sample button
        Fraud probability gauge chart
        SHAP waterfall explanation for the prediction
        Batch CSV upload with download results button

---

## 8. Model Performance

    Metric                  Value
    ────────────────────────────────
    ROC-AUC                 0.9757
    Average Precision       0.8390
    Fraud Recall            88%
    False Positive Rate     ~0.1%

    Confusion Matrix (test set — 56,962 transactions):
        True Negatives  (legit, correctly cleared):   56,778
        False Positives (legit, wrongly flagged):         86
        False Negatives (fraud, missed):                  12
        True Positives  (fraud, correctly caught):        86

    Interpretation:
        The model catches 86 out of 98 fraud cases in the test set.
        It only wrongly flags 86 out of 56,864 legitimate transactions (0.15%).
        The 12 missed fraud cases represent the cost of the conservative 0.5 threshold —
        lowering the threshold catches more of them at the cost of more false alarms.

---

## 9. Key Concepts Explained

    Class Imbalance
        When one category is far rarer than another. Here: 0.17% fraud vs 99.83% legit.
        A naive model predicting "legit" always scores 99.83% accuracy but is useless.

    SMOTE
        Synthetic Minority Oversampling Technique. Creates artificial fraud examples by
        interpolating between existing fraud cases. Applied only to training data.

    XGBoost
        Gradient boosting algorithm. Builds 200 trees sequentially, each correcting the
        mistakes of the previous ones. Best-in-class for tabular data.

    ROC-AUC
        Area Under the ROC Curve. Measures how well the model separates classes.
        1.0 = perfect. 0.5 = random. 0.9757 = excellent.

    Precision
        Of all transactions flagged as fraud, what fraction were actually fraud.
        Low precision = many false alarms.

    Recall
        Of all actual fraud cases, what fraction were caught.
        Low recall = many missed frauds.

    SHAP Values
        SHapley Additive exPlanations. Assigns each feature a contribution score for
        each individual prediction. Enables regulatory-compliant model explanations.

    Decision Threshold
        The minimum probability score to classify a transaction as fraud.
        Default is 0.5, but the optimal value depends on business costs.

    API Key Authentication
        A secret token that must be included in every API request.
        Prevents unauthorized access to the model.

---

## 10. What Makes This Industry-Ready

    1. SHAP Explainability
       Satisfies regulatory requirements (GDPR Art. 22, SR 11-7).
       Every prediction can be explained to a compliance officer or customer.

    2. REST API
       The model is a service, not a script. Any system can consume it.
       Reflects how ML is actually deployed in financial institutions.

    3. API Authentication
       Keys stored in .env, never in code or Git history.
       Protected against unauthorized access and model probing attacks.

    4. Cost-Benefit Analysis
       Converts ML metrics into business dollars.
       Shows optimal threshold tuning — something most ML engineers skip.

    5. Unit Tests (19 passing)
       Proves the system works. Catches regressions. Signals engineering professionalism.

    6. Separation of Concerns
       Training, serving, and visualization are fully independent.
       You can retrain the model without touching the API or dashboard.

    7. Version Controlled
       Full Git history on GitHub. Model artifacts committed for zero-setup reproducibility.
       .env and large data files correctly excluded.

---

End of Documentation
