"""
Train XGBoost fraud detection model and save artifacts.
Run this once before launching the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import joblib
import json
import os

DATA_PATH = "../creditcard.csv"
OUT_DIR = "model_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Scale Amount and Time
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
df.drop(columns=["Amount", "Time"], inplace=True)

X = df.drop("Class", axis=1)
y = df["Class"]

print(f"Class distribution: {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Applying SMOTE to balance training set...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"\nROC-AUC:  {roc_auc:.4f}")
print(f"Avg Precision: {avg_precision:.4f}")
print(classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
# Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)

# Feature importance
feat_imp = dict(zip(X.columns.tolist(), model.feature_importances_.tolist()))

# SHAP explainer — use a background sample for speed
print("Computing SHAP explainer...")
background = X_train_res.sample(200, random_state=42)
explainer = shap.TreeExplainer(model, background)
shap_values_sample = explainer(X_test.sample(500, random_state=42))
shap_global = dict(zip(
    X.columns.tolist(),
    np.abs(shap_values_sample.values).mean(axis=0).tolist()
))

# Save everything
joblib.dump(model, f"{OUT_DIR}/xgb_model.pkl")
joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")
joblib.dump(explainer, f"{OUT_DIR}/shap_explainer.pkl")

metrics = {
    "roc_auc": roc_auc,
    "avg_precision": avg_precision,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "feature_importance": feat_imp,
    "roc_curve": {
        "fpr": fpr.tolist()[::10],
        "tpr": tpr.tolist()[::10],
    },
    "pr_curve": {
        "precision": precision.tolist()[::10],
        "recall": recall.tolist()[::10],
    },
    "test_size": len(y_test),
    "fraud_count": int(y_test.sum()),
    "total_train": len(y_train_res),
    "feature_names": X.columns.tolist(),
    "shap_global_importance": shap_global,
}

with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"\nArtifacts saved to ./{OUT_DIR}/")
