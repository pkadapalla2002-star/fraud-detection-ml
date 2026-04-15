"""
Fraud Detection Dashboard — Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import shap
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS = "model_artifacts"

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(f"{ARTIFACTS}/xgb_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load(f"{ARTIFACTS}/scaler.pkl")

@st.cache_resource
def load_explainer():
    return joblib.load(f"{ARTIFACTS}/shap_explainer.pkl")

@st.cache_data
def load_metrics():
    with open(f"{ARTIFACTS}/metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_data():
    # Support both local dev and cloud deployment paths
    for path in ["../creditcard.csv", "creditcard.csv", "data/creditcard.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            scaler = load_scaler()
            df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
            df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
            return df
    return None  # data not available (cloud deploy without CSV)

artifacts_exist = os.path.exists(f"{ARTIFACTS}/xgb_model.pkl")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Fraud Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model Performance", "Feature Analysis", "Explainability (SHAP)", "Cost-Benefit Analysis", "Live Predictor"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Credit Card Fraud (Kaggle)\n284,807 transactions · 492 frauds")

if not artifacts_exist:
    st.error("Model not trained yet. Run `python train_model.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Credit Card Fraud Detection")
    st.markdown("An XGBoost model trained on 284K real transactions with SMOTE balancing.")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}", "Excellent (>0.99)")
    col2.metric("Avg Precision", f"{metrics['avg_precision']:.4f}")
    col3.metric("Test Frauds Detected",
                f"{metrics['confusion_matrix'][1][1]}",
                f"out of {metrics['fraud_count']}")
    col4.metric("False Positives", f"{metrics['confusion_matrix'][0][1]}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Class distribution
    with col_left:
        st.subheader("Class Distribution")
        fig = go.Figure(go.Pie(
            labels=["Legit", "Fraud"],
            values=[284315, 492],
            hole=0.5,
            marker_colors=["#2ecc71", "#e74c3c"],
        ))
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    with col_right:
        st.subheader("Confusion Matrix (Test Set)")
        cm = metrics["confusion_matrix"]
        labels = ["Legit", "Fraud"]
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        ))
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Classification report table
    st.subheader("Classification Report")
    rep = metrics["classification_report"]
    rows = []
    for label in ["0", "1"]:
        name = "Legit" if label == "0" else "Fraud"
        rows.append({
            "Class": name,
            "Precision": f"{rep[label]['precision']:.4f}",
            "Recall": f"{rep[label]['recall']:.4f}",
            "F1-Score": f"{rep[label]['f1-score']:.4f}",
            "Support": int(rep[label]["support"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance")

    col1, col2 = st.columns(2)

    # ROC Curve
    with col1:
        st.subheader("ROC Curve")
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC={metrics['roc_auc']:.4f})",
                                 line=dict(color="#3498db", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          height=400, legend=dict(x=0.6, y=0.1))
        st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall Curve
    with col2:
        st.subheader("Precision-Recall Curve")
        precision = metrics["pr_curve"]["precision"]
        recall = metrics["pr_curve"]["recall"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                 name=f"AP={metrics['avg_precision']:.4f}",
                                 line=dict(color="#e74c3c", width=2)))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                          height=400, legend=dict(x=0.6, y=0.9))
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix numbers
    st.subheader("Confusion Matrix Detail")
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negatives (Legit correct)", f"{tn:,}")
    c2.metric("False Positives (Legit as Fraud)", f"{fp:,}", delta=f"-{fp}", delta_color="inverse")
    c3.metric("False Negatives (Fraud missed)", f"{fn:,}", delta=f"-{fn}", delta_color="inverse")
    c4.metric("True Positives (Fraud caught)", f"{tp:,}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: Feature Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Feature Analysis":
    st.title("Feature Analysis")

    # Feature importance
    st.subheader("Top 15 Feature Importances")
    feat_imp = metrics["feature_importance"]
    fi_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"])
    fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=500, yaxis=dict(autorange="reversed"),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Load data for distributions
    st.subheader("Transaction Amount Distribution")
    with st.spinner("Loading data..."):
        df = load_data()

    if df is None:
        st.info("Raw dataset not available in this deployment. Amount distribution and time charts require the local creditcard.csv file.")
        st.stop()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Legit Transactions", "Fraudulent Transactions"])
    legit = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]
    fig.add_trace(go.Histogram(x=legit, nbinsx=80, marker_color="#2ecc71", name="Legit"), row=1, col=1)
    fig.add_trace(go.Histogram(x=fraud, nbinsx=80, marker_color="#e74c3c", name="Fraud"), row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Transactions over time
    st.subheader("Fraud vs Legit Over Time")
    df["time_hour"] = (df["Time"] // 3600).astype(int) if "Time" in df.columns else (df["Time_scaled"] * 10).astype(int)
    time_df = df.groupby(["time_hour", "Class"]).size().reset_index(name="count")
    time_df["Type"] = time_df["Class"].map({0: "Legit", 1: "Fraud"})
    fig = px.line(time_df, x="time_hour", y="count", color="Type",
                  color_discrete_map={"Legit": "#2ecc71", "Fraud": "#e74c3c"},
                  labels={"time_hour": "Hour", "count": "Transactions"})
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: Explainability (SHAP)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Explainability (SHAP)":
    st.title("Model Explainability — SHAP")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** tells us *why* the model made a specific prediction.
    Each bar below shows how much a feature pushed the prediction toward fraud (positive) or legit (negative).
    This is critical in finance — regulators require models to be explainable.
    """)

    explainer = load_explainer()
    feature_names = metrics["feature_names"]

    # Global SHAP importance
    st.subheader("Global Feature Importance (SHAP)")
    st.caption("Average impact of each feature across all predictions — more reliable than raw feature importance.")
    shap_global = metrics.get("shap_global_importance", {})
    shap_df = pd.DataFrame(list(shap_global.items()), columns=["Feature", "Mean |SHAP|"])
    shap_df = shap_df.sort_values("Mean |SHAP|", ascending=False).head(15)
    fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                 color="Mean |SHAP|", color_continuous_scale="Reds")
    fig.update_layout(height=500, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Per-transaction SHAP explanation
    st.subheader("Explain a Single Transaction")
    st.caption("Load a random transaction and see exactly why it was flagged or cleared.")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Load Random Transaction", type="primary"):
            df = load_data()
            sample = df.sample(1).iloc[0]
            st.session_state["shap_sample"] = sample

    if "shap_sample" in st.session_state:
        sample = st.session_state["shap_sample"]
        scaler = load_scaler()
        model = load_model()

        v_cols = [f"V{i}" for i in range(1, 29)]
        row = {col: sample[col] for col in v_cols}
        row["Amount_scaled"] = sample["Amount_scaled"]
        row["Time_scaled"] = sample["Time_scaled"]
        input_df = pd.DataFrame([row])[feature_names]

        prob = model.predict_proba(input_df)[0][1]
        pred = "FRAUD" if prob >= 0.5 else "Legit"
        actual = "FRAUD" if sample["Class"] == 1 else "Legit"

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prediction", pred)
        col_b.metric("Fraud Probability", f"{prob*100:.2f}%")
        col_c.metric("Actual Label", actual, delta="Correct" if pred == actual else "Wrong",
                     delta_color="normal" if pred == actual else "inverse")

        # SHAP waterfall for this transaction
        shap_vals = explainer(input_df)
        values = shap_vals.values[0]
        base = float(shap_vals.base_values[0])

        shap_df_single = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": values,
        }).sort_values("SHAP Value", key=abs, ascending=False).head(12)

        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df_single["SHAP Value"]]

        fig = go.Figure(go.Bar(
            x=shap_df_single["SHAP Value"],
            y=shap_df_single["Feature"],
            orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(
            title=f"Why this transaction was predicted as: {pred}",
            xaxis_title="SHAP Value (red = pushes toward FRAUD, green = pushes toward LEGIT)",
            height=450,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Model base value (average prediction): {base:.4f}. "
                   f"Sum of SHAP values shifts it to: {prob:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: Cost-Benefit Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Cost-Benefit Analysis":
    st.title("Cost-Benefit Analysis")
    st.markdown("""
    Finance isn't about maximising accuracy — it's about **maximising money saved**.
    This page lets you tune the decision threshold and see the real business impact in dollars.
    """)

    # ── Inputs ────────────────────────────────────────────────────────────────
    st.subheader("Business Parameters")
    st.caption("Adjust these to reflect your bank's real costs.")

    col1, col2, col3 = st.columns(3)
    avg_fraud_amount = col1.number_input(
        "Avg Fraud Transaction ($)",
        value=122.0, min_value=1.0, step=1.0,
        help="Average dollar value of a fraudulent transaction. Dataset avg = $122."
    )
    fp_cost = col2.number_input(
        "Cost per False Positive ($)",
        value=5.0, min_value=0.0, step=0.5,
        help="Cost of wrongly blocking a legit transaction — customer service call, card unblock, churn risk."
    )
    fn_cost = col3.number_input(
        "Cost per False Negative ($)",
        value=avg_fraud_amount, min_value=0.0, step=1.0,
        help="Cost of missing a fraud — typically the full transaction amount lost."
    )

    st.markdown("---")

    # ── Compute metrics across thresholds ─────────────────────────────────────
    pr_curve = metrics["pr_curve"]
    roc_curve_data = metrics["roc_curve"]
    cm = metrics["confusion_matrix"]
    test_size = metrics["test_size"]
    fraud_in_test = metrics["fraud_count"]
    legit_in_test = test_size - fraud_in_test

    # Reconstruct threshold sweep from PR curve points
    precisions = np.array(pr_curve["precision"])
    recalls = np.array(pr_curve["recall"])

    # Estimate TP, FP, FN, TN at each threshold point
    tp_arr = (recalls * fraud_in_test).astype(int)
    fn_arr = fraud_in_test - tp_arr
    # precision = tp / (tp + fp) → fp = tp/precision - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        fp_arr = np.where(precisions > 0, (tp_arr / precisions - tp_arr).astype(int), 0)
    tn_arr = legit_in_test - fp_arr

    fraud_saved   = tp_arr * fn_cost
    fp_losses     = fp_arr * fp_cost
    net_benefit   = fraud_saved - fp_losses
    no_model_loss = fraud_in_test * fn_cost  # baseline: catch nothing

    threshold_labels = np.linspace(0.1, 0.9, len(recalls))

    # ── Chart 1: Net benefit across thresholds ────────────────────────────────
    st.subheader("Net Dollar Benefit at Each Decision Threshold")
    st.caption("Threshold = the minimum probability score required to flag a transaction as fraud.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=threshold_labels, y=net_benefit,
        mode="lines+markers", name="Net Benefit",
        line=dict(color="#2ecc71", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=threshold_labels, y=fraud_saved,
        mode="lines", name="Fraud Saved",
        line=dict(color="#3498db", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=threshold_labels, y=-fp_losses,
        mode="lines", name="False Positive Cost (negative)",
        line=dict(color="#e74c3c", width=1, dash="dot"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
    best_idx = int(np.argmax(net_benefit))
    fig.add_vline(
        x=threshold_labels[best_idx], line_dash="dash", line_color="#f39c12",
        annotation_text=f"Optimal threshold ≈ {threshold_labels[best_idx]:.2f}",
    )
    fig.update_layout(
        xaxis_title="Decision Threshold",
        yaxis_title="Dollar Impact ($)",
        height=420,
        legend=dict(x=0.6, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.subheader("At Optimal Threshold")
    opt_tp  = int(tp_arr[best_idx])
    opt_fp  = int(fp_arr[best_idx])
    opt_fn  = int(fn_arr[best_idx])
    opt_net = float(net_benefit[best_idx])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Optimal Threshold", f"{threshold_labels[best_idx]:.2f}")
    k2.metric("Fraud Cases Caught", f"{opt_tp} / {fraud_in_test}")
    k3.metric("False Alarms", f"{opt_fp}")
    k4.metric("Net Benefit (test set)", f"${opt_net:,.0f}")
    k5.metric("vs. No Model (loss)", f"${no_model_loss:,.0f}", delta=f"+${opt_net:,.0f} saved", delta_color="normal")

    st.markdown("---")

    # ── Chart 2: Precision vs Recall tradeoff ────────────────────────────────
    st.subheader("Precision vs Recall Tradeoff")
    st.caption("Higher recall = catch more fraud but more false alarms. The right balance depends on your cost inputs above.")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=recalls, y=precisions,
        mode="lines", name="Precision-Recall",
        line=dict(color="#9b59b6", width=2),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.1)",
    ))
    fig2.add_trace(go.Scatter(
        x=[recalls[best_idx]], y=[precisions[best_idx]],
        mode="markers", name="Optimal Point",
        marker=dict(color="#f39c12", size=12, symbol="star"),
    ))
    fig2.update_layout(
        xaxis_title="Recall (% of fraud caught)",
        yaxis_title="Precision (% of flags that are real fraud)",
        height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Scenario comparison table ─────────────────────────────────────────────
    st.subheader("Scenario Comparison")
    st.caption("How different threshold choices compare on business metrics.")

    scenario_rows = []
    for label, idx in [("Conservative (low threshold)", 5), ("Optimal", best_idx), ("Aggressive (high threshold)", -10)]:
        idx = min(max(idx, 0), len(tp_arr)-1)
        scenario_rows.append({
            "Strategy": label,
            "Threshold": f"{threshold_labels[idx]:.2f}",
            "Fraud Caught": f"{tp_arr[idx]} / {fraud_in_test}",
            "False Alarms": int(fp_arr[idx]),
            "Fraud $$ Saved": f"${tp_arr[idx] * fn_cost:,.0f}",
            "False Alarm Cost": f"${fp_arr[idx] * fp_cost:,.0f}",
            "Net Benefit": f"${net_benefit[idx]:,.0f}",
        })
    st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)

    st.info(
        "**Key insight:** The default 0.5 threshold is rarely optimal for imbalanced fraud detection. "
        "Tuning the threshold based on real business costs can significantly increase net dollar benefit."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: Live Predictor
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Live Predictor":
    st.title("Live Transaction Predictor")
    st.markdown("Manually enter transaction values or load a random sample from the dataset.")

    feature_names = metrics["feature_names"]

    tab1, tab2 = st.tabs(["Manual Input", "Batch from CSV"])

    with tab1:
        st.markdown("### Enter Transaction Details")

        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=149.62, step=0.01)
            time_val = st.number_input("Time (seconds from first transaction)", min_value=0, value=0)

        if st.button("Load Random Sample from Dataset"):
            df = load_data()
            sample = df.sample(1).iloc[0]
            st.session_state["sample"] = sample
            st.success("Random sample loaded — see values below.")

        # Build V1-V28 sliders
        v_vals = {}
        st.markdown("#### PCA Features (V1–V28)")
        cols = st.columns(4)
        for i, fname in enumerate([f for f in feature_names if f.startswith("V")]):
            default = float(st.session_state.get("sample", {}).get(fname, 0.0)) if "sample" in st.session_state else 0.0
            v_vals[fname] = cols[i % 4].number_input(fname, value=round(default, 4), format="%.4f", key=fname)

        if st.button("Predict", type="primary"):
            scaler = load_scaler()
            amount_scaled = scaler.fit_transform([[amount]])[0][0]
            time_scaled = scaler.fit_transform([[time_val]])[0][0]

            row = {**v_vals, "Amount_scaled": amount_scaled, "Time_scaled": time_scaled}
            input_df = pd.DataFrame([row])[feature_names]

            prob = model.predict_proba(input_df)[0][1]
            pred = int(prob >= 0.5)

            st.markdown("---")
            if pred == 1:
                st.error(f"FRAUD DETECTED — Confidence: {prob*100:.1f}%")
            else:
                st.success(f"Legitimate Transaction — Fraud probability: {prob*100:.2f}%")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Fraud Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if pred == 1 else "#2ecc71"},
                    "steps": [
                        {"range": [0, 30], "color": "#d5f5e3"},
                        {"range": [30, 70], "color": "#fdebd0"},
                        {"range": [70, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "value": 50},
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # SHAP explanation for this prediction
            st.subheader("Why did the model predict this?")
            explainer = load_explainer()
            shap_vals = explainer(input_df)
            values = shap_vals.values[0]
            shap_df_single = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": values,
            }).sort_values("SHAP Value", key=abs, ascending=False).head(10)
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df_single["SHAP Value"]]
            fig2 = go.Figure(go.Bar(
                x=shap_df_single["SHAP Value"],
                y=shap_df_single["Feature"],
                orientation="h",
                marker_color=colors,
            ))
            fig2.update_layout(
                xaxis_title="SHAP Value (red = toward FRAUD, green = toward LEGIT)",
                height=380,
                yaxis=dict(autorange="reversed"),
                margin=dict(t=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("### Upload a CSV for batch prediction")
        st.caption("CSV must have the same V1–V28, Amount, Time columns as the original dataset.")
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            batch_df = pd.read_csv(uploaded)
            scaler = load_scaler()
            batch_df["Amount_scaled"] = scaler.fit_transform(batch_df[["Amount"]])
            batch_df["Time_scaled"] = scaler.fit_transform(batch_df[["Time"]])
            batch_df.drop(columns=["Amount", "Time"], inplace=True, errors="ignore")
            if "Class" in batch_df.columns:
                batch_df.drop(columns=["Class"], inplace=True)

            preds = model.predict(batch_df[feature_names])
            probs = model.predict_proba(batch_df[feature_names])[:, 1]
            batch_df["Fraud_Probability"] = (probs * 100).round(2)
            batch_df["Prediction"] = ["FRAUD" if p == 1 else "Legit" for p in preds]

            fraud_n = (preds == 1).sum()
            st.metric("Fraudulent transactions found", f"{fraud_n} / {len(preds)}")
            st.dataframe(batch_df[["Fraud_Probability", "Prediction"]].head(200), use_container_width=True)

            csv = batch_df.to_csv(index=False)
            st.download_button("Download Results", csv, "fraud_predictions.csv", "text/csv")
