import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from xgboost import XGBClassifier
from style import apply_style

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(layout="wide")
apply_style()

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("Model Evaluation")

st.markdown(
    """
    This page evaluates the **final XGBoost fraud detection model**
    using classification metrics and visual diagnostics.
    """
)

# ===============================
# LOAD DATA
# ===============================
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").iloc[:, 0]
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").iloc[:, 0]

# ===============================
# TRAIN MODEL
# ===============================
model = XGBClassifier(
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# ===============================
# PREDICTIONS
# ===============================
threshold = 0.20
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= threshold).astype(int)

# ===============================
# METRICS
# ===============================
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)

cm = confusion_matrix(y_test, preds)

# ===============================
# KPI SECTION
# ===============================
st.subheader("Key Performance Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", round(precision, 3))
c2.metric("Recall (Fraud)", round(recall, 3))
c3.metric("F1-score", round(f1, 3))
c4.metric("ROC-AUC", round(roc_auc, 3))

st.markdown("---")

# ===============================
# VISUAL STYLE FIX
# ===============================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black"
})

# ===============================
# MODEL DIAGNOSTICS
# ===============================
st.subheader("Model Diagnostics")

col1, col2 = st.columns([1, 1], gap="large")

# ---------- Confusion Matrix ----------
with col1:
    fig_cm, ax_cm = plt.subplots(figsize=(3.2, 3.2), facecolor="white")
    ax_cm.set_facecolor("white")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legit", "Fraud"]
    )

    disp.plot(
        ax=ax_cm,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax_cm.tick_params(labelsize=9)
    ax_cm.set_title("Confusion Matrix", fontsize=11)

    st.pyplot(fig_cm, clear_figure=True, use_container_width=False)

# ---------- ROC Curve ----------
with col2:
    fpr, tpr, _ = roc_curve(y_test, probs)

    fig_roc, ax_roc = plt.subplots(figsize=(3.6, 3.0), facecolor="white")
    ax_roc.set_facecolor("white")

    ax_roc.plot(
        fpr,
        tpr,
        color="#2563eb",
        linewidth=2,
        label=f"AUC = {roc_auc:.3f}"
    )

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax_roc.set_xlabel("False Positive Rate", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate", fontsize=9)
    ax_roc.legend(fontsize=8)
    ax_roc.set_title("ROC Curve", fontsize=11)

    st.pyplot(fig_roc, clear_figure=True, use_container_width=False)

st.markdown("---")

# ===============================
# INTERPRETATION
# ===============================
st.subheader("Interpretation")

st.markdown(
    f"""
    - The model achieves **high fraud recall ({recall:.2%})**, which is critical for fraud loss prevention.
    - Precision is lower due to extreme class imbalance, which is expected in real-world fraud detection.
    - The decision threshold (**{threshold}**) prioritizes catching fraud over minimizing false positives.
    - A high ROC-AUC confirms strong ranking performance across thresholds.
    """
)

st.markdown('</div>', unsafe_allow_html=True)
