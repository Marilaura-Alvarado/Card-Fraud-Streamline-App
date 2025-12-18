import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from style import apply_style

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(layout="wide")
apply_style()

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("Transaction Simulator")

st.markdown(
    """
    Simulate a **single transaction** and observe how the fraud detection
    system behaves **in real time**.
    """
)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").iloc[:, 0]
    return X_train, y_train

X_train, y_train = load_data()

# ===============================
# TRAIN MODEL
# ===============================
@st.cache_resource
def train_model(X_train, y_train):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        scale_pos_weight=pos_weight,
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.title("Simulation Controls")

threshold = st.sidebar.slider(
    "Fraud decision threshold",
    min_value=0.05,
    max_value=0.60,
    value=0.20,
    step=0.01
)

FN_COST = 10
FP_COST = 1

# ===============================
# INPUT SECTION
# ===============================
st.subheader("Enter Transaction Features")


st.info(
    """
    **How to test**
    
    • Values close to **0** → **LEGIT**  
    • To simulate **FRAUD**:
      - Increase **Amount_scaled** (≈ 4–6)
      - Set a few features (**V10, V12, V14, V17**) to strong **negative values**
    
    **Fraud decision threshold**
    
    • Lower threshold → **more FRAUD detections** (higher recall, more false positives)  
    • Higher threshold → **fewer FRAUD detections** (lower recall, fewer false positives)  
    • Default (**0.20**) balances fraud prevention and customer experience
    """
)
with st.form("transaction_form"):
    inputs = {}

    cols = st.columns(3)
    for i, col in enumerate(X_train.columns):
        with cols[i % 3]:
            inputs[col] = st.number_input(
                label=col,
                value=float(X_train[col].median())
            )

    submitted = st.form_submit_button("Run Fraud Check")


if submitted:
    tx = pd.DataFrame([inputs], columns=X_train.columns)
    prob = model.predict_proba(tx)[0, 1]
    is_fraud = prob >= threshold

    st.markdown("---")
    st.subheader("Simulation Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability", round(prob, 4))
    c2.metric("Decision Threshold", threshold)
    c3.metric(
        "Decision",
        "FRAUD 🚨" if is_fraud else "LEGIT ✅"
    )

    st.markdown("---")
    st.subheader("Business Interpretation")

    if is_fraud:
        st.error(
            f"""
            **Transaction flagged as FRAUD**
            
            - Routed to manual review or blocked
            - Potential fraud loss avoided: **{FN_COST} units**
            """
        )
    else:
        st.success(
            f"""
            **Transaction approved**
            
            - No customer friction
            - False positive cost avoided: **{FP_COST} unit**
            """
        )

st.markdown('</div>', unsafe_allow_html=True)
