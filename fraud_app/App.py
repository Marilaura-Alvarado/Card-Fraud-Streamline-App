import streamlit as st

from style import apply_style

st.set_page_config(layout="wide")
apply_style()

st.title("Fraud Detection System")

st.markdown(
    """
    ### Business-driven fraud detection using XGBoost

    This application demonstrates a **production-ready fraud detection pipeline**
    optimized for **financial impact**, not just ML metrics.

    **Key highlights**
    - Extreme class imbalance handling
    - Cost-aware threshold optimization
    - Business-driven model selection
    - Deployment-ready pipeline

    👉 Navigate using the menu on the left.
    """
)

st.info("Final model: XGBoost with optimized business threshold")
