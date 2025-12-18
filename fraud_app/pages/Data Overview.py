import streamlit as st
import pandas as pd

from style import apply_style

st.set_page_config(layout="wide")
apply_style()
st.title("Data Overview")

@st.cache_data
def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").iloc[:, 0]
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").iloc[:, 0]
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Train samples", X_train.shape[0])
c2.metric("Test samples", X_test.shape[0])
c3.metric("Features", X_train.shape[1])
c4.metric("Fraud rate", round(y_train.mean(), 4))

st.markdown(
    """
    **Main challenge**
    - Fraud is extremely rare
    - Accuracy is misleading
    - Recall and business cost are critical
    """
)
