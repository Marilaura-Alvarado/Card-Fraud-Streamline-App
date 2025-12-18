import streamlit as st

def apply_style():
    st.markdown(
        """
        <style>
        /* =====================================================
           GLOBAL APP STYLE
        ===================================================== */

        .stApp {
            background: radial-gradient(circle at top left, #111827, #020617);
            color: #e5e7eb;
            font-family: "Inter", system-ui, sans-serif;
        }

        /* =====================================================
           SIDEBAR
        ===================================================== */

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #020617);
            border-right: 1px solid #1f2937;
            padding-top: 24px;
        }

        /* Sidebar text */
        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
            font-size: 15px;
        }

        /* Page links */
        section[data-testid="stSidebar"] a {
            display: block;
            padding: 10px 14px;
            margin: 6px 10px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 500;
            transition: background-color 0.15s ease;
        }

        /* Hover effect */
        section[data-testid="stSidebar"] a:hover {
            background-color: #1f2937;
        }

        /* Active page */
        section[data-testid="stSidebar"] a[aria-current="page"] {
            background-color: #2563eb;
            color: white !important;
            font-weight: 600;
        }

        /* Collapse button */
        button[kind="header"] {
            color: #9ca3af;
        }

        /* =====================================================
           MAIN LAYOUT
        ===================================================== */

        .main-container {
            max-width: 1150px;
            margin: auto;
            padding-top: 40px;
        }

        /* =====================================================
           CARDS
        ===================================================== */

        .card {
            background: linear-gradient(180deg, #020617, #020617);
            border-radius: 18px;
            padding: 32px;
            box-shadow: 0px 12px 30px rgba(0,0,0,0.6);
            border: 1px solid #1f2937;
            margin-bottom: 28px;
        }

        .card p {
            color: #cbd5f5;
            font-size: 0.95rem;
        }

        /* =====================================================
           HEADERS
        ===================================================== */

        h1, h2, h3 {
            color: #ffffff;
            font-weight: 700;
        }

        h4 {
            color: #93c5fd;
            font-weight: 600;
        }

        /* =====================================================
           KPI METRICS
        ===================================================== */

        div[data-testid="metric-container"] {
            background: linear-gradient(180deg, #020617, #020617);
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #1f2937;
            box-shadow: 0px 8px 24px rgba(0,0,0,0.5);
        }

        div[data-testid="metric-container"] label {
            color: #9ca3af !important;
        }

        div[data-testid="metric-container"] div {
            color: #ffffff !important;
            font-size: 1.7rem;
            font-weight: 700;
        }

        /* =====================================================
           TABLES
        ===================================================== */

        .stDataFrame {
            background: #020617;
            border-radius: 16px;
            padding: 12px;
            border: 1px solid #1f2937;
        }

        /* =====================================================
           BUTTONS
        ===================================================== */

        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border-radius: 10px;
            font-weight: 600;
            border: none;
            padding: 0.5em 1.2em;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
        }

        /* =====================================================
           ALERTS
        ===================================================== */

        .stAlert {
            border-left: 5px solid #3b82f6;
            background: #020617;
        }

        </style>
        """,
        unsafe_allow_html=True
    )
