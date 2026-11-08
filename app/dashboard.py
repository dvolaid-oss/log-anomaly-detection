"""Streamlit dashboard (built in Stage 7).

Run with:  streamlit run app/dashboard.py
"""

import streamlit as st

st.set_page_config(page_title="Log Anomaly Detection", page_icon="🔎")
st.title("Log Anomaly Detection")
st.info("Coming in Stage 7: upload logs → see anomalies → read why each one was flagged.")
