import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("LLM_DRIFT_API_URL", "http://127.0.0.1:8000")

st.title("LLM Drift Detector Dashboard")
st.markdown(
    """
This dashboard visualizes data drift in production queries submitted to your LLMs compared to a baseline dataset.
"""
)

# Sidebar inputs to simulate requests
st.sidebar.header("Simulate Production Request")
new_query = st.sidebar.text_input("Query Text", "Explain machine learning simply.")
if st.sidebar.button("Log Query"):
    try:
        res = requests.post(f"{API_BASE_URL}/log_query", json={"text": new_query})
        if res.status_code == 200:
            st.sidebar.success(f"Logged! Total target queries: {res.json().get('current_count')}")
        else:
            st.sidebar.error(f"API returned an error: {res.text}")
    except Exception as exc:
        st.sidebar.error(f"Connection error: {exc}")

# Main panel to calculate & visualize drift
st.subheader("Drift Performance Metrics")
if st.button("Calculate Current Drift"):
    try:
        res = requests.get(f"{API_BASE_URL}/calculate_drift")
        if res.status_code == 200:
            data = res.json()
            score = float(data.get("drift_score", 0.0))
            status = data.get("status", "")

            if status == "insufficient_data":
                st.warning("No target queries logged yet. Please log some queries via the sidebar first!")
            else:
                st.metric(label="Embedding Drift (1 - Cosine Similarity)", value=f"{score:.4f}")

                threshold = 0.25
                if score > threshold:
                    st.error(f"🚨 ALERT: High Drift detected (exceeds threshold of {threshold})!")
                else:
                    st.success("✅ Status Normal: Semantic distribution matches baseline reference.")
        else:
            st.error(f"Could not retrieve drift scores from backend API: {res.text}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")
