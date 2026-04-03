"""
DEBASS Meta-Classifier Dashboard
=================================
Transient alert triage for spectroscopic follow-up prioritisation.

Launch:  streamlit run dashboard/app.py
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="DEBASS — Transient Alert Dashboard",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar branding ──────────────────────────────────────────────
st.sidebar.title("🔭 DEBASS")
st.sidebar.caption(
    "Detection-Based Astronomical\n"
    "Source Spectroscopic Meta-Classifier"
)
st.sidebar.divider()
st.sidebar.markdown(
    "Fuses broker outputs (ALeRCE, Fink, Lasair) "
    "with lightcurve features to prioritise "
    "spectroscopic follow-up after 3–5 detections."
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Pipeline**: ZTF + LSST  \n"
    "**Experts**: 16 registered  \n"
    "**Model**: LightGBM trust heads"
)

# ── Landing / home ────────────────────────────────────────────────
st.title("🔭 DEBASS — Transient Alert Dashboard")
st.markdown(
    "Welcome to the DEBASS meta-classifier dashboard. "
    "Use the sidebar to navigate between views."
)

col1, col2, col3 = st.columns(3)
col1.page_link("pages/1_Triage.py", label="🎯 Triage", icon="🎯")
col2.page_link("pages/2_Expert_Matrix.py", label="🧠 Expert Matrix", icon="🧠")
col3.page_link("pages/3_Lightcurve.py", label="📈 Lightcurves", icon="📈")

col4, col5, _ = st.columns(3)
col4.page_link("pages/4_Object_Detail.py", label="🔍 Object Detail", icon="🔍")
col5.page_link("pages/5_Model_Performance.py", label="📊 Model Performance", icon="📊")
