"""DEBASS Dashboard — Page 2: Expert Confidence Matrix.

Heatmap showing all experts' opinions per object, with trust overlay.
"""
import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data_loader import load_scores, get_all_expert_keys, load_trust_metrics
from lib.charts import expert_heatmap, expert_agreement_bars
from lib.style import CUSTOM_CSS, expert_name

st.set_page_config(page_title="DEBASS — Expert Matrix", page_icon="🧠", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🧠 Expert Confidence Matrix")

# ── Controls ──────────────────────────────────────────────────────
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    ndet = st.selectbox("Number of detections", [3, 5, 10, 15, 20], index=0)
with col_c2:
    color_by = st.selectbox("Color by class", ["p(SNIa)", "p(Non-Ia)", "p(Other)"])
with col_c3:
    max_obj = st.slider("Max objects shown", 10, 100, 40)

prefix_map = {
    "p(SNIa)": "p_snia__",
    "p(Non-Ia)": "p_nonIa__",
    "p(Other)": "p_other__",
}

# ── Load data ─────────────────────────────────────────────────────
df = load_scores(ndet=ndet)
expert_keys = get_all_expert_keys()

if df.empty:
    st.warning("No scored data found for this n_det.")
    st.stop()

# Filter to experts that have at least some availability
active_experts = []
for ek in expert_keys:
    col = f"avail__{ek}"
    if col in df.columns and df[col].any():
        active_experts.append(ek)

if not active_experts:
    st.warning("No experts have data at this epoch.")
    st.stop()

# ── Heatmap ───────────────────────────────────────────────────────
st.subheader(f"Top {min(max_obj, len(df))} Candidates × {len(active_experts)} Active Experts")
fig = expert_heatmap(
    df, active_experts,
    value_col_prefix=prefix_map[color_by],
    max_objects=max_obj,
    height=max(400, max_obj * 16),
)
st.plotly_chart(fig, use_container_width=True)

# ── Expert summary table ─────────────────────────────────────────
st.subheader("Expert Summary")
metrics = load_trust_metrics()

rows = []
for ek in active_experts:
    avail_col = f"avail__{ek}"
    n_avail = int(df[avail_col].sum()) if avail_col in df.columns else 0
    coverage = 100 * n_avail / max(len(df), 1)
    trust_col = f"trust__{ek}"
    mean_trust = df[trust_col].dropna().mean() if trust_col in df.columns else np.nan

    # Metrics from trained models
    m = metrics.get(ek, {})
    auc = m.get("roc_auc", np.nan)
    brier = m.get("brier", np.nan)

    rows.append({
        "Expert": expert_name(ek),
        "Coverage %": f"{coverage:.1f}",
        "Mean Trust": f"{mean_trust:.4f}" if not np.isnan(mean_trust) else "N/A",
        "AUC": f"{auc:.3f}" if not np.isnan(auc) else "—",
        "Brier": f"{brier:.4f}" if not np.isnan(brier) else "—",
        "Trained?": "Yes" if m else "No",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Per-object agreement breakdown ────────────────────────────────
st.divider()
st.subheader("Per-Object Expert Agreement")
selected = st.selectbox(
    "Select object",
    df["object_id"].tolist(),
    index=0,
)
if selected:
    row = df[df["object_id"] == selected].iloc[0]
    experts_raw = row.get("_experts_raw", {})
    if experts_raw:
        fig = expert_agreement_bars(experts_raw)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No raw expert data available for this object.")
