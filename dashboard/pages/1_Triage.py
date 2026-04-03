"""DEBASS Dashboard — Page 1: Daily Triage.

Ranked candidate table for spectroscopic follow-up prioritisation.
"""
import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data_loader import load_scores, get_all_expert_keys
from lib.style import CUSTOM_CSS, rec_color, rec_label, expert_name

st.set_page_config(page_title="DEBASS — Triage", page_icon="🎯", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🎯 Daily Triage — Follow-up Candidates")

# ── Sidebar filters ───────────────────────────────────────────────
st.sidebar.header("Filters")
all_ndet = [3, 4, 5, 10, 15, 20]
ndet_choice = st.sidebar.selectbox("Detections (n_det)", [None] + all_ndet,
                                    format_func=lambda x: "All" if x is None else str(x))
min_pfollow = st.sidebar.slider("Min p(follow-up)", 0.0, 1.0, 0.0, 0.05)
only_rec = st.sidebar.checkbox("Show only recommended", value=False)

# ── Load data ─────────────────────────────────────────────────────
df = load_scores(ndet=ndet_choice)

if df.empty:
    st.warning("No scored data found. Run the scoring pipeline first, or check `published_results/`.")
    st.stop()

# Apply filters
if min_pfollow > 0:
    df = df[df["p_follow"] >= min_pfollow]
if only_rec:
    df = df[df["recommended"] == True]

# ── Summary metrics ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Candidates", len(df))
n_rec = int(df["recommended"].sum())
col2.metric("Recommended", f"{n_rec} ({100*n_rec/max(len(df),1):.0f}%)")
med_snia = df['p_snia'].dropna().median()
med_follow = df['p_follow'].dropna().median()
col3.metric("Median p(SNIa)", f"{med_snia:.3f}" if not np.isnan(med_snia) else "—")
col4.metric("Median p(follow)", f"{med_follow:.3f}" if not np.isnan(med_follow) else "—")

# ── Main table ────────────────────────────────────────────────────
st.subheader(f"Candidates ({len(df)} objects)")

display_cols = ["object_id", "n_det", "p_snia", "p_nonIa", "p_other",
                "p_follow", "recommended", "n_experts", "top_expert", "top_trust"]
df_show = df[display_cols].copy()

# Clean NaN → readable placeholders
df_show["top_expert"] = df_show["top_expert"].replace("", "—").map(
    lambda x: expert_name(x) if x and x != "—" else "—"
)
df_show["p_snia"] = df_show["p_snia"].fillna(0)
df_show["p_nonIa"] = df_show["p_nonIa"].fillna(0)
df_show["p_other"] = df_show["p_other"].fillna(0)

df_show = df_show.rename(columns={
    "object_id": "Object ID",
    "n_det": "n_det",
    "p_snia": "p(SNIa)",
    "p_nonIa": "p(non-Ia)",
    "p_other": "p(Other)",
    "p_follow": "p(Follow)",
    "recommended": "Rec?",
    "n_experts": "#Experts",
    "top_expert": "Top Expert",
    "top_trust": "Top Trust",
})

# Color the recommendation column
def highlight_rec(row):
    if row["Rec?"]:
        return ["background-color: rgba(46, 204, 113, 0.15)"] * len(row)
    elif row["p(Follow)"] >= 0.3:
        return ["background-color: rgba(243, 156, 18, 0.1)"] * len(row)
    return [""] * len(row)

def _fmt_or_dash(v, fmt="{:.3f}"):
    """Format a float or return — for NaN."""
    try:
        if pd.isna(v):
            return "—"
        return fmt.format(v)
    except (TypeError, ValueError):
        return "—"

styled = df_show.style.apply(highlight_rec, axis=1).format({
    "p(SNIa)": "{:.3f}",
    "p(non-Ia)": "{:.3f}",
    "p(Other)": "{:.3f}",
    "p(Follow)": "{:.3f}",
    "Top Trust": lambda v: _fmt_or_dash(v, "{:.4f}"),
})

st.dataframe(styled, use_container_width=True, height=600)

# ── Distribution plots ────────────────────────────────────────────
st.subheader("Score Distributions")
col_a, col_b = st.columns(2)

with col_a:
    import plotly.express as px
    fig = px.histogram(df, x="p_follow", nbins=40, title="p(Follow-up) Distribution",
                       color_discrete_sequence=["#2ecc71"],
                       template="plotly_dark")
    fig.add_vline(x=0.5, line_dash="dash", line_color="#e74c3c",
                  annotation_text="threshold=0.5")
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    fig = px.histogram(df, x="p_snia", nbins=40, title="p(SNIa) Distribution",
                       color_discrete_sequence=["#e74c3c"],
                       template="plotly_dark")
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

# ── Quick-look: select object to jump to detail ──────────────────
st.divider()
selected = st.selectbox("Jump to Object Detail →", df["object_id"].tolist(),
                         index=0 if not df.empty else None)
if selected:
    st.info(f"Selected **{selected}** — go to the 🔍 Object Detail page to see the full analysis.")
    st.session_state["selected_object"] = selected
