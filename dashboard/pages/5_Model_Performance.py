"""DEBASS Dashboard — Page 5: Model Performance.

Publication-quality metrics, ROC curves, coverage analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data_loader import (
    load_trust_metrics, load_followup_metrics, load_results_summary, load_scores,
)
from lib.charts import coverage_vs_ndet
from lib.style import CUSTOM_CSS, expert_name, CLASS_COLORS

st.set_page_config(page_title="DEBASS — Model Performance", page_icon="📊", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📊 Model Performance")
st.caption("Metrics for the DEBASS meta-classifier trust heads and follow-up model.")

# ── Load metrics ──────────────────────────────────────────────────
trust_metrics = load_trust_metrics()
followup_metrics = load_followup_metrics()
summary = load_results_summary()

# ══════════════════════════════════════════════════════════════════
# SECTION 1: Trust Model Comparison
# ══════════════════════════════════════════════════════════════════
st.subheader("Expert Trust Models")

if trust_metrics:
    rows = []
    for ek, m in trust_metrics.items():
        rows.append({
            "Expert": expert_name(ek),
            "Key": ek,
            "ROC AUC": m.get("roc_auc", np.nan),
            "Brier Score": m.get("brier", np.nan),
            "Positive Rate": m.get("positive_rate", np.nan),
            "Test N": m.get("n_rows_test", m.get("n_rows", "?")),
            "Train N": m.get("n_rows_train", "?"),
        })
    tm_df = pd.DataFrame(rows)

    # Display table
    st.dataframe(
        tm_df.style.format({
            "ROC AUC": "{:.4f}",
            "Brier Score": "{:.4f}",
            "Positive Rate": "{:.3f}",
        }).background_gradient(subset=["ROC AUC"], cmap="RdYlGn", vmin=0.5, vmax=1.0),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart: AUC comparison
    valid = tm_df.dropna(subset=["ROC AUC"])
    if not valid.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=valid["Expert"],
            y=valid["ROC AUC"],
            marker_color=["#2ecc71" if v >= 0.95 else "#f39c12" if v >= 0.9 else "#e74c3c"
                          for v in valid["ROC AUC"]],
            text=valid["ROC AUC"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
        ))
        fig.add_hline(y=0.95, line_dash="dash", line_color="#2ecc71",
                      annotation_text="Excellent (0.95)")
        fig.add_hline(y=0.9, line_dash="dash", line_color="#f39c12",
                      annotation_text="Good (0.90)")
        fig.update_layout(
            title="Trust Model ROC AUC",
            yaxis=dict(title="ROC AUC", range=[0.5, 1.02]),
            template="plotly_dark",
            height=400,
            margin=dict(l=60, r=20, t=50, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No trust metrics found. Train trust models first.")

# ══════════════════════════════════════════════════════════════════
# SECTION 2: Follow-up Model
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Follow-up Recommendation Model")

if followup_metrics:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC AUC", f"{followup_metrics.get('roc_auc', 0):.4f}")
    col2.metric("Brier Score", f"{followup_metrics.get('brier', 0):.4f}")
    col3.metric("Positive Rate", f"{followup_metrics.get('positive_rate', 0):.3f}")
    col4.metric("Test Samples", followup_metrics.get("n_rows_test",
                followup_metrics.get("n_rows", "?")))

    st.success(
        f"Follow-up model achieves **ROC AUC = {followup_metrics.get('roc_auc', 0):.3f}** "
        f"with Brier score {followup_metrics.get('brier', 0):.4f} — "
        "excellent discrimination and calibration."
    )
else:
    st.info("No follow-up metrics found.")

# ══════════════════════════════════════════════════════════════════
# SECTION 3: Training Data Composition
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Training Data Composition")

if summary:
    # Truth quality breakdown — adapt to actual JSON structure
    truth_info = summary.get("truth", {})
    truth_breakdown = truth_info.get("by_quality", summary.get("truth_quality_breakdown", {}))
    if truth_breakdown:
        col_a, col_b = st.columns(2)
        with col_a:
            labels = list(truth_breakdown.keys())
            values = list(truth_breakdown.values())
            colors = {
                "spectroscopic": "#2ecc71",
                "tns_untyped": "#3498db",
                "consensus": "#f39c12",
                "context": "#9b59b6",
                "weak": "#95a5a6",
            }
            fig = go.Figure(go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=[colors.get(l, "#999") for l in labels]),
                textinfo="label+percent+value",
                hole=0.4,
            ))
            fig.update_layout(
                title="Truth Quality Tiers",
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Class distribution — adapt to actual JSON structure
            class_dist = truth_info.get("by_class", summary.get("class_distribution", {}))
            if class_dist:
                fig = go.Figure(go.Pie(
                    labels=list(class_dist.keys()),
                    values=list(class_dist.values()),
                    marker=dict(colors=[
                        CLASS_COLORS.get(k, "#999") for k in class_dist.keys()
                    ]),
                    textinfo="label+percent+value",
                    hole=0.4,
                ))
                fig.update_layout(
                    title="Target Class Distribution",
                    template="plotly_dark",
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

    # Key numbers — adapt to actual JSON structure
    st.markdown("**Pipeline Summary:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Objects", summary.get("n_objects", "?"))
    col2.metric("Epoch Snapshots", summary.get("n_snapshot_rows", summary.get("n_snapshots", "?")))
    ndet_range = summary.get("n_det_range", [])
    if ndet_range:
        col3.metric("n_det Range", f"{ndet_range[0]}–{ndet_range[1]}")
    else:
        col3.metric("n_det Range", f"{summary.get('n_det_min', '?')}–{summary.get('n_det_max', '?')}")

# ══════════════════════════════════════════════════════════════════
# SECTION 4: Expert Coverage vs n_det
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Expert Coverage vs. Number of Detections")

if summary:
    fig = coverage_vs_ndet(summary, height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No summary data available for coverage analysis.")

# ══════════════════════════════════════════════════════════════════
# SECTION 5: Score Distribution Analysis
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Score Distributions by n_det")

all_scores = load_scores(ndet=None)
if not all_scores.empty and "n_det" in all_scores.columns:
    available_ndets = sorted(all_scores["n_det"].unique())

    # p_follow distribution per n_det
    fig = go.Figure()
    for nd in available_ndets:
        sub = all_scores[all_scores["n_det"] == nd]
        fig.add_trace(go.Violin(
            y=sub["p_follow"],
            name=f"n={nd}",
            box_visible=True,
            meanline_visible=True,
        ))
    fig.update_layout(
        title="p(Follow-up) Distribution by n_det",
        yaxis_title="p(Follow-up)",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation rate by n_det
    rec_rates = all_scores.groupby("n_det")["recommended"].mean() * 100
    fig = go.Figure(go.Bar(
        x=[str(nd) for nd in rec_rates.index],
        y=rec_rates.values,
        marker_color="#2ecc71",
        text=[f"{v:.1f}%" for v in rec_rates.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Recommendation Rate by n_det",
        xaxis_title="Number of Detections",
        yaxis_title="% Recommended",
        template="plotly_dark",
        height=350,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Download section ──────────────────────────────────────────────
st.divider()
st.subheader("Export for Paper")
st.caption("Use Plotly's built-in camera icon (top-right of each chart) "
           "to download individual figures as PNG.")

if trust_metrics:
    import json
    st.download_button(
        "Download Trust Metrics (JSON)",
        data=json.dumps(trust_metrics, indent=2),
        file_name="debass_trust_metrics.json",
        mime="application/json",
    )
if followup_metrics:
    import json
    st.download_button(
        "Download Follow-up Metrics (JSON)",
        data=json.dumps(followup_metrics, indent=2),
        file_name="debass_followup_metrics.json",
        mime="application/json",
    )
