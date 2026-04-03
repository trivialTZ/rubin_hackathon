"""DEBASS Dashboard — Page 3: Lightcurve Explorer.

Interactive multi-band lightcurve viewer with feature annotations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data_loader import (
    load_lightcurve, list_lightcurve_ids, load_scores, load_gold_snapshots,
)
from lib.charts import lightcurve_plot
from lib.style import CUSTOM_CSS, BAND_COLORS

st.set_page_config(page_title="DEBASS — Lightcurves", page_icon="📈", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📈 Lightcurve Explorer")

# ── Object selector ───────────────────────────────────────────────
lc_ids = list_lightcurve_ids()
scored = load_scores()

if not lc_ids:
    st.warning("No lightcurve files found in `data/lightcurves/`. "
               "Run `fetch_lightcurves.py` first.")
    st.stop()

# Prefer object from session state (e.g. from Triage page link)
default_idx = 0
if "selected_object" in st.session_state:
    sel = st.session_state["selected_object"]
    if sel in lc_ids:
        default_idx = lc_ids.index(sel)

selected = st.selectbox(
    "Select object",
    lc_ids,
    index=default_idx,
    help="Objects with cached lightcurve data",
)

if not selected:
    st.stop()

# ── Load lightcurve ───────────────────────────────────────────────
lc_data = load_lightcurve(selected)
if lc_data is None:
    st.error(f"Could not load lightcurve for {selected}")
    st.stop()

# Parse detections — handle different JSON formats
detections = []
if isinstance(lc_data, list):
    detections = lc_data
elif isinstance(lc_data, dict):
    # May have "detections" key, or "prv_candidates" + "candidate"
    if "detections" in lc_data:
        detections = lc_data["detections"]
    elif "prv_candidates" in lc_data:
        cands = lc_data.get("prv_candidates", []) or []
        if lc_data.get("candidate"):
            cands.append(lc_data["candidate"])
        detections = cands

# Normalise detection fields
norm_dets = []
for d in detections:
    if not isinstance(d, dict):
        continue
    rec = {}
    # JD
    rec["jd"] = d.get("jd") or d.get("mjd") or d.get("midpointtai") or 0
    # Band
    band_raw = d.get("band") or d.get("fid") or d.get("filter") or d.get("filterName", "")
    fid_map = {1: "g", 2: "r", 3: "i", "1": "g", "2": "r", "3": "i"}
    rec["band"] = fid_map.get(band_raw, str(band_raw).strip().lower()[:1])
    # Mag
    rec["mag"] = d.get("mag") or d.get("magpsf") or d.get("psfFlux")
    rec["magerr"] = d.get("magerr") or d.get("sigmapsf") or d.get("psfFluxErr") or 0.1
    if rec["mag"] is not None:
        try:
            rec["mag"] = float(rec["mag"])
            rec["magerr"] = float(rec["magerr"])
            norm_dets.append(rec)
        except (ValueError, TypeError):
            pass

# ── Plot ──────────────────────────────────────────────────────────
st.subheader(f"Lightcurve: {selected}")

n_det_markers = [3, 5, 10]
fig = lightcurve_plot(
    norm_dets,
    title=f"{selected} — {len(norm_dets)} detections",
    height=500,
    n_det_markers=n_det_markers,
)
st.plotly_chart(fig, use_container_width=True)

# ── Detection table ───────────────────────────────────────────────
with st.expander(f"Detection table ({len(norm_dets)} rows)", expanded=False):
    if norm_dets:
        det_df = pd.DataFrame(norm_dets).sort_values("jd")
        det_df.index = range(1, len(det_df) + 1)
        det_df.index.name = "#"
        st.dataframe(det_df, use_container_width=True)

# ── Feature sidebar ───────────────────────────────────────────────
st.divider()
st.subheader("Lightcurve Features")

if norm_dets:
    det_df = pd.DataFrame(norm_dets).sort_values("jd")
    mags = det_df["mag"].dropna()
    jds = det_df["jd"].dropna()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Detections", len(det_df))
    col2.metric("Time Baseline", f"{jds.max() - jds.min():.1f} d" if len(jds) > 1 else "—")
    col3.metric("Mag Range", f"{mags.min():.2f} – {mags.max():.2f}" if len(mags) > 0 else "—")
    col4.metric("Mag Std", f"{mags.std():.3f}" if len(mags) > 1 else "—")

    # Per-band counts
    band_counts = det_df["band"].value_counts()
    st.markdown("**Detections per band:** " + "  ".join(
        f"`{b}: {int(band_counts.get(b, 0))}`" for b in "ugrizy"
        if band_counts.get(b, 0) > 0
    ))

    # Quick colour estimate
    if len(det_df) > 1:
        g_mags = det_df[det_df["band"] == "g"]["mag"]
        r_mags = det_df[det_df["band"] == "r"]["mag"]
        if len(g_mags) > 0 and len(r_mags) > 0:
            color_gr = g_mags.mean() - r_mags.mean()
            st.metric("Mean g-r colour", f"{color_gr:.3f}")

# ── Score context (if available) ──────────────────────────────────
if not scored.empty:
    obj_scores = scored[scored["object_id"] == selected]
    if not obj_scores.empty:
        st.divider()
        st.subheader("Classification Scores")
        for _, row in obj_scores.iterrows():
            nd = int(row["n_det"])
            rec_emoji = "🟢" if row["recommended"] else "⚪"
            st.markdown(
                f"**n_det={nd}**: p(SNIa)={row['p_snia']:.3f}, "
                f"p(follow)={row['p_follow']:.3f} {rec_emoji}"
            )
