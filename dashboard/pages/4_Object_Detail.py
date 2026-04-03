"""DEBASS Dashboard — Page 4: Object Detail.

Single-object deep dive: ensemble verdict, expert cards, lightcurve, context.
"""
import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data_loader import (
    load_scores, load_lightcurve, load_truth, list_lightcurve_ids,
)
from lib.charts import (
    ternary_donut, expert_agreement_bars, lightcurve_plot, temporal_evolution,
)
from lib.style import (
    CUSTOM_CSS, CLASS_COLORS, STATUS_COLORS, expert_name, rec_label, rec_color,
)

st.set_page_config(page_title="DEBASS — Object Detail", page_icon="🔍", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🔍 Object Detail")

# ── Load all scores (all n_det) ───────────────────────────────────
all_scores = load_scores(ndet=None)
if all_scores.empty:
    st.warning("No scored data found.")
    st.stop()

object_ids = sorted(all_scores["object_id"].unique())

# Pick from session state or selector
default_idx = 0
if "selected_object" in st.session_state:
    sel = st.session_state["selected_object"]
    if sel in object_ids:
        default_idx = object_ids.index(sel)

selected = st.selectbox("Select object", object_ids, index=default_idx)
if not selected:
    st.stop()
st.session_state["selected_object"] = selected

obj_df = all_scores[all_scores["object_id"] == selected].sort_values("n_det")

# Use the row with the MOST detections for the main display
latest = obj_df.iloc[-1]

# ══════════════════════════════════════════════════════════════════
# SECTION 1: Ensemble Verdict
# ══════════════════════════════════════════════════════════════════
st.divider()
p_follow = latest["p_follow"]
recommended = latest["recommended"]

verdict_color = rec_color(p_follow)
verdict_text = rec_label(p_follow)

col_v1, col_v2 = st.columns([2, 1])
with col_v1:
    st.markdown(f"### {selected}")
    st.markdown(f"**n_det = {int(latest['n_det'])}** &nbsp; | &nbsp; "
                f"JD = {latest.get('alert_jd', '?')}")

    # Large verdict
    if recommended:
        st.success(f"**{verdict_text}** for spectroscopic follow-up — "
                   f"p(follow) = **{p_follow:.3f}**")
    elif p_follow >= 0.3:
        st.warning(f"**{verdict_text}** — p(follow) = **{p_follow:.3f}**")
    else:
        st.info(f"**{verdict_text}** — p(follow) = **{p_follow:.3f}**")

    # Quick stats — handle NaN gracefully
    def _fmt(v):
        return f"{v:.3f}" if not np.isnan(v) else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("p(SNIa)", _fmt(latest['p_snia']))
    c2.metric("p(non-Ia)", _fmt(latest['p_nonIa']))
    c3.metric("p(Other)", _fmt(latest['p_other']))
    c4.metric("# Experts", int(latest["n_experts"]))

with col_v2:
    # For donut chart, replace NaN with 0
    _ps = 0 if np.isnan(latest["p_snia"]) else latest["p_snia"]
    _pn = 0 if np.isnan(latest["p_nonIa"]) else latest["p_nonIa"]
    _po = 0 if np.isnan(latest["p_other"]) else latest["p_other"]
    fig = ternary_donut(
        _ps, _pn, _po,
        title="Ensemble Classification",
        height=250,
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 2: Expert Opinion Cards
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Expert Opinions")

experts_raw = latest.get("_experts_raw", {})
if not experts_raw:
    st.info("No per-expert data available.")
else:
    # Determine ensemble consensus class
    ens_probs = {"snia": latest["p_snia"], "nonIa_snlike": latest["p_nonIa"],
                 "other": latest["p_other"]}
    ens_class = max(ens_probs, key=ens_probs.get)

    # Grid of expert cards (3 per row)
    available = {k: v for k, v in experts_raw.items() if v.get("available")}
    unavailable = {k: v for k, v in experts_raw.items() if not v.get("available")}

    if available:
        cols = st.columns(3)
        for i, (ek, ev) in enumerate(sorted(available.items())):
            proj = ev.get("projected", {})
            trust = ev.get("trust")
            mapped_cls = ev.get("mapped_pred_class", "?")
            exactness = ev.get("exactness", "?")
            p_snia = proj.get("p_snia", 0)

            # Agreement with ensemble
            agrees = (mapped_cls == ens_class)
            border_color = "#2ecc71" if agrees else "#e74c3c"
            cls_color = CLASS_COLORS.get(mapped_cls, "#95a5a6")

            with cols[i % 3]:
                trust_str = f"{trust:.4f}" if trust is not None else "N/A"
                # Darken "other" gray for readability on light backgrounds
                dot_color = cls_color if cls_color != "#95a5a6" else "#636e72"
                st.markdown(
                    f'<div style="background:#f7f8fa; border-radius:8px; '
                    f'padding:14px; margin-bottom:8px; '
                    f'border-left:4px solid {border_color}; '
                    f'box-shadow:0 1px 3px rgba(0,0,0,0.06)">'
                    f'<b style="color:#1a1a2e">{expert_name(ek)}</b> &nbsp; '
                    f'<span style="color:#555">Trust: {trust_str}</span><br>'
                    f'<span style="color:{dot_color}; font-size:1.1em; font-weight:600">'
                    f'● {mapped_cls}</span> &nbsp; '
                    f'<span style="color:#333">(p={p_snia:.3f} SNIa)</span><br>'
                    f'<span style="color:#777; font-size:0.85em">'
                    f'Exactness: {exactness}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Collapsed: unavailable experts
    if unavailable:
        with st.expander(f"Unavailable experts ({len(unavailable)})"):
            for ek, ev in sorted(unavailable.items()):
                reason = ev.get("reason", "Unknown")
                st.markdown(f"**{expert_name(ek)}**: {reason}")

    # Agreement bar chart
    fig = expert_agreement_bars(experts_raw, height=350)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 3: Lightcurve
# ══════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Lightcurve")

lc_data = load_lightcurve(selected)
if lc_data is not None:
    # Parse detections (same logic as Lightcurve page)
    detections = []
    if isinstance(lc_data, list):
        detections = lc_data
    elif isinstance(lc_data, dict):
        if "detections" in lc_data:
            detections = lc_data["detections"]
        elif "prv_candidates" in lc_data:
            cands = lc_data.get("prv_candidates", []) or []
            if lc_data.get("candidate"):
                cands.append(lc_data["candidate"])
            detections = cands

    norm_dets = []
    fid_map = {1: "g", 2: "r", 3: "i", "1": "g", "2": "r", "3": "i"}
    for d in detections:
        if not isinstance(d, dict):
            continue
        jd = d.get("jd") or d.get("mjd") or d.get("midpointtai")
        band_raw = d.get("band") or d.get("fid") or d.get("filter") or d.get("filterName", "")
        mag = d.get("mag") or d.get("magpsf") or d.get("psfFlux")
        magerr = d.get("magerr") or d.get("sigmapsf") or d.get("psfFluxErr") or 0.1
        if jd is not None and mag is not None:
            try:
                norm_dets.append({
                    "jd": float(jd),
                    "band": fid_map.get(band_raw, str(band_raw).strip().lower()[:1]),
                    "mag": float(mag),
                    "magerr": float(magerr),
                })
            except (ValueError, TypeError):
                pass

    if norm_dets:
        fig = lightcurve_plot(norm_dets, title=selected, height=400,
                              n_det_markers=[3, 5, 10])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Lightcurve file exists but no valid detections could be parsed.")
else:
    st.info("No cached lightcurve. Run `fetch_lightcurves.py` for this object.")

# ══════════════════════════════════════════════════════════════════
# SECTION 4: Temporal Evolution
# ══════════════════════════════════════════════════════════════════
if len(obj_df) > 1:
    st.divider()
    st.subheader("Confidence Evolution")
    fig = temporal_evolution(obj_df, height=350)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 5: Context (TNS / Truth)
# ══════════════════════════════════════════════════════════════════
def _clean(val, fallback="—"):
    """Return fallback for NaN / None / empty values."""
    if val is None:
        return fallback
    if isinstance(val, float) and np.isnan(val):
        return fallback
    if isinstance(val, str) and val.strip() == "":
        return fallback
    return val

truth = load_truth()
if not truth.empty:
    obj_truth = truth[truth["object_id"] == selected]
    if not obj_truth.empty:
        st.divider()
        st.subheader("External Context")
        t = obj_truth.iloc[0]
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            label = _clean(t.get("final_class_ternary", t.get("target_label")))
            quality = _clean(t.get("label_quality", t.get("truth_quality")))
            source = _clean(t.get("label_source"))
            st.metric("True Class", label)
            st.caption(f"Quality: {quality} | Source: {source}")
        with col_t2:
            tns_name = _clean(t.get("tns_name"))
            tns_type = _clean(t.get("tns_type"))
            st.metric("TNS Name", tns_name)
            st.caption(f"TNS Type: {tns_type}")
        with col_t3:
            z = t.get("redshift", t.get("tns_redshift"))
            z_str = f"{z:.4f}" if z is not None and isinstance(z, (int, float)) and not np.isnan(z) else "—"
            st.metric("Redshift", z_str)

        # Additional context row
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            host = _clean(t.get("host_simbad"))
            st.metric("Host (SIMBAD)", host)
        with col_h2:
            zphot = t.get("host_zphot")
            st.metric("Photo-z", f"{zphot:.3f}" if zphot is not None and isinstance(zphot, (int, float)) and not np.isnan(zphot) else "—")
        with col_h3:
            pstar = t.get("host_pstar")
            st.metric("p(star)", f"{pstar:.3f}" if pstar is not None and isinstance(pstar, (int, float)) and not np.isnan(pstar) else "—")
        with col_h4:
            plx = t.get("gaia_parallax")
            st.metric("Gaia Parallax", f"{plx:.3f}" if plx is not None and isinstance(plx, (int, float)) and not np.isnan(plx) else "—")
