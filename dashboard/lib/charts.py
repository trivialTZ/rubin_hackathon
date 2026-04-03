"""DEBASS Dashboard — Plotly chart factory functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .style import BAND_COLORS, BAND_ORDER, CLASS_COLORS, STATUS_COLORS, expert_name


# ── Lightcurve ────────────────────────────────────────────────────
def lightcurve_plot(
    detections: list[dict],
    title: str = "",
    height: int = 450,
    n_det_markers: list[int] | None = None,
) -> go.Figure:
    """Interactive multi-band lightcurve with error bars.

    *detections* is a list of dicts with keys:
        jd, band, mag, magerr  (optionally: quality, source)
    """
    fig = go.Figure()
    if not detections:
        fig.add_annotation(text="No lightcurve data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16)
        return fig

    df = pd.DataFrame(detections)
    # Normalise band names
    if "band" not in df.columns and "filter" in df.columns:
        df["band"] = df["filter"]

    # Time axis: days since first detection
    if "jd" in df.columns:
        t0 = df["jd"].min()
        df["t"] = df["jd"] - t0
        xlab = f"Days since first detection (JD₀={t0:.2f})"
    elif "mjd" in df.columns:
        t0 = df["mjd"].min()
        df["t"] = df["mjd"] - t0
        xlab = f"Days since first detection (MJD₀={t0:.2f})"
    else:
        df["t"] = range(len(df))
        xlab = "Detection index"

    for band in BAND_ORDER:
        sub = df[df["band"] == band] if "band" in df.columns else pd.DataFrame()
        if sub.empty:
            continue
        color = BAND_COLORS.get(band, "#ffffff")
        err_y = dict(type="data", array=sub["magerr"].tolist(), visible=True) if "magerr" in sub.columns else None
        fig.add_trace(go.Scatter(
            x=sub["t"], y=sub["mag"],
            mode="markers+lines",
            name=band,
            marker=dict(color=color, size=8, line=dict(width=1, color="#fff")),
            line=dict(color=color, width=1.5, dash="dot"),
            error_y=err_y,
            hovertemplate=f"<b>{band}</b><br>t=%{{x:.2f}}d<br>mag=%{{y:.3f}}<extra></extra>",
        ))

    # Vertical markers at key n_det values
    if n_det_markers and "jd" in df.columns:
        df_sorted = df.sort_values("jd")
        for nd in n_det_markers:
            if nd <= len(df_sorted):
                t_nd = df_sorted.iloc[nd - 1]["t"]
                fig.add_vline(x=t_nd, line_dash="dash", line_color="#666",
                              annotation_text=f"n={nd}", annotation_position="top right")

    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title="Magnitude",
        yaxis=dict(autorange="reversed"),  # Astronomers: brighter = lower mag
        template="plotly_dark",
        height=height,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


# ── Expert confidence heatmap ─────────────────────────────────────
def expert_heatmap(
    df: pd.DataFrame,
    expert_keys: list[str],
    value_col_prefix: str = "p_snia__",
    trust_col_prefix: str = "trust__",
    max_objects: int = 50,
    height: int = 600,
) -> go.Figure:
    """Heatmap: rows=objects, cols=experts, colour=p(SNIa), opacity=trust."""
    obj_ids = df["object_id"].tolist()[:max_objects]
    display_names = [expert_name(k) for k in expert_keys]

    z_vals = []
    hover_texts = []
    for _, row in df.head(max_objects).iterrows():
        z_row = []
        h_row = []
        for ek in expert_keys:
            avail = row.get(f"avail__{ek}", False)
            if not avail:
                z_row.append(np.nan)
                h_row.append(f"{expert_name(ek)}<br>Unavailable")
            else:
                p = row.get(f"{value_col_prefix}{ek}", np.nan)
                t = row.get(f"{trust_col_prefix}{ek}")
                t_str = f"{t:.3f}" if t is not None and not np.isnan(t) else "N/A"
                cls = row.get(f"mapped_class__{ek}", "?")
                z_row.append(p if not np.isnan(p) else 0)
                h_row.append(
                    f"<b>{expert_name(ek)}</b><br>"
                    f"p(SNIa)={p:.3f}<br>"
                    f"Trust={t_str}<br>"
                    f"Class={cls}"
                )
        z_vals.append(z_row)
        hover_texts.append(h_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=display_names,
        y=obj_ids,
        colorscale=[
            [0.0, "#2c3e50"],    # low p(SNIa) — dark blue
            [0.5, "#f39c12"],    # mid — orange
            [1.0, "#e74c3c"],    # high p(SNIa) — red
        ],
        colorbar=dict(title="p(SNIa)"),
        hovertext=hover_texts,
        hoverinfo="text",
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title="Expert Confidence Matrix — p(SNIa)",
        xaxis_title="Expert Classifier",
        yaxis_title="Object ID",
        template="plotly_dark",
        height=height,
        xaxis=dict(tickangle=45),
        margin=dict(l=120, r=20, t=50, b=100),
    )
    return fig


# ── Ternary probability pie / donut ──────────────────────────────
def ternary_donut(p_snia: float, p_nonIa: float, p_other: float,
                  title: str = "", height: int = 250) -> go.Figure:
    labels = ["SN Ia", "Non-Ia SN-like", "Other"]
    values = [p_snia, p_nonIa, p_other]
    colors = [CLASS_COLORS["snia"], CLASS_COLORS["nonIa_snlike"], CLASS_COLORS["other"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        hole=0.55,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


# ── Expert agreement bar chart ───────────────────────────────────
def expert_agreement_bars(
    experts_raw: dict,
    height: int = 350,
) -> go.Figure:
    """Grouped bar chart: each expert's ternary projection side-by-side."""
    names, snia, nonIa, other, trusts = [], [], [], [], []
    for ek, ev in experts_raw.items():
        if not ev.get("available"):
            continue
        proj = ev.get("projected", {})
        names.append(expert_name(ek))
        snia.append(proj.get("p_snia", 0))
        nonIa.append(proj.get("p_nonIa_snlike", 0))
        other.append(proj.get("p_other", 0))
        trusts.append(ev.get("trust") or 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="SN Ia", x=names, y=snia,
                         marker_color=CLASS_COLORS["snia"]))
    fig.add_trace(go.Bar(name="Non-Ia", x=names, y=nonIa,
                         marker_color=CLASS_COLORS["nonIa_snlike"]))
    fig.add_trace(go.Bar(name="Other", x=names, y=other,
                         marker_color=CLASS_COLORS["other"]))
    # Trust overlay as line
    fig.add_trace(go.Scatter(
        x=names, y=trusts, mode="markers+lines",
        name="Trust", yaxis="y2",
        marker=dict(color="#2ecc71", size=10, symbol="diamond"),
        line=dict(color="#2ecc71", dash="dot"),
    ))
    fig.update_layout(
        barmode="stack",
        title="Expert Predictions + Trust",
        template="plotly_dark",
        height=height,
        yaxis=dict(title="Probability", range=[0, 1.05]),
        yaxis2=dict(title="Trust", overlaying="y", side="right", range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=60, r=60, t=50, b=80),
    )
    return fig


# ── Temporal evolution chart ─────────────────────────────────────
def temporal_evolution(
    df_obj: pd.DataFrame,
    height: int = 350,
) -> go.Figure:
    """p_follow and p_snia vs n_det for a single object."""
    if df_obj.empty:
        fig = go.Figure()
        fig.add_annotation(text="No temporal data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=14)
        return fig

    df_obj = df_obj.sort_values("n_det")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df_obj["n_det"], y=df_obj["p_follow"],
        mode="markers+lines", name="p(follow-up)",
        marker=dict(color=STATUS_COLORS["recommended"], size=10),
        line=dict(color=STATUS_COLORS["recommended"], width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df_obj["n_det"], y=df_obj["p_snia"],
        mode="markers+lines", name="p(SNIa)",
        marker=dict(color=CLASS_COLORS["snia"], size=8),
        line=dict(color=CLASS_COLORS["snia"], width=2, dash="dot"),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df_obj["n_det"], y=df_obj["n_avail"],
        mode="markers+lines", name="# Experts",
        marker=dict(color="#95a5a6", size=6),
        line=dict(color="#95a5a6", width=1, dash="dash"),
    ), secondary_y=True)

    fig.update_layout(
        title="Confidence Evolution with Detections",
        template="plotly_dark",
        height=height,
        margin=dict(l=60, r=60, t=50, b=40),
    )
    fig.update_xaxes(title_text="Number of Detections")
    fig.update_yaxes(title_text="Probability", range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text="# Available Experts", secondary_y=True)
    return fig


# ── Expert coverage vs n_det ─────────────────────────────────────
def coverage_vs_ndet(summary: dict, height: int = 400) -> go.Figure:
    """Line chart of expert coverage % at different n_det thresholds.

    Supports two JSON formats:
    - {"expert_coverage_by_ndet": {expert: {ndet: pct}}}
    - {"expert_availability_sample": [{expert, n_det, coverage_pct, ...}]}
    """
    # Try the list-of-dicts format first (actual pipeline output)
    avail_sample = summary.get("expert_availability_sample", [])
    if avail_sample:
        import pandas as _pd
        df = _pd.DataFrame(avail_sample)
        if "expert" in df.columns and "n_det" in df.columns and "coverage_pct" in df.columns:
            fig = go.Figure()
            for ek in df["expert"].unique():
                sub = df[df["expert"] == ek].sort_values("n_det")
                if sub["coverage_pct"].sum() == 0:
                    continue  # skip experts with 0% everywhere
                fig.add_trace(go.Scatter(
                    x=sub["n_det"], y=sub["coverage_pct"],
                    mode="markers+lines",
                    name=expert_name(ek),
                ))
            fig.update_layout(
                title="Expert Coverage vs. Number of Detections",
                xaxis_title="n_det",
                yaxis_title="Coverage (%)",
                template="plotly_dark",
                height=height,
                margin=dict(l=60, r=20, t=50, b=40),
            )
            return fig

    # Fallback: dict-of-dicts format
    coverage = summary.get("expert_coverage_by_ndet", {})
    if not coverage:
        fig = go.Figure()
        fig.add_annotation(text="No coverage data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure()
    all_ndets = sorted(set(int(k) for v in coverage.values() for k in v.keys()))
    for ek, ndet_map in coverage.items():
        ys = [ndet_map.get(str(nd), 0) for nd in all_ndets]
        fig.add_trace(go.Scatter(
            x=all_ndets, y=ys,
            mode="markers+lines",
            name=expert_name(ek),
        ))
    fig.update_layout(
        title="Expert Coverage vs. Number of Detections",
        xaxis_title="n_det",
        yaxis_title="Coverage (%)",
        template="plotly_dark",
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig
