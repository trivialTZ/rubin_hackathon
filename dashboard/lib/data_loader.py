"""DEBASS Dashboard — data loading and caching utilities.

All loaders use @st.cache_data so re-runs are instant.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ── Path resolution ───────────────────────────────────────────────
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _DASHBOARD_DIR.parent          # rubin_hackathon/
DATA_DIR = PROJECT_ROOT / "data"
PUBLISHED_DIR = PROJECT_ROOT / "published_results"
MODELS_DIR = PROJECT_ROOT / "models"
LC_DIR = DATA_DIR / "lightcurves"


# ── JSONL score loading ──────────────────────────────────────────
@st.cache_data(ttl=300)
def load_scores(ndet: int | None = None) -> pd.DataFrame:
    """Load scored JSONL files into a DataFrame.

    Each row = one (object_id, n_det) with flattened expert + ensemble cols.
    If *ndet* is None, loads all available n_det files.
    """
    rows: list[dict] = []
    # Try published_results first, then data/nightly/
    search_dirs = [PUBLISHED_DIR, DATA_DIR / "nightly"]
    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("sample_scores_ndet*.jsonl")):
            with open(p) as f:
                for line in f:
                    rec = json.loads(line)
                    if ndet is not None and rec.get("n_det") != ndet:
                        continue
                    rows.append(_flatten_score(rec))
        if rows:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Deduplicate: keep the latest n_det per object (or all if ndet=None)
    df = df.drop_duplicates(subset=["object_id", "n_det"], keep="last")
    return df.sort_values("p_follow", ascending=False).reset_index(drop=True)


def _flatten_score(rec: dict) -> dict:
    """Flatten nested JSONL record into a flat dict for DataFrame."""
    ens = rec.get("ensemble", {})
    row = {
        "object_id": rec["object_id"],
        "n_det":     rec["n_det"],
        "alert_jd":  rec.get("alert_jd"),
        "p_snia":    ens.get("p_snia_weighted", np.nan),
        "p_nonIa":   ens.get("p_nonIa_snlike_weighted", np.nan),
        "p_other":   ens.get("p_other_weighted", np.nan),
        "p_follow":  ens.get("p_follow_proxy", np.nan),
        "recommended": ens.get("recommended", False),
        "n_experts": ens.get("n_trusted_experts", 0),
    }
    # Per-expert columns
    experts = rec.get("expert_confidence", {})
    row["_experts_raw"] = experts          # keep raw dict for detail pages
    avail_experts = []
    top_trust = (-1.0, "")
    for ek, ev in experts.items():
        avail = ev.get("available", False)
        row[f"avail__{ek}"] = avail
        trust = ev.get("trust")
        row[f"trust__{ek}"] = trust
        if avail:
            avail_experts.append(ek)
            proj = ev.get("projected", {})
            row[f"p_snia__{ek}"] = proj.get("p_snia", np.nan)
            row[f"p_nonIa__{ek}"] = proj.get("p_nonIa_snlike", np.nan)
            row[f"p_other__{ek}"] = proj.get("p_other", np.nan)
            row[f"mapped_class__{ek}"] = ev.get("mapped_pred_class")
            if trust is not None and trust > top_trust[0]:
                top_trust = (trust, ek)
    row["n_avail"] = len(avail_experts)
    row["top_expert"] = top_trust[1] if top_trust[1] else ""
    row["top_trust"] = top_trust[0] if top_trust[0] >= 0 else np.nan
    return row


# ── Metrics loading ──────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_trust_metrics() -> dict:
    p = PUBLISHED_DIR / "expert_trust_metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_data(ttl=600)
def load_followup_metrics() -> dict:
    p = PUBLISHED_DIR / "followup_metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_data(ttl=600)
def load_results_summary() -> dict:
    p = PUBLISHED_DIR / "results_summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ── Lightcurve loading ───────────────────────────────────────────
@st.cache_data(ttl=600)
def load_lightcurve(object_id: str) -> list[dict] | None:
    """Load a cached lightcurve JSON for one object."""
    p = LC_DIR / f"{object_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(ttl=600)
def list_lightcurve_ids() -> list[str]:
    """List all object IDs with cached lightcurves."""
    if not LC_DIR.exists():
        return []
    return sorted(p.stem for p in LC_DIR.glob("*.json"))


# ── Gold table loading ───────────────────────────────────────────
@st.cache_data(ttl=600)
def load_gold_snapshots() -> pd.DataFrame:
    p = DATA_DIR / "gold" / "object_epoch_snapshots.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


# ── Truth table loading ──────────────────────────────────────────
@st.cache_data(ttl=600)
def load_truth() -> pd.DataFrame:
    p = DATA_DIR / "truth" / "object_truth.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


# ── Expert list from a score record ──────────────────────────────
@st.cache_data(ttl=600)
def get_all_expert_keys() -> list[str]:
    """Extract all expert keys from the first score record."""
    for d in [PUBLISHED_DIR, DATA_DIR / "nightly"]:
        if not d.exists():
            continue
        for p in sorted(d.rglob("sample_scores_ndet*.jsonl")):
            with open(p) as f:
                rec = json.loads(f.readline())
                return sorted(rec.get("expert_confidence", {}).keys())
    return []
