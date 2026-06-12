"""Broker score trajectory features (fusion_v8, design §1.3).

``data/silver/broker_events.parquet`` preserves the full per-alert score
history of every broker expert, but the gold builder keeps only the *latest*
as-of event per (object, expert, epoch).  This module mines the discarded
history: for each gold row ``(object_id, n_det, alert_jd)`` and each of the
six experts with dense ``exact_alert`` trajectories it summarizes the
time-series ``x_t = p_snia(t)`` with 10 statistics, plus 6 cross-expert
aggregates — 66 columns total (:data:`TRAJ_FEATURE_NAMES`).

No-leakage argument
-------------------
Only events satisfying ``event_time_jd <= alert_jd`` **and**
``temporal_exactness == 'exact_alert'`` contribute to the features of a gold
row.  This is exactly the as-of rule of
:func:`debass_meta.ingest.gold.select_events_asof` branch (1) (timed events at
or before the alert), so the same leakage proof applies: every contributing
broker score was emitted on an alert published at or before the epoch's
alert time.  Unsafe latest-object snapshots, static context labels and
re-runs without timestamps are excluded entirely.  The implementation uses
prefix (cumulative) statistics over events sorted by time followed by a
backward ``merge_asof`` (``allow_exact_matches=True`` ⇒ ``<=``), so an event
with ``event_time_jd > alert_jd`` can never affect the output row — this is
asserted by ``tests/test_trajectory_asof.py``.

Projection reuse
----------------
Per-event ternary ``p_snia`` mirrors the experts' EXISTING projectors in
``src/debass_meta/projectors/`` (fink.py, fink_lsst.py, pittgoogle.py); no new
score→class mappings are invented.  The per-event math is re-expressed in
vectorized form for throughput (the projectors are per-event-list Python
functions; silver holds ~4M rows) and parity with
:func:`debass_meta.projectors.base._dispatch_projector` is enforced by a
unit test covering every branch.

Conventions
-----------
* ``std`` / ``disagreement_last`` use population std (ddof=0); ``std`` is 0.0
  for a single event, ``disagreement_last`` is NaN with fewer than 2 experts.
* ``slope`` is the OLS slope of p_snia vs time in 1/day (shift-invariant, so
  "vs (t - t_last)" and "vs t" give identical slopes); NaN when fewer than 2
  events or zero time variance.
* ``ewm`` is the exponentially weighted mean with half-life 5 days evaluated
  at the latest as-of event (weights ``0.5**(dt/5)``).
* ``volatility`` is the mean absolute first difference; NaN for n=1.
* ``traj_x__disagreement_trend`` is the prefix OLS slope (per day) of
  ``traj_x__disagreement_last`` over the object's own epochs at
  ``alert_jd' <= alert_jd`` — past rows only, hence leak-free.
* All 10 per-expert stats are NaN when the expert has no prior events;
  ``traj_x__n_experts`` is the count of experts with >= 1 prior event (0.0
  when none).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from debass_meta.projectors.base import sanitize_expert_key

# The six experts with dense exact_alert score trajectories in silver
# (design §1.3).  Order is frozen — it defines the column order of
# TRAJ_FEATURE_NAMES.
TRAJ_EXPERTS: list[str] = [
    "fink/snn",
    "fink/rf_ia",
    "fink_lsst/snn",
    "fink_lsst/cats",
    "fink_lsst/early_snia",
    "pittgoogle/supernnova_lsst",
]

# Per-expert trajectory statistics (10) — frozen order.
TRAJ_STATS: tuple[str, ...] = (
    "n", "last", "mean", "std", "min", "max",
    "slope", "delta", "volatility", "ewm",
)

# Cross-expert aggregates (6) — frozen order.
TRAJ_CROSS_STATS: tuple[str, ...] = (
    "n_experts", "mean_last", "max_last", "mean_slope",
    "disagreement_last", "disagreement_trend",
)

# 6 experts × 10 stats + 6 cross-expert = 66 columns.
TRAJ_FEATURE_NAMES: list[str] = [
    f"traj__{sanitize_expert_key(expert)}__{stat}"
    for expert in TRAJ_EXPERTS
    for stat in TRAJ_STATS
] + [f"traj_x__{stat}" for stat in TRAJ_CROSS_STATS]

_EWM_HALFLIFE_DAYS = 5.0
_NS_PER_DAY = 86_400.0 * 1e9
# Guard against 0/0 in OLS slopes when all timestamps coincide (days^2 units).
_SLOPE_DENOM_EPS = 1e-12

_REQUIRED_EVENT_COLS = (
    "object_id", "expert_key", "event_time_jd",
    "canonical_projection", "temporal_exactness",
)
_EVENT_KEYS = ["object_id", "event_time_jd", "_alert_key"]


# ---------------------------------------------------------------------------
# Event preparation (filter + dedupe)
# ---------------------------------------------------------------------------

def _string_col(events: pd.DataFrame, name: str) -> pd.Series:
    """Return ``events[name]`` as str with missing values as '' (or all-'' if absent)."""
    if name not in events.columns:
        return pd.Series("", index=events.index, dtype=object)
    series = events[name]
    missing = series.isna()
    out = series.astype(str)
    out[missing] = ""
    return out


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Filter silver events to safe trajectory rows and dedupe.

    Keeps only ``temporal_exactness == 'exact_alert'`` rows of the six
    TRAJ_EXPERTS with a timestamp, drops ``availability == False`` rows
    (matching ``select_events_asof``), and removes duplicate event rows —
    e.g. the same payload normalized twice (identical ``payload_hash``) or
    per-detection copies of one alert — by dropping duplicates on
    ``(object_id, expert_key, event_time_jd, alert_id, field, class_name)``
    after a deterministic stable sort that ends on ``payload_hash``.
    """
    missing = [col for col in _REQUIRED_EVENT_COLS if col not in events.columns]
    if missing:
        raise ValueError(f"events frame missing required columns: {missing}")

    mask = (
        events["expert_key"].isin(TRAJ_EXPERTS)
        & (events["temporal_exactness"] == "exact_alert")
        & events["event_time_jd"].notna()
    )
    if "availability" in events.columns:
        mask &= events["availability"].fillna(True).astype(bool)
    sub = events[mask]

    if "alert_id" in sub.columns:
        alert_key = pd.to_numeric(sub["alert_id"], errors="coerce").fillna(-1.0)
    else:
        # Some brokers lack alert_id — group purely by (object_id, event_time_jd).
        alert_key = pd.Series(-1.0, index=sub.index)

    out = pd.DataFrame({
        "object_id": sub["object_id"].astype(str),
        "expert_key": sub["expert_key"].astype(str),
        "event_time_jd": pd.to_numeric(sub["event_time_jd"], errors="coerce").astype(float),
        "canonical_projection": pd.to_numeric(sub["canonical_projection"], errors="coerce"),
        "_alert_key": alert_key.astype(float),
        "_field": _string_col(sub, "field"),
        "_class": _string_col(sub, "class_name"),
        "_hash": _string_col(sub, "payload_hash"),
    })
    if "canonical_complement" in sub.columns:
        out["canonical_complement"] = pd.to_numeric(sub["canonical_complement"], errors="coerce")

    sort_cols = ["object_id", "expert_key", "event_time_jd", "_alert_key", "_field", "_class", "_hash"]
    out = out.sort_values(sort_cols, kind="mergesort")
    out = out.drop_duplicates(
        subset=["object_id", "expert_key", "event_time_jd", "_alert_key", "_field", "_class"],
        keep="last",
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-event p_snia projection (vectorized mirrors of the existing projectors)
# ---------------------------------------------------------------------------

def _one_row_per_event(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (object, time, alert) keeping the last (sorted) row."""
    return frame.drop_duplicates(subset=_EVENT_KEYS, keep="last")


def _project_fink_snn(sub: pd.DataFrame) -> pd.DataFrame:
    """Mirror projectors/fink.py 'fink/snn': p_snia = snia_vs_nonia * sn_vs_all.

    The ternary (a·b, (1-a)·b, 1-b) sums to 1 exactly, so summarize_ternary's
    normalization is the identity and p_snia is the raw product.  Events
    missing either field are dropped (the projector reports "missing snn
    fields" and gold gets no projection).
    """
    a = _one_row_per_event(sub[sub["_field"] == "snn_snia_vs_nonia"])
    b = _one_row_per_event(sub[sub["_field"] == "snn_sn_vs_all"])
    merged = a[_EVENT_KEYS + ["canonical_projection"]].merge(
        b[_EVENT_KEYS + ["canonical_projection"]],
        on=_EVENT_KEYS,
        suffixes=("_snia_vs_nonia", "_sn_vs_all"),
    )
    merged["p_snia"] = (
        merged["canonical_projection_snia_vs_nonia"]
        * merged["canonical_projection_sn_vs_all"]
    )
    return merged[_EVENT_KEYS + ["p_snia"]]


def _project_fink_rf_ia(sub: pd.DataFrame) -> pd.DataFrame:
    """Mirror projectors/fink.py 'fink/rf_ia': p_snia = score (0.7/0.3 split sums to 1)."""
    frame = _one_row_per_event(sub[sub["_field"] == "rf_snia_vs_nonia"]).copy()
    frame["p_snia"] = frame["canonical_projection"]
    return frame[_EVENT_KEYS + ["p_snia"]]


def _project_fink_lsst_snn(sub: pd.DataFrame) -> pd.DataFrame:
    """Mirror projectors/fink_lsst.py _project_snn_lsst: p_snia = 0.5 * P(SN)."""
    frame = _one_row_per_event(sub[sub["canonical_projection"].notna()]).copy()
    frame["p_snia"] = 0.5 * frame["canonical_projection"]
    return frame[_EVENT_KEYS + ["p_snia"]]


def _project_fink_lsst_cats(sub: pd.DataFrame) -> pd.DataFrame:
    """Mirror projectors/fink_lsst.py _project_cats per event.

    Class 11 (SN-like) → p_snia = 0.5 * score (Ia/non-Ia indistinguishable);
    every other class (21/31 non-Ia, 41/51 + unknown → other) has p_snia = 0.
    Missing score is treated as 0 like the projector's ``or 0`` fallback.
    Rows without a parseable class are dropped (projector raises → no
    projection in gold).
    """
    frame = sub[sub["_class"] != ""].copy()
    frame["_class_num"] = pd.to_numeric(frame["_class"], errors="coerce")
    frame = frame[frame["_class_num"].notna()]
    # Last class row per event ≙ the projector's class_events[-1].
    frame = _one_row_per_event(frame)
    score = frame["canonical_projection"].fillna(0.0)
    frame["p_snia"] = np.where(frame["_class_num"] == 11, 0.5 * score, 0.0)
    return frame[_EVENT_KEYS + ["p_snia"]]


def _project_fink_lsst_early_snia(sub: pd.DataFrame) -> pd.DataFrame:
    """Mirror projectors/fink_lsst.py _project_early_snia: p_snia = score, sentinel -1 dropped."""
    valid = sub["canonical_projection"].notna() & (sub["canonical_projection"] >= 0)
    frame = _one_row_per_event(sub[valid]).copy()
    frame["p_snia"] = frame["canonical_projection"]
    return frame[_EVENT_KEYS + ["p_snia"]]


def _project_pittgoogle_snn(sub: pd.DataFrame) -> pd.DataFrame:
    """Mirror projectors/pittgoogle.py _project_supernnova per event.

    With only P(Ia) the ternary (p, 0.7(1-p), 0.3(1-p)) sums to 1 →
    p_snia = P(Ia).  If a ``canonical_complement`` (P(non-Ia)) is present the
    projector normalizes by (p + complement); silver currently has no such
    column so this is a defensive mirror.
    """
    frame = sub[
        (sub["_field"] == "supernnova_prob_class0")
        & sub["canonical_projection"].notna()
    ]
    frame = _one_row_per_event(frame).copy()
    p_ia = frame["canonical_projection"]
    if "canonical_complement" in frame.columns:
        comp = frame["canonical_complement"]
        total = p_ia + comp
        normalized = np.where(total > 0, p_ia / total.where(total > 0), p_ia)
        frame["p_snia"] = np.where(comp.notna(), normalized, p_ia)
    else:
        frame["p_snia"] = p_ia
    return frame[_EVENT_KEYS + ["p_snia"]]


_EVENT_PROJECTORS = {
    "fink/snn": _project_fink_snn,
    "fink/rf_ia": _project_fink_rf_ia,
    "fink_lsst/snn": _project_fink_lsst_snn,
    "fink_lsst/cats": _project_fink_lsst_cats,
    "fink_lsst/early_snia": _project_fink_lsst_early_snia,
    "pittgoogle/supernnova_lsst": _project_pittgoogle_snn,
}


def project_events_per_event(expert_key: str, prepared: pd.DataFrame) -> pd.DataFrame:
    """Per-event p_snia for one TRAJ expert from a :func:`_prepare_events` frame.

    Returns DataFrame[object_id, event_time_jd, _alert_key, p_snia] with one
    row per distinct alert event and no NaN p_snia.  Exposed for the
    projector-parity unit test.
    """
    if expert_key not in _EVENT_PROJECTORS:
        raise KeyError(f"{expert_key} is not a TRAJ expert")
    sub = prepared[prepared["expert_key"] == expert_key]
    if len(sub) == 0:
        return pd.DataFrame(columns=_EVENT_KEYS + ["p_snia"])
    frame = _EVENT_PROJECTORS[expert_key](sub)
    frame = frame[frame["p_snia"].notna()]
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prefix (cumulative, leak-free) trajectory statistics
# ---------------------------------------------------------------------------

def _prefix_stats(per_event: pd.DataFrame) -> pd.DataFrame:
    """Cumulative trajectory stats at each event, in event-time order.

    Row i of the result holds the 10 statistics of the trajectory truncated
    at event i (inclusive), so a backward as-of join on ``event_time_jd``
    yields the statistics over exactly the events with
    ``event_time_jd <= alert_jd`` — no future event participates in any
    cumulative quantity (no-leakage).
    """
    ev = per_event.sort_values(_EVENT_KEYS, kind="mergesort").reset_index(drop=True)
    x = ev["p_snia"].astype(float)

    # Derived columns must exist BEFORE the groupby object is created.
    pre = ev.groupby("object_id", sort=False)
    ev["_x2"] = x * x
    ev["_absdiff"] = (x - pre["p_snia"].shift(1)).abs().fillna(0.0)
    # Slope basis: slope is shift-invariant in t so using t - t_first
    # (numerically stable for JD ~ 2.46e6) equals "vs (t - t_last)".
    ev["_t"] = ev["event_time_jd"] - pre["event_time_jd"].transform("first")
    ev["_t2"] = ev["_t"] * ev["_t"]
    ev["_xt"] = x * ev["_t"]

    grp = ev.groupby("object_id", sort=False)
    n = (grp.cumcount() + 1).astype(float)
    csum = grp["p_snia"].cumsum()
    mean = csum / n
    csum2 = grp["_x2"].cumsum()
    std = np.sqrt((csum2 / n - mean * mean).clip(lower=0.0))  # ddof=0
    cmin = grp["p_snia"].cummin()
    cmax = grp["p_snia"].cummax()
    delta = x - grp["p_snia"].transform("first")
    volatility = (grp["_absdiff"].cumsum() / (n - 1.0)).where(n >= 2)

    # OLS slope of p_snia vs time (1/day).
    st = grp["_t"].cumsum()
    st2 = grp["_t2"].cumsum()
    sxt = grp["_xt"].cumsum()
    denom = n * st2 - st * st
    slope = ((n * sxt - st * csum) / denom).where((n >= 2) & (denom > _SLOPE_DENOM_EPS))

    # Exponentially weighted mean, half-life 5 days, evaluated at each event.
    # Times are converted to datetime64[ns] relative to the frame's earliest
    # event (only differences matter; relative origin avoids the ±292-year
    # datetime64[ns] range limit for arbitrary JD inputs).
    rel_ns = ((ev["event_time_jd"] - ev["event_time_jd"].min()) * _NS_PER_DAY).round().astype("int64")
    times = pd.to_datetime(rel_ns, unit="ns").to_numpy()
    ewm = (
        grp["p_snia"]
        .ewm(halflife=pd.Timedelta(days=_EWM_HALFLIFE_DAYS), times=times)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out = pd.DataFrame({
        "object_id": ev["object_id"],
        "event_time_jd": ev["event_time_jd"],
        "n": n,
        "last": x,
        "mean": mean,
        "std": std,
        "min": cmin,
        "max": cmax,
        "slope": slope,
        "delta": delta,
        "volatility": volatility,
        "ewm": ewm,
    })
    # merge_asof needs the right frame globally sorted on the join key; the
    # stable sort keeps same-time rows of one object in (alert_key) order so
    # the backward join picks the latest event, whose prefix stats already
    # include earlier same-time events.
    return out.sort_values("event_time_jd", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_trajectory_features(events: pd.DataFrame, gold_keys: pd.DataFrame) -> pd.DataFrame:
    """Build the 66 trajectory feature columns for every gold row.

    Parameters
    ----------
    events : pd.DataFrame
        Silver ``broker_events`` rows (columns incl. object_id, expert_key,
        class_name, canonical_projection, event_time_jd, alert_id,
        payload_hash, temporal_exactness).  May be empty.
    gold_keys : pd.DataFrame
        One row per gold row with columns ``object_id, n_det, alert_jd``.

    Returns
    -------
    pd.DataFrame
        ``gold_keys`` (all input columns, original row order) plus the 66
        :data:`TRAJ_FEATURE_NAMES` columns; per-expert stats are NaN where
        the expert has no events at ``event_time_jd <= alert_jd`` with
        ``temporal_exactness == 'exact_alert'`` (strict as-of — see module
        docstring for the no-leakage argument).

    Fully vectorized: one stable sort + cumulative groupby per expert and a
    backward ``merge_asof`` per expert against the gold keys; no per-gold-row
    Python loop over events (handles ~4M silver rows in well under a minute).
    """
    for col in ("object_id", "n_det", "alert_jd"):
        if col not in gold_keys.columns:
            raise ValueError(f"gold_keys missing required column: {col}")

    out = gold_keys.reset_index(drop=True).copy()
    nan_block = pd.DataFrame(
        np.full((len(out), len(TRAJ_FEATURE_NAMES)), np.nan),
        columns=TRAJ_FEATURE_NAMES,
        index=out.index,
    )
    out = pd.concat([out, nan_block], axis=1)
    if len(out) == 0:
        return out

    # Left key frame for the as-of joins (merge_asof needs sorted, non-NaN keys).
    key = pd.DataFrame({
        "object_id": out["object_id"].astype(str),
        "alert_jd": pd.to_numeric(out["alert_jd"], errors="coerce"),
        "__row": np.arange(len(out)),
    })
    key = key[key["alert_jd"].notna()].sort_values("alert_jd", kind="mergesort")

    if len(events) > 0 and len(key) > 0:
        prepared = _prepare_events(events)
        for expert in TRAJ_EXPERTS:
            per_event = project_events_per_event(expert, prepared)
            if len(per_event) == 0:
                continue
            stats = _prefix_stats(per_event)
            merged = pd.merge_asof(
                key,
                stats,
                by="object_id",
                left_on="alert_jd",
                right_on="event_time_jd",
                direction="backward",
                allow_exact_matches=True,  # <= alert_jd, matching select_events_asof
            )
            rows = merged["__row"].to_numpy()
            san = sanitize_expert_key(expert)
            for stat in TRAJ_STATS:
                values = np.full(len(out), np.nan)
                values[rows] = merged[stat].to_numpy(dtype=float)
                out[f"traj__{san}__{stat}"] = values

    _add_cross_expert_aggregates(out)
    return out


def _add_cross_expert_aggregates(out: pd.DataFrame) -> None:
    """Fill the 6 traj_x__* columns in place from the per-expert columns."""
    sans = [sanitize_expert_key(expert) for expert in TRAJ_EXPERTS]
    lasts = out[[f"traj__{san}__last" for san in sans]]
    slopes = out[[f"traj__{san}__slope" for san in sans]]

    n_experts = lasts.notna().sum(axis=1).astype(float)
    out["traj_x__n_experts"] = n_experts
    out["traj_x__mean_last"] = lasts.mean(axis=1)
    out["traj_x__max_last"] = lasts.max(axis=1)
    out["traj_x__mean_slope"] = slopes.mean(axis=1)
    # Cross-sectional population std of the latest p_snia across available
    # experts; undefined (NaN) with fewer than 2 experts.
    out["traj_x__disagreement_last"] = lasts.std(axis=1, ddof=0).where(n_experts >= 2)
    out["traj_x__disagreement_trend"] = _disagreement_trend(out)


def _disagreement_trend(out: pd.DataFrame) -> pd.Series:
    """Prefix OLS slope (per day) of disagreement_last over each object's epochs.

    For gold row k of an object, fit disagreement_last vs alert_jd over the
    object's rows with ``alert_jd <= alert_jd_k`` (stable sort by alert_jd,
    then n_det) that have a defined disagreement.  Uses only the current and
    PAST epochs of the same object — leak-free by construction.  NaN with
    fewer than 2 defined points or zero time variance.
    """
    s = pd.DataFrame({
        "object_id": out["object_id"].astype(str),
        "alert_jd": pd.to_numeric(out["alert_jd"], errors="coerce"),
        "n_det": pd.to_numeric(out["n_det"], errors="coerce"),
        "d": out["traj_x__disagreement_last"].astype(float),
    }, index=out.index)
    s = s.sort_values(["object_id", "alert_jd", "n_det"], kind="mergesort")

    valid = s["d"].notna() & s["alert_jd"].notna()
    grp_first = s.groupby("object_id", sort=False)["alert_jd"].transform("first")
    s["_t"] = (s["alert_jd"] - grp_first).where(valid, 0.0)
    s["_x"] = s["d"].where(valid, 0.0)
    s["_m"] = valid.astype(float)
    s["_t2"] = s["_t"] * s["_t"]
    s["_xt"] = s["_x"] * s["_t"]

    grp = s.groupby("object_id", sort=False)
    k = grp["_m"].cumsum()
    st = grp["_t"].cumsum()
    st2 = grp["_t2"].cumsum()
    sx = grp["_x"].cumsum()
    sxt = grp["_xt"].cumsum()
    denom = k * st2 - st * st
    trend = ((k * sxt - st * sx) / denom).where((k >= 2) & (denom > _SLOPE_DENOM_EPS))
    return trend.reindex(out.index)
