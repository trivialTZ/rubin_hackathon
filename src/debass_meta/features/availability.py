"""Expert-input groups and availability masking for fusion gold tables.

A gold row carries, per expert, a block of columns keyed by the sanitized expert key (``proj__fink_lsst__snn__p_snia``,
``avail__fink_lsst__snn``, ``traj__fink_lsst__snn__last``, ...). ``mask_experts`` blanks whole blocks so a row looks as
if those experts never fired: numeric columns become NaN, ``avail__*`` flags 0, text columns None. Cross-expert
trajectory summaries (``traj_x__*``) are built from broker score histories and go with the brokers.

Groups:
  LOCAL_EXPERTS   re-run by scripts/local_infer.py on the lightcurve (always obtainable at scoring time)
  BROKER_EXPERTS  everything else (Fink, ALeRCE, Lasair, Babamul, Pitt-Google, ANTARES, AMPEL ParSNIP)

Used by the input-availability ablation (scripts/eval_input_ablation.py).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from debass_meta.projectors.base import EXPERT_REGISTRY, sanitize_expert_key

# ampel/snguess is the vendored SNGuess model, re-run locally (source "ampel" in the registry).
LOCAL_EXPERTS = frozenset(k for k, (_, src) in EXPERT_REGISTRY.items() if src.startswith("local")) | {"ampel/snguess"}
BROKER_EXPERTS = frozenset(EXPERT_REGISTRY) - LOCAL_EXPERTS

EXPERT_COL_PREFIXES = ("proj", "traj", "avail", "exact", "temporal_exactness", "source_event_time_jd", "reason",
                       "event_count", "prediction_type", "mapped_pred_class", "context_tag", "q", "trust_source")
CROSS_TRAJ_PREFIX = "traj_x__"

_BY_SAN = {sanitize_expert_key(k): k for k in EXPERT_REGISTRY}
# longest first, so "ampel__snguess" wins over a hypothetical "ampel"
_SAN_ORDER = sorted(_BY_SAN, key=len, reverse=True)

GROUPS = {"brokers": BROKER_EXPERTS, "local": LOCAL_EXPERTS, "all": frozenset(EXPERT_REGISTRY)}


def expert_of_column(col: str) -> str | None:
    """Registry key of the expert a gold column belongs to, or None for non-expert columns."""
    prefix, sep, rest = col.partition("__")
    if not sep or prefix not in EXPERT_COL_PREFIXES:
        return None
    for san in _SAN_ORDER:
        if rest == san or rest.startswith(san + "__"):
            return _BY_SAN[san]
    return None


def expert_columns(columns: Iterable[str], experts: Iterable[str], *, cross_traj: bool = False) -> list[str]:
    """Columns of ``columns`` that belong to ``experts`` (plus traj_x__* when ``cross_traj``)."""
    want = set(experts)
    out = []
    for c in columns:
        k = expert_of_column(c)
        if (k is not None and k in want) or (cross_traj and c.startswith(CROSS_TRAJ_PREFIX)):
            out.append(c)
    return out


def mask_experts(df: pd.DataFrame, experts: Iterable[str], *, cross_traj: bool | None = None,
                 rows: np.ndarray | pd.Series | None = None) -> pd.DataFrame:
    """Copy of ``df`` with every column of ``experts`` blanked (on ``rows`` only, if given).

    ``cross_traj`` defaults to True when any broker is masked."""
    experts = set(experts)
    if cross_traj is None:
        cross_traj = bool(experts & BROKER_EXPERTS)
    out = df.copy()
    idx = slice(None) if rows is None else np.asarray(rows, dtype=bool)
    for c in expert_columns(out.columns, experts, cross_traj=cross_traj):
        if c.startswith("avail__"):
            out.loc[idx, c] = 0.0 if pd.api.types.is_numeric_dtype(out[c]) else False
        elif pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c]):
            if not pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].astype(float)
            out.loc[idx, c] = np.nan
        else:
            if out[c].dtype != object:
                out[c] = out[c].astype(object)
            out.loc[idx, c] = None
    return out


def mask_group(df: pd.DataFrame, group: str, **kw) -> pd.DataFrame:
    """``mask_experts`` for a named group: "brokers", "local" or "all"."""
    return mask_experts(df, GROUPS[group], **kw)
