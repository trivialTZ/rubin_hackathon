#!/usr/bin/env python3
"""Epoch-aware LSST-live truth engine (fusion_v11 P1).

Three products (fusion_v11 spec §1, §6, review-integration §3.7):

  1. ``data/truth/lsst_live_truth.parquet`` — the 20-column ``object_truth``
     schema (dtypes pinned), one row per wanted LSST object. Built from a
     paged Lasair ``objects,crossmatch_tns`` seed, with **TNS-bulk-authoritative**
     type/discoverydate, a client-side sep <=2" cut, an epoch-window
     (tns_discovery in [firstDiaSourceMjdTai-90d, lastDiaSourceMjdTai+30d])
     stale demotion (``label_quality='stale_xmatch'`` — excluded from train AND
     eval), plus NEW catalog negatives for train/cal (B5, target >=300).

  2. ``<cohort>_cleaned.parquet`` — the 2026-07-04 eval cohort re-derived with
     the same epoch-aware crossmatch (NEVER overwrites ``data/live_eval_20260704/**``).

  3. ``data/gold/lsst_live_locked_test.json`` — the frozen benchmark manifest
     ``{"test_ids", "frozen_utc", "policy", "source"}``. test_ids = cleaned spec
     survivors + the 150 catalog others (the 78 untyped join SSL corpora only).

The pure helpers (window / demotion / <=2" cut / manifest / Fink-"Fail" /
hash-routing) carry no I/O and are unit-tested in tests/test_truth_lsst_live.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

from debass_meta.access.tns import is_ambiguous_type, map_tns_type_to_ternary

# 20-column object_truth schema (review-integration §3.7 dtype pins).
TRUTH_COLUMNS = [
    "object_id", "final_class_ternary", "follow_proxy", "label_source",
    "label_quality", "bts_type", "tns_name", "redshift", "final_class_raw",
    "truth_timestamp", "tns_prefix", "tns_type", "tns_has_spectra",
    "tns_redshift", "tns_ra", "tns_dec", "tns_discovery_date",
    "consensus_experts", "consensus_n_agree", "consensus_n_total",
]
_FLOAT64_COLS = (
    "truth_timestamp", "tns_redshift", "tns_ra", "tns_dec",
    "consensus_experts", "consensus_n_agree", "consensus_n_total",
)

FINK_FAIL_SENTINEL = "Fail"

# Epoch window (days) around the LSST detection baseline.
WINDOW_PRE_DAYS = 90.0
WINDOW_POST_DAYS = 30.0
MAX_SEP_ARCSEC = 2.0

STALE_QUALITY = "stale_xmatch"
TAIL_QUALITY = "tail_xmatch"

# Sherlock context classes that mark a non-transient (catalog negative).
# Lasair sherlock_classifications.classification codes; SN/NT(nuclear)/ORPHAN
# are potential transients and are NOT harvested as negatives.
SHERLOCK_NEGATIVE_CLASSES = {"VS", "AGN", "CV", "BS", "STAR"}

# Default Fink LSST per-alert MJD field (spec §1).
FINK_MJD_FIELD = "i:midpointMjdTai"


# ------------------------------------------------------------------ #
# Pure helpers (no I/O; unit-tested)                                  #
# ------------------------------------------------------------------ #


def discovery_date_to_mjd(date_str: Any) -> float | None:
    """Parse a TNS discovery-date string to MJD (client-side MJD<->date)."""
    if date_str is None:
        return None
    text = str(date_str).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        from astropy.time import Time

        return float(Time(text.replace(" ", "T"), format="isot").mjd)
    except Exception:
        # Fallback: bare-date parse -> MJD via the Unix-epoch offset (40587).
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(text.split(".")[0], fmt).replace(
                    tzinfo=timezone.utc
                )
                return dt.timestamp() / 86400.0 + 40587.0
            except ValueError:
                continue
        return None


def window_position(
    disc_mjd: float | None,
    first_mjd: float | None,
    last_mjd: float | None,
    *,
    pre: float = WINDOW_PRE_DAYS,
    post: float = WINDOW_POST_DAYS,
) -> str:
    """Where the TNS discovery MJD sits relative to the detection baseline.

    Returns ``"in"`` / ``"before"`` / ``"after"``. Unknown discovery/baseline
    MJDs → ``"in"`` (cannot prove staleness). ``"before"`` = discovery predates
    the LSST detections (a positional false match → stale); ``"after"`` = a late
    same-transient match (tail).
    """
    if disc_mjd is None or first_mjd is None or last_mjd is None:
        return "in"
    if disc_mjd < (first_mjd - pre):
        return "before"
    if disc_mjd > (last_mjd + post):
        return "after"
    return "in"


def in_discovery_window(
    disc_mjd: float | None,
    first_mjd: float | None,
    last_mjd: float | None,
    *,
    pre: float = WINDOW_PRE_DAYS,
    post: float = WINDOW_POST_DAYS,
) -> bool:
    """True iff the TNS discovery MJD falls within the detection baseline."""
    return window_position(disc_mjd, first_mjd, last_mjd, pre=pre, post=post) == "in"


def sep_ok(sep_arcsec: float | None, *, max_sep: float = MAX_SEP_ARCSEC) -> bool:
    """Client-side sep <=2" cut (the Lasair join is wider — max 2.97", spec §1)."""
    if sep_arcsec is None:
        return True
    try:
        return float(sep_arcsec) <= max_sep
    except (TypeError, ValueError):
        return True


def resolve_label_quality(
    tns_type: Any,
    tns_name: Any,
    disc_mjd: float | None,
    first_mjd: float | None,
    last_mjd: float | None,
) -> str | None:
    """Assign label_quality with epoch-aware stale demotion.

    * discovery before the window -> ``'stale_xmatch'`` (positional false match)
    * discovery after the window  -> ``'tail_xmatch'`` (late same-transient match)
      — both excluded from train AND eval, kept with the rejection reason.
    * TNS type present (non-ambiguous) -> ``'spectroscopic'``
    * ambiguous generic ``SN`` -> ``'tns_ambiguous'``
    * TNS name only -> ``'tns_untyped'``
    * nothing -> ``None``
    """
    has_name = bool(tns_name and str(tns_name).strip() and str(tns_name).lower() != "nan")
    has_type = bool(tns_type and str(tns_type).strip() and str(tns_type).lower() != "nan")
    if not has_name and not has_type:
        return None
    pos = window_position(disc_mjd, first_mjd, last_mjd)
    if pos == "before":
        return STALE_QUALITY
    if pos == "after":
        return TAIL_QUALITY
    if has_type:
        if is_ambiguous_type(str(tns_type)):
            return "tns_ambiguous"
        return "spectroscopic"
    return "tns_untyped"


def map_fink_xm(value: Any) -> Any:
    """Map a Fink LSST xm field, treating the ``"Fail"`` sentinel as missing."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == FINK_FAIL_SENTINEL or text.lower() == "nan":
        return None
    return value


def aggregate_fink_xm(
    alerts: list[dict],
    field: str,
    *,
    mjd_field: str = FINK_MJD_FIELD,
) -> Any:
    """Max-MJD non-nan cross-alert aggregation of a Fink LSST xm field (spec §1).

    The per-alert ``f:xm_*`` values are intermittently missing even after
    classification (SN 2026ctw: type absent on 178/470 post-classification
    alerts incl. the latest). Truth extraction therefore takes the value from
    the *latest* (max-MJD) alert whose field is non-nan/non-``"Fail"``. Returns
    ``None`` when no alert carries the field.
    """
    best_mjd: float | None = None
    best_val: Any = None
    for alert in alerts or []:
        if not isinstance(alert, dict):
            continue
        value = map_fink_xm(alert.get(field))
        if value is None:
            continue
        mjd = _to_float(alert.get(mjd_field))
        if mjd is None:
            # No timestamp: accept only if we have nothing better yet.
            if best_mjd is None and best_val is None:
                best_val = value
            continue
        if best_mjd is None or mjd > best_mjd:
            best_mjd = mjd
            best_val = value
    return best_val


def catalog_negative_rows(
    records: list[dict],
    *,
    frozen_ids: set[str] | None = None,
    negative_classes: set[str] = SHERLOCK_NEGATIVE_CLASSES,
) -> list[dict]:
    """Build NEW catalog-negative truth rows from Lasair sherlock records (B5).

    One row per LSST object whose sherlock ``classification`` is a non-transient
    context class (VS/AGN/CV/BS/STAR), EXCLUDING the frozen benchmark cohort
    (they stay benchmark-only, spec §6 B5). De-duplicated on object id.
    """
    frozen_ids = frozen_ids or set()
    seen: set[str] = set()
    rows: list[dict] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        oid = _record_lsst_id_local(rec)
        if not oid or oid in frozen_ids or oid in seen:
            continue
        cls = rec.get("classification") or rec.get("sherlock_classification")
        cls = str(cls).strip().upper() if cls is not None else ""
        if cls not in negative_classes:
            continue
        seen.add(oid)
        rows.append(make_catalog_negative_row(
            object_id=oid,
            label_source="lasair_sherlock",
            basis=f"sherlock:{cls}",
        ))
    return rows


def _record_lsst_id_local(record: dict) -> str | None:
    for key in ("diaObjectId", "objects.diaObjectId", "objectId", "object_id"):
        val = record.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def hash_route(object_id: Any) -> str:
    """Post-freeze routing: sha1(object_id) last hex digit even->TEST, odd->train (B1)."""
    digest = hashlib.sha1(str(object_id).strip().encode("utf-8")).hexdigest()
    return "test" if int(digest[-1], 16) % 2 == 0 else "train"


def load_prior_manifest(path: Path) -> dict[str, Any] | None:
    """Read an existing frozen benchmark manifest (or None if absent/unreadable).

    Used to keep the manifest APPEND-ONLY (B1/§6): a re-run must never drop a
    previously-frozen id and must preserve the original ``frozen_utc``."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # degrade loudly, never silently
        print(f"WARNING: could not read prior manifest {path}: {exc}")
        return None


def build_manifest(
    test_ids: list[str],
    *,
    frozen_utc: str | None = None,
    policy: str,
    source: str,
    prior_test_ids: list[str] | None = None,
    excluded_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Frozen benchmark manifest (P2's ``--lsst-live-locked`` schema, B11).

    APPEND-ONLY (B1/§6): ``prior_test_ids`` (from an existing manifest) are
    emitted FIRST and never dropped; new ids append.  ``frozen_utc`` is
    preserved across re-runs — pass the prior manifest's value so the freeze
    timestamp does not churn.  'Frozen ids never move' is enforced here by
    construction, not merely by convention.

    ``excluded_ids`` ({object_id: reason}) is the ONLY sanctioned way to
    remove an id from the benchmark: ids whose association counterpart sits in
    the locked v6e2/v10 train/cal set (G6 counterpart integrity — e.g. 2026ekf
    / 2026gzf, whose ZTF twins were trained on) move here instead of being
    silently dropped. Excluded ids are themselves append-only (carried forward
    from the prior manifest), never emitted in ``test_ids``, and never
    re-frozen by later refreshes.
    """
    if frozen_utc is None:
        frozen_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    excluded: dict[str, str] = {
        str(k).strip(): str(v) for k, v in (excluded_ids or {}).items() if str(k).strip()
    }
    # De-dup, order-stable, string ids; prior (frozen) ids lead so they can
    # never be dropped by a changing live seed. Excluded ids never re-enter.
    seen: set[str] = set()
    ids: list[str] = []
    for t in list(prior_test_ids or []) + list(test_ids):
        s = str(t).strip()
        if s and s not in seen and s not in excluded:
            seen.add(s)
            ids.append(s)
    manifest = {
        "test_ids": ids,
        "frozen_utc": frozen_utc,
        "policy": policy,
        "source": source,
    }
    if excluded:
        manifest["excluded_ids"] = excluded
    return manifest


def truth_rows_to_df(rows: list[dict]):
    """Coerce truth rows into the pinned 20-column schema/dtypes."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(rows, columns=TRUTH_COLUMNS)
    if len(df) == 0:
        # Still return a correctly-typed empty frame.
        for c in TRUTH_COLUMNS:
            df[c] = pd.Series(dtype="float64" if c in _FLOAT64_COLS else "object")
    df["follow_proxy"] = df["follow_proxy"].fillna(0).astype("int64")
    for c in _FLOAT64_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


# Truth dedup priority: higher-quality labels win when the same object_id
# appears more than once (an object can be a TNS-crossmatch seed row AND a
# sherlock catalog negative; Lasair offset pagination without ORDER BY can
# also repeat rows across pages — the 2026-07-05 SCC build crash).
_QUALITY_PRIORITY = {
    "spectroscopic": 0, "tns_untyped": 1, "context": 2, "weak": 3,
    "bts_untyped": 4, "stale_xmatch": 5, "tail_xmatch": 6,
}


def dedupe_truth_frame(df):
    """One row per ``object_id``, keeping the highest-quality label.

    Priority: spectroscopic > tns_untyped > context > weak > bts_untyped >
    stale_xmatch > tail_xmatch > unlabeled. Ties keep a tns_name-bearing row,
    then the first occurrence (stable/deterministic). The gold builder's
    ``_load_truth_lookup`` requires a UNIQUE object_id index — every truth
    frame written for it must pass through here.
    """
    import pandas as pd

    if df is None or len(df) == 0 or df["object_id"].is_unique:
        return df
    prio = df["label_quality"].map(_QUALITY_PRIORITY).fillna(9).astype(int)
    if "tns_name" in df.columns:
        unnamed = df["tns_name"].isna() | (df["tns_name"].astype(str).str.strip() == "")
    else:
        unnamed = pd.Series(True, index=df.index)
    order = pd.DataFrame({"p": prio.values, "u": unnamed.values}, index=df.index)
    best = df.loc[order.sort_values(["p", "u"], kind="stable").index]
    best = best.drop_duplicates("object_id", keep="first")
    return best.sort_index().reset_index(drop=True)


def merge_live_into_truth(base_df, live_df):
    """Merge LSST-live truth rows into the v11 ZTF truth (TRUTH PLUMBING PIN).

    Keyed on ``object_id``; LSST-live rows WIN for their ids. ``stale_xmatch`` /
    ``tail_xmatch`` rows are carried through with their demoted quality so the
    builder can exclude them (spec §3, P1 FINAL step). Pure over its inputs —
    unit-tested with synthetic frames.

    Both inputs and the merged result are deduped on ``object_id`` (quality-
    priority, see :func:`dedupe_truth_frame`) — the output feeds the gold
    builder, which hard-requires unique ids.

    The result is the training-build ``--truth`` file: the ZTF spec corpus
    (rederived under B0) plus every wanted LSST-live object.
    """
    import pandas as pd

    live_df = dedupe_truth_frame(live_df)
    base_df = dedupe_truth_frame(base_df)
    if live_df is None or len(live_df) == 0:
        return base_df.reset_index(drop=True) if base_df is not None else live_df
    if base_df is None or len(base_df) == 0:
        return live_df.reset_index(drop=True)

    live_ids = set(live_df["object_id"].astype(str))
    kept_base = base_df[~base_df["object_id"].astype(str).isin(live_ids)]
    # Align columns: keep the base (20-col object_truth) schema, in order.
    cols = list(base_df.columns)
    live_aligned = live_df.reindex(columns=cols)
    merged = pd.concat([kept_base, live_aligned], ignore_index=True)
    merged = dedupe_truth_frame(merged)
    assert merged["object_id"].is_unique, "merged truth must have unique object_ids"
    return merged.reset_index(drop=True)


def _blank_truth_row(object_id: str) -> dict:
    import numpy as np

    return {
        "object_id": str(object_id),
        "final_class_ternary": None,
        "follow_proxy": 0,
        "label_source": None,
        "label_quality": None,
        "bts_type": None,
        "tns_name": None,
        "redshift": None,
        "final_class_raw": None,
        "truth_timestamp": float(datetime.now(timezone.utc).timestamp()),
        "tns_prefix": None,
        "tns_type": None,
        "tns_has_spectra": False,
        "tns_redshift": np.nan,
        "tns_ra": np.nan,
        "tns_dec": np.nan,
        "tns_discovery_date": None,
        "consensus_experts": np.nan,
        "consensus_n_agree": np.nan,
        "consensus_n_total": np.nan,
    }


def make_truth_row(
    *,
    object_id: str,
    tns_name: Any = None,
    tns_type: Any = None,
    tns_prefix: Any = None,
    discovery_date: Any = None,
    redshift: Any = None,
    first_mjd: float | None = None,
    last_mjd: float | None = None,
    label_source: str = "lasair_crossmatch_tns",
    tns_ra: float | None = None,
    tns_dec: float | None = None,
) -> dict:
    """Assemble one epoch-aware truth row (used by live + cohort-clean paths)."""
    import numpy as np

    disc_mjd = discovery_date_to_mjd(discovery_date)
    quality = resolve_label_quality(tns_type, tns_name, disc_mjd, first_mjd, last_mjd)
    ternary = None
    if quality == "spectroscopic":
        ternary = map_tns_type_to_ternary(str(tns_type))
    row = _blank_truth_row(object_id)
    row.update({
        "final_class_ternary": ternary,
        "follow_proxy": int(ternary == "snia"),
        "label_source": label_source if (tns_name or tns_type) else None,
        "label_quality": quality,
        "tns_name": str(tns_name) if tns_name else None,
        "redshift": str(redshift) if redshift is not None else None,
        "final_class_raw": str(tns_type) if tns_type else None,
        "tns_prefix": str(tns_prefix) if tns_prefix else None,
        "tns_type": str(tns_type) if tns_type else None,
        "tns_has_spectra": bool(quality == "spectroscopic"),
        "tns_redshift": float(redshift) if _isnum(redshift) else np.nan,
        "tns_ra": float(tns_ra) if _isnum(tns_ra) else np.nan,
        "tns_dec": float(tns_dec) if _isnum(tns_dec) else np.nan,
        "tns_discovery_date": str(discovery_date) if discovery_date else None,
    })
    return row


def make_catalog_negative_row(
    *, object_id: str, label_source: str = "catalog", basis: str | None = None
) -> dict:
    """A catalog-basis negative (ternary 'other', label_quality 'context')."""
    row = _blank_truth_row(object_id)
    row.update({
        "final_class_ternary": "other",
        "follow_proxy": 0,
        "label_source": str(basis or label_source),
        "label_quality": "context",
    })
    return row


def _isnum(v: Any) -> bool:
    if v is None:
        return False
    try:
        f = float(v)
        return f == f
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------ #
# TNS-bulk authoritative lookup                                       #
# ------------------------------------------------------------------ #


def load_tns_authoritative(tns_bulk_path: Path | None) -> dict[str, dict]:
    """Map TNS name (lowercased, no prefix) -> {type, discoverydate, redshift, ...}.

    TNS bulk is the authoritative type/discoverydate source (spec §1); the
    Lasair ``crossmatch_tns`` join is a match SEED only.
    """
    if tns_bulk_path is None or not Path(tns_bulk_path).exists():
        return {}
    import pandas as pd

    df = pd.read_parquet(tns_bulk_path)
    name_col = "objname" if "objname" in df.columns else "name"
    lookup: dict[str, dict] = {}
    for _, r in df.iterrows():
        name = str(r.get(name_col) or "").strip()
        if not name:
            continue
        lookup[_norm_name(name)] = {
            "type": r.get("type"),
            "discoverydate": r.get("discoverydate"),
            "redshift": r.get("redshift"),
            "name_prefix": r.get("name_prefix"),
            "ra": r.get("ra"),
            "declination": r.get("declination"),
        }
    return lookup


def _norm_name(name: Any) -> str:
    text = str(name or "").strip()
    for pref in ("SN ", "AT ", "SN", "AT"):
        if text.startswith(pref):
            text = text[len(pref):]
            break
    return text.strip().lower()


# ------------------------------------------------------------------ #
# Cohort re-derivation (offline-capable) + manifest                   #
# ------------------------------------------------------------------ #


def clean_cohort_truth(
    *,
    cohort_df,
    tns_auth: dict[str, dict],
    authoritative_truth=None,
):
    """Re-derive cleaned truth rows for the 2026-07-04 eval cohort.

    ``cohort_df`` columns (from ``cohort/merged.csv``): object_id, tns_name,
    tns_type, first_det_mjd, last_det_mjd, cohort ('transients'|'others'),
    coarse_class, label_basis. Authoritative type/discoverydate come from
    ``tns_auth`` (TNS bulk); ``authoritative_truth`` (an existing object_truth
    parquet) is a fallback for discoverydate/type when the bulk is absent.
    """
    import pandas as pd

    auth_truth_map: dict[str, dict] = {}
    if authoritative_truth is not None and len(authoritative_truth) > 0:
        for _, r in authoritative_truth.iterrows():
            auth_truth_map[str(r["object_id"]).strip()] = {
                "tns_type": r.get("tns_type"),
                "tns_discovery_date": r.get("tns_discovery_date"),
                "tns_name": r.get("tns_name"),
            }

    rows: list[dict] = []
    for _, c in cohort_df.iterrows():
        oid = str(c["object_id"]).strip()
        cohort = str(c.get("cohort") or "").strip().lower()
        first_mjd = _isnum(c.get("first_det_mjd")) and float(c["first_det_mjd"]) or None
        last_mjd = _isnum(c.get("last_det_mjd")) and float(c["last_det_mjd"]) or None

        if cohort == "others":
            rows.append(make_catalog_negative_row(
                object_id=oid,
                basis=str(c.get("label_basis") or "catalog"),
            ))
            continue

        tns_name = c.get("tns_name")
        # Authoritative type/disc: TNS bulk first, then existing truth fallback.
        auth = tns_auth.get(_norm_name(tns_name)) if tns_name else None
        tns_type = auth.get("type") if auth else None
        disc_date = auth.get("discoverydate") if auth else None
        redshift = auth.get("redshift") if auth else None
        if (tns_type is None or disc_date is None) and oid in auth_truth_map:
            fb = auth_truth_map[oid]
            tns_type = tns_type if tns_type is not None else fb.get("tns_type")
            disc_date = disc_date if disc_date is not None else fb.get("tns_discovery_date")
        # Cohort's own seed type is the last resort.
        if tns_type is None and _isnum_or_str(c.get("tns_type")):
            tns_type = c.get("tns_type")

        rows.append(make_truth_row(
            object_id=oid,
            tns_name=tns_name,
            tns_type=tns_type,
            discovery_date=disc_date,
            redshift=redshift,
            first_mjd=first_mjd,
            last_mjd=last_mjd,
            label_source="tns_cohort_cleaned",
        ))
    return truth_rows_to_df(rows)


def _isnum_or_str(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return bool(s and s.lower() != "nan")


def cohort_manifest_test_ids(cleaned_truth) -> list[str]:
    """Benchmark test_ids = cleaned spec survivors + catalog others.

    Excludes stale_xmatch, tns_untyped, tns_ambiguous (78 untyped join SSL
    only, spec §6). 'context' rows are the 150 catalog others.
    """
    keep_quality = {"spectroscopic", "context"}
    mask = cleaned_truth["label_quality"].isin(keep_quality)
    return [str(x) for x in cleaned_truth.loc[mask, "object_id"].tolist()]


# ------------------------------------------------------------------ #
# Live seed (network) — Lasair objects,crossmatch_tns                 #
# ------------------------------------------------------------------ #


def build_live_truth(
    *,
    tns_auth: dict[str, dict],
    cache_dir: Path,
    page_size: int = 1000,
    max_pages: int = 50,
):
    """Build lsst_live_truth rows from the Lasair seed + TNS-bulk authority."""
    from scripts.harvest_ztf_lsst_associations import (
        _get,
        _record_disc_int_name,
        _record_lsst_id,
        _record_sep_arcsec,
        harvest_records,
        parse_ztf_ids,
    )

    records = harvest_records(
        cache_dir=cache_dir, page_size=page_size, max_pages=max_pages
    )
    rows: list[dict] = []
    for rec in records:
        oid = _record_lsst_id(rec)
        if not oid:
            continue
        tns_name = _get(rec, "tns_name", "crossmatch_tns.tns_name")
        auth = tns_auth.get(_norm_name(tns_name)) if tns_name else None
        # Sep cut: prefer TNS-bulk-authoritative coordinates when present
        # (the Lasair crossmatch row is a seed; its coords are the fallback).
        sep = _auth_sep_arcsec(rec, auth)
        if sep is None:
            sep = _record_sep_arcsec(rec)
        if not sep_ok(sep):
            continue
        tns_type = auth.get("type") if auth else _get(rec, "type", "crossmatch_tns.type")
        disc_date = (
            auth.get("discoverydate") if auth
            else _get(rec, "disc_date", "crossmatch_tns.disc_date")
        )
        first_mjd = _to_float(_get(rec, "firstDiaSourceMjdTai",
                                   "objects.firstDiaSourceMjdTai"))
        last_mjd = _to_float(_get(rec, "lastDiaSourceMjdTai",
                                  "objects.lastDiaSourceMjdTai"))
        # assoc-spec (B6): an LSST SPEC-tier row that ALSO carries a ZTF
        # association (parsed from disc_int_name) is the higher-confidence pool
        # G2's union keys on via ``label_source == 'ztf_assoc_spec'``.  Without
        # this the G2 assoc-spec term is structurally empty (nothing ever writes
        # that label_source into truth → gold).  Gate on the SAME
        # spectroscopic-quality decision make_truth_row uses so stale/tail
        # xmatches never masquerade as assoc-spec.
        disc_mjd = discovery_date_to_mjd(disc_date)
        quality = resolve_label_quality(
            tns_type, tns_name, disc_mjd, first_mjd, last_mjd
        )
        has_ztf = bool(parse_ztf_ids(_record_disc_int_name(rec)))
        label_source = (
            "ztf_assoc_spec"
            if (has_ztf and quality == "spectroscopic")
            else "lasair_crossmatch_tns"
        )
        rows.append(make_truth_row(
            object_id=oid,
            tns_name=tns_name,
            tns_type=tns_type,
            tns_prefix=_get(rec, "tns_prefix", "crossmatch_tns.tns_prefix"),
            discovery_date=disc_date,
            redshift=_get(rec, "z", "crossmatch_tns.z"),
            first_mjd=first_mjd,
            last_mjd=last_mjd,
            label_source=label_source,
        ))
    return truth_rows_to_df(rows)


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _auth_sep_arcsec(rec: dict, auth: dict | None) -> float | None:
    """Sep between the LSST object and the TNS-bulk-authoritative position.

    Returns None when either side lacks finite coordinates, letting the caller
    fall back to the Lasair-crossmatch sep.
    """
    if not auth:
        return None
    from scripts.crossmatch_lsst_to_ztf import separation_arcsec
    from scripts.harvest_ztf_lsst_associations import _get

    obj_ra = _to_float(_get(rec, "obj_ra", "ra", "objects.ra"))
    obj_dec = _to_float(_get(rec, "obj_decl", "decl", "dec", "objects.decl"))
    tns_ra = _to_float(auth.get("ra"))
    tns_dec = _to_float(auth.get("declination"))
    if None in (obj_ra, obj_dec, tns_ra, tns_dec):
        return None
    return separation_arcsec(obj_ra, obj_dec, tns_ra, tns_dec)


# ------------------------------------------------------------------ #
# Catalog negatives (network; cached + resumable) — B5               #
# ------------------------------------------------------------------ #

_SHERLOCK_SELECTED = (
    "objects.diaObjectId, objects.ra, objects.decl, "
    "sherlock_classifications.classification"
)


def harvest_sherlock_records(
    *,
    cache_dir: Path,
    page_size: int = 1000,
    max_pages: int = 50,
    conditions: str | None = None,
) -> list[dict]:
    """Paged Lasair ``objects,sherlock_classifications`` harvest (cached).

    Resumable: one JSON page per file; a short/empty page ends the harvest.
    ``conditions`` defaults to non-transient sherlock classes.
    """
    from debass_meta.access.lasair import LasairAdapter

    if conditions is None:
        classes = ",".join(f"'{c}'" for c in sorted(SHERLOCK_NEGATIVE_CLASSES))
        # ORDER BY makes offset pagination deterministic — without it MySQL may
        # repeat/skip rows across pages (duplicate object_ids, 2026-07-05 crash).
        conditions = (f"sherlock_classifications.classification IN ({classes}) "
                      f"ORDER BY objects.diaObjectId")

    cache_dir.mkdir(parents=True, exist_ok=True)
    adapter = LasairAdapter()
    records: list[dict] = []
    # Cache key includes the query so pages cached under an older (unordered)
    # conditions string are never reused.
    tag = hashlib.sha1(f"{_SHERLOCK_SELECTED}|{conditions}".encode()).hexdigest()[:8]
    for page in range(max_pages):
        page_path = cache_dir / f"sherlock_{tag}_page_{page:04d}.json"
        if page_path.exists():
            rows = json.loads(page_path.read_text())
        else:
            rows = adapter._query_api(
                selected=_SHERLOCK_SELECTED,
                tables="objects,sherlock_classifications",
                conditions=conditions,
                limit=page_size,
                offset=page * page_size,
                survey="lsst",
            )
            page_path.write_text(json.dumps(rows))
        records.extend(rows)
        print(f"  sherlock page {page}: {len(rows)} rows (total {len(records)})")
        if len(rows) < page_size:
            break
    return records


def build_catalog_negatives(
    *,
    cache_dir: Path,
    frozen_ids: set[str],
    page_size: int = 1000,
    max_pages: int = 50,
    target: int = 300,
):
    """Harvest NEW catalog negatives (B5) as a truth frame; report the count.

    The harvest is capped at ~10× ``target`` (B5 asks for ≥300, not for every
    sherlock-classified object on the sky): the first SCC build pulled 49,928
    negatives at max_pages=50, drowning the LSST truth in context rows. Pages
    are fetched only until the cap is satisfiable.
    """
    neg_cap = max(target * 10, page_size)
    pages_needed = min(max_pages, -(-neg_cap // page_size))  # ceil div
    records = harvest_sherlock_records(
        cache_dir=cache_dir, page_size=page_size, max_pages=pages_needed
    )
    rows = catalog_negative_rows(records, frozen_ids=frozen_ids)
    df = truth_rows_to_df(rows)
    df = dedupe_truth_frame(df)
    if len(df) > neg_cap:
        df = df.head(neg_cap).reset_index(drop=True)
    if len(df) < target:
        print(f"WARNING: catalog negatives {len(df)} < target {target} "
              f"(B5) — widen the harvest or raise --max-pages")
    return df


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tns-bulk", default="data/truth/tns_public.parquet",
                    help="Authoritative TNS bulk parquet (from download_tns_bulk.py)")
    ap.add_argument("--mode", choices=("live", "cohort-clean", "merge"),
                    default="cohort-clean",
                    help="live: Lasair harvest -> lsst_live_truth.parquet; "
                         "cohort-clean: re-derive the 2026-07-04 cohort + manifest; "
                         "merge: fold lsst_live_truth into object_truth_v11 "
                         "(TRUTH PLUMBING PIN — the training-build --truth file)")
    # merge inputs (P1 FINAL step: LSST-live rows win, keyed object_id)
    ap.add_argument("--base-truth", default="data/truth/object_truth_v11.parquet",
                    help="[merge] rederived ZTF v11 truth (from rederive_spec_truth.py)")
    ap.add_argument("--merge-live", default="data/truth/lsst_live_truth.parquet",
                    help="[merge] lsst_live_truth.parquet to fold in")
    ap.add_argument("--merged-out",
                    default="data/truth/object_truth_v11_merged.parquet",
                    help="[merge] output training-build truth file")
    # cohort-clean inputs
    ap.add_argument("--cohort", default="data/live_eval_20260704/cohort/merged.csv")
    ap.add_argument("--authoritative-truth",
                    default="data/live_eval_20260704/truth/object_truth.parquet",
                    help="Fallback type/discoverydate source when TNS bulk absent")
    ap.add_argument("--cleaned-out",
                    default="data/truth/object_truth_20260704_cleaned.parquet")
    ap.add_argument("--manifest-out", default="data/gold/lsst_live_locked_test.json")
    # live inputs
    ap.add_argument("--live-out", default="data/truth/lsst_live_truth.parquet")
    ap.add_argument("--cache-dir", default="data/truth/lasair_live_cache")
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--catalog-negatives", dest="catalog_negatives",
                    action="store_true", default=True,
                    help="Harvest NEW catalog negatives into the live truth (B5, default on)")
    ap.add_argument("--no-catalog-negatives", dest="catalog_negatives",
                    action="store_false")
    ap.add_argument("--catalog-negatives-target", type=int, default=300)
    ap.add_argument("--manifest-in", default="data/gold/lsst_live_locked_test.json",
                    help="Frozen benchmark manifest; its ids are excluded from catalog negatives")
    args = ap.parse_args()

    if args.mode == "merge":
        base_path = Path(args.base_truth)
        if not base_path.exists():
            raise SystemExit(
                f"--mode merge needs the rederived v11 truth at {base_path}; "
                f"run rederive_spec_truth.py first"
            )
        base = pd.read_parquet(base_path)
        live = None
        live_path = Path(args.merge_live)
        if live_path.exists():
            live = pd.read_parquet(live_path)
            print(f"loaded {len(live)} lsst_live rows from {live_path}")
        else:
            print(f"WARNING: no lsst_live truth at {live_path}; "
                  f"merged == base (ZTF-only until the live seed runs)")
        merged = merge_live_into_truth(base, live)
        out = Path(args.merged_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out, index=False)
        print(f"OK wrote {len(merged)} merged truth rows -> {out}")
        print("  quality:", merged["label_quality"].value_counts(dropna=False).to_dict())
        return

    tns_auth = load_tns_authoritative(Path(args.tns_bulk))
    print(f"TNS authoritative names: {len(tns_auth)}")

    if args.mode == "cohort-clean":
        cohort_df = pd.read_csv(args.cohort, dtype={"object_id": str})
        auth_truth = None
        if args.authoritative_truth and Path(args.authoritative_truth).exists():
            auth_truth = pd.read_parquet(args.authoritative_truth)
        cleaned = clean_cohort_truth(
            cohort_df=cohort_df, tns_auth=tns_auth, authoritative_truth=auth_truth,
        )
        # RUNBOOK: a truth row for EVERY cohort object; assert count parity.
        assert len(cleaned) == len(cohort_df), (
            f"cleaned truth rows {len(cleaned)} != cohort {len(cohort_df)}"
        )
        out = Path(args.cleaned_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(out, index=False)
        print(f"OK wrote {len(cleaned)} cleaned truth rows -> {out}")
        print("  quality:", cleaned["label_quality"].value_counts(dropna=False).to_dict())

        test_ids = cohort_manifest_test_ids(cleaned)
        man_out = Path(args.manifest_out)
        # APPEND-ONLY (B1/§6): merge with any existing manifest — previously
        # frozen ids (incl. even-hash spec arrivals appended by refresh) are
        # NEVER dropped, and the original frozen_utc is preserved.  Without this
        # the cohort-clean rewrite would silently drop appended TEST ids and
        # churn frozen_utc whenever the live-seed/crossmatch/type state changed.
        prior = load_prior_manifest(man_out)
        prior_ids = list((prior or {}).get("test_ids", []))
        prior_frozen = (prior or {}).get("frozen_utc")
        prior_excluded = dict((prior or {}).get("excluded_ids", {}))
        if prior is not None:
            print(f"  merging with existing manifest ({len(prior_ids)} frozen ids, "
                  f"{len(prior_excluded)} excluded, frozen_utc={prior_frozen}) — append-only")
        manifest = build_manifest(
            test_ids,
            frozen_utc=prior_frozen,
            prior_test_ids=prior_ids,
            excluded_ids=prior_excluded,
            policy="sha1(object_id) last-hex even->TEST(appended), odd->train/cal; "
                   "frozen ids never move (fusion_v11 B1/§6)",
            source=f"cohort-clean:{args.cohort}",
        )
        man_out.parent.mkdir(parents=True, exist_ok=True)
        man_out.write_text(json.dumps(manifest, indent=2))
        print(f"OK wrote benchmark manifest ({len(manifest['test_ids'])} test_ids; "
              f"{len(prior_ids)} preserved) -> {man_out}")
        return

    # mode == live
    if not tns_auth:
        raise SystemExit(
            "--mode live needs a TNS bulk parquet (--tns-bulk); run download_tns_bulk.py first"
        )
    live = build_live_truth(
        tns_auth=tns_auth,
        cache_dir=Path(args.cache_dir),
        page_size=args.page_size,
        max_pages=args.max_pages,
    )

    if args.catalog_negatives:
        frozen_ids: set[str] = set()
        man_in = Path(args.manifest_in)
        if man_in.exists():
            try:
                frozen_ids = {
                    str(x).strip()
                    for x in json.loads(man_in.read_text()).get("test_ids", [])
                }
            except Exception as exc:  # degrade loudly, never silently
                print(f"WARNING: could not read frozen manifest {man_in}: {exc}")
        try:
            neg = build_catalog_negatives(
                cache_dir=Path(args.cache_dir),
                frozen_ids=frozen_ids,
                page_size=args.page_size,
                max_pages=args.max_pages,
                target=args.catalog_negatives_target,
            )
            # Drop negatives that collide with an already-labeled live object.
            live_ids = set(live["object_id"].astype(str))
            if len(neg):
                neg = neg[~neg["object_id"].astype(str).isin(live_ids)]
            print(f"  catalog negatives harvested: {len(neg)}")
            live = pd.concat([live, neg], ignore_index=True)
        except Exception as exc:
            print(f"WARNING: catalog-negative harvest failed ({exc}); "
                  f"continuing with spec/untyped rows only")

    n_pre = len(live)
    live = dedupe_truth_frame(live)
    if len(live) != n_pre:
        print(f"  deduped {n_pre - len(live)} duplicate object_id rows "
              f"(quality-priority keep)")

    out = Path(args.live_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    live.to_parquet(out, index=False)
    print(f"OK wrote {len(live)} live truth rows -> {out}")
    print("  quality:", live["label_quality"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
