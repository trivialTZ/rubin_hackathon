#!/usr/bin/env python3
"""Build the fusion_v8 gold object-epoch snapshot table.

Extends the locked v6e2 gold builder (scripts/build_object_epoch_snapshots.py +
src/debass_meta/ingest/gold.py — reused by import, never reimplemented) with:

  1. Training-set expansion: object list = union of labels.csv ZTF IDs and every
     cached lightcurve stem (data/lightcurves/*.json) that has truth in
     data/truth/object_truth.parquet or data/truth/ztf_bts.parquet.  This
     includes the catalog-sourced 'other' harvest (Gaia-parallax stars, SIMBAD
     galaxies/QSO, asteroids, ALeRCE self-labels) from the context/weak truth
     tiers — their `label_source` / `label_quality` columns are passed through
     verbatim so the Stage-B trainer's provenance masking
     (multiclass_followup.apply_provenance_masking) can act on them.
  2. 50 EXT lightcurve features (debass_meta.features.lightcurve_ext) computed
     on the SAME truncated detection list as the base-51 features.
  3. 66 broker score-trajectory features (debass_meta.features.trajectory).
  4. A split manifest (data/gold/split_fusion_v8.json): the original objects in
     models/trust/metadata.json keep their train/cal/test assignment VERBATIM
     (asserted); new objects are assigned with group_train_cal_test_split
     (seed=42) and their would-be-test share is reassigned to train, so the
     locked v6e2 test set is byte-identical.
  5. A DP1 path (--dp1): data/gold/dp1_snapshots_fusion_v8.parquet =
     data/gold/dp1_snapshots_50k.parquet + EXT features (computed from
     data/lightcurves/dp1/*.parquet where cached, NaN elsewhere), preserving
     every existing column including the truth-catalog columns required by
     scripts/build_enrichment_metrics.py.

No-leakage argument
-------------------
Base and EXT features at n_det=N are functions of the lightcurve truncated to
exactly the first N positive detections (the truncation replicates
extract_features_at_each_epoch's preprocessing and is asserted to agree with it
row-by-row via n_det/alert_mjd).  Expert blocks use select_events_asof with
allow_unsafe_latest_snapshot=False, i.e. only events with
event_time_jd <= alert_jd (or static/rerun-safe scopes).  Trajectory features
enforce the same as-of rule inside build_trajectory_features.

Usage
-----
    python scripts/build_snapshots_fusion.py                       # full build
    python scripts/build_snapshots_fusion.py --smoke               # 200 objects
    python scripts/build_snapshots_fusion.py --skip-traj --skip-experts
    python scripts/build_snapshots_fusion.py --dp1                 # + DP1 table
    python scripts/build_snapshots_fusion.py --dp1-only            # DP1 only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd

from debass_meta.access.identifiers import infer_identifier_kind
from debass_meta.access.tns import map_tns_type_to_ternary
from debass_meta.features.detection import normalize_lightcurve
from debass_meta.features.lightcurve import (
    FEATURE_NAMES,
    _ensure_normalized,
    extract_features_at_each_epoch,
)
from debass_meta.ingest.gold import (
    _attach_expert_projection,
    _load_all_event_rows,
    _load_lightcurve,
    _load_truth_lookup,
    _resolve_lightcurve_path,
    _to_jd,
)
from debass_meta.models.splitters import group_train_cal_test_split
from debass_meta.projectors import ALL_EXPERT_KEYS, sanitize_expert_key

# ------------------------------------------------------------------ #
# fusion_v8 feature modules are implemented in parallel (G1/G2).      #
# Guard the imports so this builder degrades loudly-but-gracefully    #
# while they land: a stub emits zero EXT/traj columns and prints a    #
# warning, but never silently fabricates values.                      #
# ------------------------------------------------------------------ #
try:
    from debass_meta.features.lightcurve_ext import (  # type: ignore
        EXT_FEATURE_NAMES,
        extract_ext_features,
    )

    _EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only pre-integration
    EXT_FEATURE_NAMES: list[str] = []

    def extract_ext_features(detections: list[dict[str, Any]]) -> dict[str, float]:
        """Stub: lightcurve_ext not yet available — emits no EXT features."""
        return {name: float("nan") for name in EXT_FEATURE_NAMES}

    _EXT_AVAILABLE = False

try:
    from debass_meta.features.trajectory import (  # type: ignore
        TRAJ_FEATURE_NAMES,
        build_trajectory_features,
    )

    _TRAJ_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only pre-integration
    TRAJ_FEATURE_NAMES: list[str] = []
    build_trajectory_features = None
    _TRAJ_AVAILABLE = False


_DEFAULT_OUTPUT = "data/gold/object_epoch_snapshots_fusion_v8.parquet"
_DEFAULT_SPLIT_MANIFEST = "data/gold/split_fusion_v8.json"
_DEFAULT_DP1_OUTPUT = "data/gold/dp1_snapshots_fusion_v8.parquet"
_SMOKE_N_OBJECTS = 200
_PROGRESS_EVERY = 500


# ------------------------------------------------------------------ #
# BTS fine-type → ternary mapping (replicates scripts/fetch_ztf_bts)  #
# ------------------------------------------------------------------ #

def bts_type_to_ternary(bts_type: Any) -> str | None:
    """Map a BTS fine type to the metaDEBASS ternary label.

    This replicates ``scripts/fetch_ztf_bts.py::_normalise_bts._map`` exactly:
    BTS types are fed through :func:`map_tns_type_to_ternary` (the single
    source of truth also used by ``scripts/build_truth_multisource.py``),
    prefixed with ``"SN "`` when they do not already start with ``"SN"``.

    NOTE (pre-existing pipeline behaviour, replicated for parity, NOT fixed
    here): non-SN BTS types like ``CV``/``TDE``/``-`` fall through the
    ``"SN "``-prefix fallback to ``nonIa_snlike``.  The runtime parity check
    (:func:`check_bts_mapping_parity`) guarantees this function agrees with
    the ``ternary`` column already persisted in ztf_bts.parquet.
    """
    if bts_type is None or (isinstance(bts_type, float) and math.isnan(bts_type)):
        return None
    cleaned = str(bts_type).strip()
    if not cleaned:
        return None
    return map_tns_type_to_ternary(
        cleaned if cleaned.startswith("SN") else "SN " + cleaned
    )


def check_bts_mapping_parity(bts_df: pd.DataFrame, *, max_examples: int = 5) -> None:
    """Assert bts_type→ternary parity against the persisted ``ternary`` column.

    Raises AssertionError listing example mismatches — a mismatch means the
    BTS catalog was built with a different mapping than this script would
    apply, which would silently shift labels between builds.
    """
    if "bts_type" not in bts_df.columns or "ternary" not in bts_df.columns:
        return
    recomputed = bts_df["bts_type"].map(bts_type_to_ternary)
    stored = bts_df["ternary"].where(bts_df["ternary"].notna(), None)
    mismatch = recomputed.where(recomputed.notna(), None) != stored
    if bool(mismatch.any()):
        examples = bts_df.loc[mismatch, ["object_id", "bts_type", "ternary"]].head(max_examples)
        raise AssertionError(
            f"BTS ternary mapping parity FAILED on {int(mismatch.sum())} rows "
            f"(mapping drift between fetch_ztf_bts and this builder). "
            f"Examples:\n{examples.to_string(index=False)}"
        )


# ------------------------------------------------------------------ #
# Per-object feature extraction (multiprocessing worker)              #
# ------------------------------------------------------------------ #

def _truncated_detection_lists(
    detections: list[dict[str, Any]],
    *,
    max_n_det: int = 20,
) -> list[list[dict[str, Any]]]:
    """Return the truncated detection list for each epoch 1..min(len, max_n_det).

    Replicates the preprocessing inside
    :func:`debass_meta.features.lightcurve.extract_features_at_each_epoch`
    (normalize → positive filter → MJD sort → truncate) so that
    ``extract_ext_features`` consumes EXACTLY the same detection list as the
    base-51 extractor — the no-leakage proof is inherited, and the agreement
    is asserted per-row in :func:`_extract_object_rows`.
    """
    ndets = [_ensure_normalized(d) for d in detections]
    pos_dets = [d for d in ndets if d.get("is_positive", True)]
    if not pos_dets:
        pos_dets = ndets
    pos_dets.sort(key=lambda d: d.get("mjd") or 0)
    return [pos_dets[: i + 1] for i in range(min(len(pos_dets), max_n_det))]


def _extract_object_rows(
    object_id: str,
    *,
    lc_dir_str: str,
    max_n_det: int,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Worker: load one lightcurve and compute base-51 + EXT features per epoch.

    Returns (object_id, lightcurve_source, epoch_rows).  Each epoch row carries
    ``n_det``, ``alert_mjd``, all FEATURE_NAMES and all EXT_FEATURE_NAMES.
    """
    lc_dir = Path(lc_dir_str)
    lc_path, lc_source = _resolve_lightcurve_path(
        lc_dir, object_id=object_id, associations=None
    )
    if lc_path is None:
        return object_id, lc_source, []
    detections = _load_lightcurve(lc_path)
    if not detections:
        return object_id, lc_source, []

    epoch_feats = extract_features_at_each_epoch(detections, max_n_det=max_n_det)
    truncations = _truncated_detection_lists(detections, max_n_det=max_n_det)
    if len(epoch_feats) != len(truncations):
        raise RuntimeError(
            f"[{object_id}] truncation mismatch: {len(epoch_feats)} base epochs "
            f"vs {len(truncations)} truncated lists"
        )

    rows: list[dict[str, Any]] = []
    for feats, truncated in zip(epoch_feats, truncations):
        n_det = int(feats["n_det"])
        alert_mjd = float(feats["alert_mjd"])
        last_mjd = float(truncated[-1].get("mjd") or 0)
        if n_det != len(truncated) or abs(last_mjd - alert_mjd) > 1e-9:
            raise RuntimeError(
                f"[{object_id}] EXT truncation diverged from base extractor at "
                f"n_det={n_det}: len={len(truncated)}, alert_mjd={alert_mjd} "
                f"vs last_mjd={last_mjd}"
            )
        row = dict(feats)
        if EXT_FEATURE_NAMES:
            ext = extract_ext_features(truncated)
            for name in EXT_FEATURE_NAMES:
                row[name] = ext.get(name, float("nan"))
        rows.append(row)
    return object_id, lc_source, rows


# ------------------------------------------------------------------ #
# Object list + truth                                                 #
# ------------------------------------------------------------------ #

def _load_labels_ztf_ids(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        return []
    with open(labels_path) as fh:
        ids = [row["object_id"] for row in csv.DictReader(fh)]
    return [o for o in ids if infer_identifier_kind(o) == "ztf_object_id"]


def resolve_object_list(
    *,
    labels_path: Path,
    lc_dir: Path,
    truth_ids: set[str],
    bts_ids: set[str],
) -> list[str]:
    """Union of labels.csv ZTF IDs and lightcurve stems that have truth."""
    label_ids = set(_load_labels_ztf_ids(labels_path))
    stems = {p.stem for p in lc_dir.glob("*.json")}
    stems_with_truth = stems & (truth_ids | bts_ids)
    return sorted(label_ids | stems_with_truth)


BTS_UNCLASSIFIED_QUALITY = "bts_unclassified"


def _stratified_smoke_sample(
    object_ids: list[str],
    truth_lookup: dict[str, dict[str, Any]],
    bts_fallback: dict[str, dict[str, Any]],
    *,
    n: int,
    seed: int,
) -> list[str]:
    """Class-stratified smoke sample so a --smoke build always carries all
    three ternary classes (a first-N alphabetical slice can be single-class,
    which breaks downstream multiclass training)."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = {}
    for oid in object_ids:
        entry = truth_lookup.get(oid) or bts_fallback.get(oid) or {}
        cls = entry.get("final_class_ternary") or "_unlabeled"
        by_class.setdefault(str(cls), []).append(oid)
    per_class = max(1, n // max(1, len(by_class)))
    chosen: list[str] = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        take = min(per_class, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[i] for i in sorted(idx))
    remainder = [o for o in object_ids if o not in set(chosen)]
    if len(chosen) < n and remainder:
        idx = rng.choice(len(remainder), size=min(n - len(chosen), len(remainder)), replace=False)
        chosen.extend(remainder[i] for i in sorted(idx))
    return sorted(chosen[:n])


def build_truth_entries(
    truth_path: Path,
    bts_path: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    """Return (object_truth lookup, BTS fallback lookup, demoted unclassified IDs).

    object_truth is the primary source (it already merges BTS via
    build_truth_multisource); the BTS fallback only fires for objects present
    in ztf_bts.parquet but absent from object_truth (possible on SCC).
    The BTS mapping parity check runs here at every build.

    Label-honesty demotion: the legacy mapping labels BTS ``bts_type == '-'``
    (spectroscopically *observed* but never classified; 3,150 objects locally)
    as ``nonIa_snlike``/``spectroscopic`` — a fabricated class.  Any object
    whose truth row came from ``ztf_bts`` and is unclassified in the BTS
    catalog is demoted here: ternary class and follow proxy are cleared and
    ``label_quality`` becomes :data:`BTS_UNCLASSIFIED_QUALITY`, which no
    training loss or spec(+tns_untyped) eval slice ever selects.
    """
    truth_lookup = _load_truth_lookup(truth_path)
    bts_fallback: dict[str, dict[str, Any]] = {}
    demoted_ids: set[str] = set()
    if bts_path is not None and bts_path.exists():
        bts_df = pd.read_parquet(bts_path)
        check_bts_mapping_parity(bts_df)
        print(f"  BTS mapping parity check PASSED on {len(bts_df):,} rows", flush=True)
        unclassified = {
            str(oid)
            for oid, btype in zip(bts_df["object_id"], bts_df.get("bts_type", []))
            if str(btype).strip() == "-"
        }
        for oid in unclassified:
            entry = truth_lookup.get(oid)
            if entry is not None and str(entry.get("label_source")) == "ztf_bts":
                entry["final_class_ternary"] = None
                entry["follow_proxy"] = None
                entry["label_quality"] = BTS_UNCLASSIFIED_QUALITY
                demoted_ids.add(oid)
        if demoted_ids:
            print(
                f"  Label-honesty demotion: {len(demoted_ids):,} BTS-unclassified "
                f"('-') objects cleared of fabricated nonIa_snlike/spectroscopic "
                f"labels (label_quality='{BTS_UNCLASSIFIED_QUALITY}')",
                flush=True,
            )
        for _, row in bts_df.iterrows():
            oid = str(row.get("object_id") or "")
            if not oid or oid in truth_lookup:
                continue
            ternary = row.get("ternary")
            if ternary is None or (isinstance(ternary, float) and math.isnan(ternary)):
                continue
            if str(row.get("bts_type")).strip() == "-":
                # Unclassified: keep provenance, never fabricate a class.
                demoted_ids.add(oid)
                bts_fallback[oid] = {
                    "object_id": oid,
                    "final_class_ternary": None,
                    "follow_proxy": None,
                    "label_source": "ztf_bts",
                    "label_quality": BTS_UNCLASSIFIED_QUALITY,
                    "bts_type": row.get("bts_type"),
                    "tns_name": row.get("tns_name"),
                }
                continue
            bts_fallback[oid] = {
                "object_id": oid,
                "final_class_ternary": str(ternary),
                "follow_proxy": int(str(ternary) == "snia"),
                "label_source": "ztf_bts",
                "label_quality": "spectroscopic",
                "bts_type": row.get("bts_type"),
                "tns_name": row.get("tns_name"),
            }
    return truth_lookup, bts_fallback, demoted_ids


# ------------------------------------------------------------------ #
# Split manifest                                                      #
# ------------------------------------------------------------------ #

def build_split_manifest(
    snapshot_object_ids: list[str],
    truth_for: dict[str, dict[str, Any]],
    trust_metadata_path: Path,
    *,
    seed: int = 42,
    smoke: bool = False,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Build the fusion_v8 split manifest.

    Original objects (those in models/trust/metadata.json) keep their
    train/cal/test assignment VERBATIM — asserted.  New objects are assigned
    via group_train_cal_test_split(seed=seed) and their would-be-test share is
    reassigned to train, so the realized test set is a subset (in full builds:
    byte-identical) of the locked v6e2 test set.
    """
    with open(trust_metadata_path) as fh:
        md = json.load(fh)
    orig_train = {str(o) for o in md["train_ids"]}
    orig_cal = {str(o) for o in md["cal_ids"]}
    orig_test = {str(o) for o in md["test_ids"]}
    assert orig_train.isdisjoint(orig_cal) and orig_train.isdisjoint(orig_test) and \
        orig_cal.isdisjoint(orig_test), "locked v6e2 split has overlapping IDs"
    orig_all = orig_train | orig_cal | orig_test

    objs = {str(o) for o in snapshot_object_ids}
    if not smoke:
        missing = orig_all - objs
        assert not missing, (
            f"{len(missing)} original locked-split objects missing from the "
            f"fusion snapshot (e.g. {sorted(missing)[:5]}) — the locked test "
            f"set would no longer be byte-identical."
        )

    new_ids = sorted(objs - orig_all)
    new_labels: dict[str, str | None] = {}
    for oid in new_ids:
        entry = truth_for.get(oid) or {}
        label = entry.get("final_class_ternary")
        if isinstance(label, float) and math.isnan(label):
            label = None
        new_labels[oid] = str(label) if label is not None else None
    split = group_train_cal_test_split(new_labels, seed=seed)

    # Would-be-test new objects are reassigned to train: new objects enter
    # train/cal only, keeping the headline test set identical to v6e2.
    n_reassigned = len(split.test_ids)
    train_ids = (orig_train & objs) | split.train_ids | split.test_ids
    cal_ids = (orig_cal & objs) | split.cal_ids
    test_ids = orig_test & objs

    # --- integrity asserts (loud failures) ---
    assert train_ids.isdisjoint(cal_ids) and train_ids.isdisjoint(test_ids) and \
        cal_ids.isdisjoint(test_ids), "fusion_v8 split has overlapping IDs"
    assert train_ids | cal_ids | test_ids == objs, "fusion_v8 split does not cover snapshot"
    assert test_ids <= orig_test, "new objects leaked into the locked test set"
    for oid in objs & orig_all:  # verbatim preservation
        expected = "train" if oid in orig_train else ("cal" if oid in orig_cal else "test")
        actual = "train" if oid in train_ids else ("cal" if oid in cal_ids else "test")
        assert actual == expected, (
            f"original object {oid} moved from {expected} to {actual} — "
            f"locked split must be preserved verbatim"
        )

    return {
        "version": "fusion_v8",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "smoke": smoke,
        "source_trust_metadata": str(trust_metadata_path),
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "n_objects": len(objs),
        "n_original": len(objs & orig_all),
        "n_new": len(new_ids),
        "n_new_test_reassigned_to_train": n_reassigned,
        "counts": {
            "train": len(train_ids),
            "cal": len(cal_ids),
            "test": len(test_ids),
        },
        "train_ids": sorted(train_ids),
        "cal_ids": sorted(cal_ids),
        "test_ids": sorted(test_ids),
        "new_object_ids": new_ids,
    }


# ------------------------------------------------------------------ #
# Main builder                                                        #
# ------------------------------------------------------------------ #

def build_fusion_snapshots(
    *,
    lc_dir: Path,
    silver_dir: Path,
    truth_path: Path,
    bts_path: Path | None,
    labels_path: Path,
    trust_metadata_path: Path,
    output_path: Path,
    split_manifest_path: Path,
    max_n_det: int = 20,
    n_jobs: int = 8,
    seed: int = 42,
    smoke: bool = False,
    limit: int | None = None,
    skip_traj: bool = False,
    skip_experts: bool = False,
) -> Path:
    t0 = time.time()
    truth_lookup, bts_fallback, demoted_ids = build_truth_entries(truth_path, bts_path)
    truth_ids = set(truth_lookup.keys())
    # BTS IDs already merged into object_truth are covered by truth_ids;
    # the fallback dict holds the BTS-only remainder.
    bts_ids = set(bts_fallback.keys())

    # Demoted BTS-unclassified objects carry no usable label: skip building
    # their epoch rows UNLESS they are pinned by labels.csv or by the locked
    # v6e2 split (those keep rows, with cleared labels, so the locked test
    # set stays structurally intact while the fabricated labels are gone).
    pinned = set(_load_labels_ztf_ids(labels_path))
    if trust_metadata_path is not None and Path(trust_metadata_path).exists():
        with open(trust_metadata_path) as fh:
            _meta = json.load(fh)
        pinned |= {
            str(o)
            for key in ("train_ids", "cal_ids", "test_ids")
            for o in _meta.get(key, [])
        }
    skippable = demoted_ids - pinned
    if skippable:
        truth_ids -= skippable
        bts_ids -= skippable
        print(
            f"  Skipping {len(skippable):,} demoted BTS-unclassified objects "
            f"with no other label evidence ({len(demoted_ids - skippable):,} "
            f"demoted objects kept because labels.csv/locked split pins them)",
            flush=True,
        )

    object_ids = resolve_object_list(
        labels_path=labels_path, lc_dir=lc_dir, truth_ids=truth_ids, bts_ids=bts_ids
    )
    if smoke:
        object_ids = _stratified_smoke_sample(
            object_ids, truth_lookup, bts_fallback, n=_SMOKE_N_OBJECTS, seed=seed
        )
    if limit is not None:
        object_ids = object_ids[:limit]
    n_total = len(object_ids)
    print(f"fusion_v8 snapshot builder: {n_total:,} objects "
          f"(EXT={'on' if _EXT_AVAILABLE else 'STUBBED'}, "
          f"traj={'on' if (_TRAJ_AVAILABLE and not skip_traj) else 'off'}, "
          f"experts={'off' if skip_experts else 'on'})", flush=True)
    if not _EXT_AVAILABLE:
        print("  WARNING: debass_meta.features.lightcurve_ext not importable — "
              "EXT features omitted (stub). Re-run after G1 lands.", flush=True)

    def _truth_for(oid: str) -> dict[str, Any]:
        return truth_lookup.get(oid) or bts_fallback.get(oid) or {}

    # Visibility into the catalog-'other' harvest (correction #1)
    n_catalog_other = sum(
        1 for oid in object_ids
        if _truth_for(oid).get("final_class_ternary") == "other"
        and _truth_for(oid).get("label_quality") in ("context", "weak")
    )
    print(f"  catalog-'other' harvest (context/weak tiers): {n_catalog_other:,} objects", flush=True)

    # ---- Pass 1: base-51 + EXT features (multiprocessing Pool) ----
    worker = partial(_extract_object_rows, lc_dir_str=str(lc_dir), max_n_det=max_n_det)
    results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    t1 = time.time()
    if n_jobs > 1 and n_total > 1:
        with Pool(processes=n_jobs) as pool:
            for i, (oid, lc_source, rows) in enumerate(
                pool.imap_unordered(worker, object_ids, chunksize=16)
            ):
                results[oid] = (lc_source, rows)
                if (i + 1) % _PROGRESS_EVERY == 0:
                    print(f"  features: {i + 1:,}/{n_total:,} objects "
                          f"({time.time() - t1:.0f}s)", flush=True)
    else:
        for i, oid in enumerate(object_ids):
            oid, lc_source, rows = worker(oid)
            results[oid] = (lc_source, rows)
            if (i + 1) % _PROGRESS_EVERY == 0:
                print(f"  features: {i + 1:,}/{n_total:,} objects "
                      f"({time.time() - t1:.0f}s)", flush=True)
    print(f"  features done: {n_total:,} objects in {time.time() - t1:.0f}s", flush=True)

    # ---- Events index for expert attach (reuses gold builder machinery) ----
    events_index: dict[tuple[str, str], Any] = {}
    if not skip_experts:
        t2 = time.time()
        events_df = _load_all_event_rows(silver_dir)
        if len(events_df) > 0:
            object_set = set(object_ids)
            events_df = events_df[events_df["object_id"].astype(str).isin(object_set)]
            for key, group in events_df.groupby(["object_id", "expert_key"]):
                events_index[key] = group
        print(f"  indexed {len(events_index):,} (object, expert) event groups "
              f"in {time.time() - t2:.0f}s", flush=True)

    # ---- Pass 2: assemble gold rows ----
    all_rows: list[dict[str, Any]] = []
    n_skipped = 0
    for i, object_id in enumerate(object_ids):
        lc_source, epoch_rows = results[object_id]
        if not epoch_rows:
            n_skipped += 1
            continue
        truth = _truth_for(object_id)
        id_kind = infer_identifier_kind(object_id)
        survey = "LSST" if id_kind == "lsst_dia_object_id" else "ZTF"

        for feats in epoch_rows:
            n_det = int(feats["n_det"])
            alert_jd = _to_jd(float(feats.pop("alert_mjd")))
            base_row: dict[str, Any] = {
                "object_id": object_id,
                "n_det": n_det,
                "alert_jd": alert_jd,
                "survey": survey,
                "lightcurve_source_object_id": lc_source["object_id"],
                "lightcurve_source_identifier_kind": lc_source["identifier_kind"],
                "lightcurve_association_kind": lc_source["association_kind"],
                "lightcurve_association_source": lc_source["association_source"],
                "lightcurve_association_sep_arcsec": lc_source["association_sep_arcsec"],
            }
            for feature_name in FEATURE_NAMES:
                base_row[feature_name] = feats.get(feature_name)
            for feature_name in EXT_FEATURE_NAMES:
                base_row[feature_name] = feats.get(feature_name)

            # Label columns: passed through verbatim (label_source carries
            # provenance strings like 'alerce_self_label' / 'fink_xm_gaia_stellar'
            # so downstream provenance masking can act structurally).
            base_row["target_class"] = truth.get("final_class_ternary")
            base_row["target_follow_proxy"] = truth.get("follow_proxy")
            base_row["label_source"] = truth.get("label_source")
            base_row["label_quality"] = truth.get("label_quality")

            if skip_experts:
                for expert_key in ALL_EXPERT_KEYS:
                    san = sanitize_expert_key(expert_key)
                    base_row[f"avail__{san}"] = 0.0
                    base_row[f"exact__{san}"] = 0.0
                    base_row[f"reason__{san}"] = "experts skipped (--skip-experts)"
            else:
                for expert_key in ALL_EXPERT_KEYS:
                    _attach_expert_projection(
                        base_row,
                        events_index,
                        object_id=object_id,
                        expert_key=expert_key,
                        alert_jd=alert_jd,
                        allow_unsafe_latest_snapshot=False,
                    )
            all_rows.append(base_row)

        if (i + 1) % _PROGRESS_EVERY == 0:
            print(f"  assemble: {i + 1:,}/{n_total:,} objects "
                  f"({len(all_rows):,} rows)", flush=True)

    if not all_rows:
        raise ValueError(f"No epoch rows built from {lc_dir}")
    print(f"  assembled {len(all_rows):,} gold rows "
          f"({n_skipped:,} objects skipped: no lightcurve)", flush=True)

    df = pd.DataFrame(all_rows)

    # ---- Pass 3: trajectory features ----
    if not skip_traj and _TRAJ_AVAILABLE and build_trajectory_features is not None:
        t3 = time.time()
        broker_events_path = silver_dir / "broker_events.parquet"
        if broker_events_path.exists():
            events = pd.read_parquet(broker_events_path)
            events = events[events["object_id"].astype(str).isin(set(df["object_id"].astype(str)))]
            gold_keys = df[["object_id", "n_det", "alert_jd"]].copy()
            traj = build_trajectory_features(events, gold_keys)
            traj_cols = [c for c in TRAJ_FEATURE_NAMES if c in traj.columns]
            df = df.merge(
                traj[["object_id", "n_det", *traj_cols]],
                on=["object_id", "n_det"],
                how="left",
                validate="one_to_one",
            )
            print(f"  trajectory features attached ({len(traj_cols)} cols) "
                  f"in {time.time() - t3:.0f}s", flush=True)
        else:
            for col in TRAJ_FEATURE_NAMES:
                df[col] = np.nan
            print(f"  no broker_events.parquet — trajectory columns set to NaN", flush=True)
    elif TRAJ_FEATURE_NAMES:
        # Keep schema stable when trajectory is skipped but names are known.
        for col in TRAJ_FEATURE_NAMES:
            df[col] = np.nan
        print("  trajectory skipped — columns present, all NaN", flush=True)
    else:
        print("  WARNING: debass_meta.features.trajectory not importable — "
              "trajectory columns omitted. Re-run after G2 lands.", flush=True)

    # Loud failure: Stage-A/B inputs must never be present at build time.
    forbidden = [c for c in df.columns if c.startswith("q__") or c.startswith("trust_source__")]
    assert not forbidden, f"label-stage columns leaked into the snapshot: {forbidden}"

    # ---- Split manifest ----
    manifest = build_split_manifest(
        sorted(df["object_id"].astype(str).unique()),
        {oid: _truth_for(oid) for oid in df["object_id"].astype(str).unique()},
        trust_metadata_path,
        seed=seed,
        smoke=smoke or (limit is not None),
        snapshot_path=output_path,
    )
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  split manifest → {split_manifest_path} "
          f"(train={manifest['counts']['train']:,} cal={manifest['counts']['cal']:,} "
          f"test={manifest['counts']['test']:,}; "
          f"{manifest['n_new_test_reassigned_to_train']:,} new test→train)", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote fusion_v8 snapshots → {output_path} "
          f"({len(df):,} rows × {len(df.columns):,} cols, "
          f"{time.time() - t0:.0f}s total)", flush=True)
    return output_path


# ------------------------------------------------------------------ #
# DP1 path                                                            #
# ------------------------------------------------------------------ #

def build_dp1_fusion(
    dp1_snapshots_path: Path,
    dp1_lc_dir: Path,
    output_path: Path,
) -> Path:
    """Append EXT features to the DP1 snapshot table (eval-only).

    All existing columns — including the truth-catalog columns
    (is_published_sn, is_gaia_known_star, is_known_variable, simbad_main_type)
    required by scripts/build_enrichment_metrics.py — are preserved verbatim.
    EXT features are computed from cached DP1 lightcurve parquets where they
    exist, NaN elsewhere.  Each EXT row is computed from the lightcurve
    truncated to that row's n_det (same no-leakage truncation as the
    original DP1 builder, asserted via alert_jd agreement).
    """
    df = pd.read_parquet(dp1_snapshots_path)
    original_columns = list(df.columns)
    n_rows_in = len(df)

    for name in EXT_FEATURE_NAMES:
        df[name] = np.nan
    if not EXT_FEATURE_NAMES:
        print("  WARNING: EXT feature module unavailable — DP1 table copied "
              "without EXT columns.", flush=True)

    lc_paths = sorted(dp1_lc_dir.glob("*.parquet")) if dp1_lc_dir.exists() else []
    oid_str = df["object_id"].astype(str)
    max_n_det = int(df["n_det"].max()) if len(df) else 20
    n_matched_objects = 0
    n_rows_filled = 0
    n_alert_mismatch = 0

    for p in lc_paths:
        if not EXT_FEATURE_NAMES:
            break
        oid = p.stem
        idx = df.index[oid_str == oid]
        if len(idx) == 0 and "diaObjectId" in df.columns:
            idx = df.index[df["diaObjectId"].astype(str) == oid]
        if len(idx) == 0:
            continue
        lc = pd.read_parquet(p)
        if "midpointMjdTai" in lc.columns:
            lc = lc.sort_values("midpointMjdTai")
        detections = lc.to_dict("records")
        if not detections:
            continue
        normalized = normalize_lightcurve(detections, survey="LSST")
        truncations = _truncated_detection_lists(normalized, max_n_det=max_n_det)
        n_matched_objects += 1
        for row_idx in idx:
            n_det = int(df.at[row_idx, "n_det"])
            if n_det < 1 or n_det > len(truncations):
                continue
            truncated = truncations[n_det - 1]
            expected_jd = _to_jd(float(truncated[-1].get("mjd") or 0))
            row_jd = float(df.at[row_idx, "alert_jd"])
            if abs(expected_jd - row_jd) > 1e-6:
                # Detection-order disagreement with the original DP1 builder:
                # leave EXT NaN rather than attach mismatched-epoch features.
                n_alert_mismatch += 1
                continue
            ext = extract_ext_features(truncated)
            for name in EXT_FEATURE_NAMES:
                df.at[row_idx, name] = ext.get(name, float("nan"))
            n_rows_filled += 1

    assert len(df) == n_rows_in, "DP1 row count changed — must be preserved"
    missing = [c for c in original_columns if c not in df.columns]
    assert not missing, f"DP1 columns dropped: {missing}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote DP1 fusion snapshots → {output_path} "
          f"({len(df):,} rows; EXT filled on {n_rows_filled:,} rows across "
          f"{n_matched_objects:,} cached-LC objects; "
          f"{n_alert_mismatch:,} alert_jd mismatches left NaN)", flush=True)
    return output_path


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--lc-dir", default="data/lightcurves")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--truth", default="data/truth/object_truth.parquet")
    parser.add_argument("--bts", default="data/truth/ztf_bts.parquet")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--trust-metadata", default="models/trust/metadata.json",
                        help="Locked v6e2 split (original IDs preserved verbatim)")
    parser.add_argument("--output", default=None,
                        help=f"Snapshot parquet (default {_DEFAULT_OUTPUT})")
    parser.add_argument("--split-manifest", default=None,
                        help=f"Split manifest JSON (default {_DEFAULT_SPLIT_MANIFEST})")
    parser.add_argument("--max-n-det", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help=f"Build only the first {_SMOKE_N_OBJECTS} objects "
                             f"(outputs get a .smoke suffix unless overridden)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Build only the first N objects (dev shortcut)")
    parser.add_argument("--skip-traj", action="store_true",
                        help="Skip trajectory features (columns NaN if names known)")
    parser.add_argument("--skip-experts", action="store_true",
                        help="Skip expert blocks (avail__*=0.0 for schema stability)")
    parser.add_argument("--dp1", action="store_true",
                        help="Also build the DP1 fusion table after the main build")
    parser.add_argument("--dp1-only", action="store_true",
                        help="Build ONLY the DP1 fusion table")
    parser.add_argument("--dp1-snapshots", default="data/gold/dp1_snapshots_50k.parquet")
    parser.add_argument("--dp1-lc-dir", default="data/lightcurves/dp1")
    parser.add_argument("--dp1-output", default=_DEFAULT_DP1_OUTPUT)
    args = parser.parse_args()

    dev_run = args.smoke or (args.limit is not None)
    output = Path(args.output) if args.output else Path(
        _DEFAULT_OUTPUT.replace(".parquet", ".smoke.parquet") if dev_run else _DEFAULT_OUTPUT
    )
    split_manifest = Path(args.split_manifest) if args.split_manifest else Path(
        _DEFAULT_SPLIT_MANIFEST.replace(".json", ".smoke.json") if dev_run else _DEFAULT_SPLIT_MANIFEST
    )

    if not args.dp1_only:
        build_fusion_snapshots(
            lc_dir=Path(args.lc_dir),
            silver_dir=Path(args.silver_dir),
            truth_path=Path(args.truth),
            bts_path=Path(args.bts) if args.bts else None,
            labels_path=Path(args.labels),
            trust_metadata_path=Path(args.trust_metadata),
            output_path=output,
            split_manifest_path=split_manifest,
            max_n_det=args.max_n_det,
            n_jobs=args.n_jobs,
            seed=args.seed,
            smoke=args.smoke,
            limit=args.limit,
            skip_traj=args.skip_traj,
            skip_experts=args.skip_experts,
        )

    if args.dp1 or args.dp1_only:
        build_dp1_fusion(
            Path(args.dp1_snapshots),
            Path(args.dp1_lc_dir),
            Path(args.dp1_output),
        )


if __name__ == "__main__":
    main()
