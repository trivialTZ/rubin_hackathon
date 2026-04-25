"""DP1-specific gold-snapshot builder.

Reads per-object DiaSource parquets under data/lightcurves/dp1/, runs the
51-feature extractor plus the 4 working local experts (supernnova,
alerce_lc, lc_features_bv, salt3_chi2) at each n_det epoch, and emits a
parquet row in v5d schema so that v5d's trust+followup models can score
it without retraining.

Broker-only experts are recorded as avail=0 (NaN projections). Per
CLAUDE.md and the feature-importance analysis, those 4 experts contribute
<1% to the followup decision on DP1, so NaN pass-through via LightGBM
native handling is safe.

This module intentionally does **not** re-implement the full
`ingest.gold` pipeline — it bypasses silver/bronze entirely because
DP1 input is already clean TAP output and there are no broker events
to join against.

Plan reference: docs/execplan_v6_dp1_validation.md §5 Stage 3.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from debass_meta.features.detection import normalize_lightcurve
from debass_meta.features.lightcurve import (
    FEATURE_NAMES,
    extract_features_at_each_epoch,
)
from debass_meta.projectors import (
    ALL_EXPERT_KEYS,
    project_expert_events,
    sanitize_expert_key,
)

# The 4 working local experts per plan §5 Stage 3 (SCC-verified 2026-04-24).
LOCAL_EXPERT_KEYS = ("supernnova", "alerce_lc", "lc_features_bv", "salt3_chi2", "oracle_lsst")


def _load_lightcurve(path: Path) -> list[dict[str, Any]]:
    """Load a DP1 DiaSource parquet → list of detection dicts, MJD-sorted."""
    df = pd.read_parquet(path)
    if len(df) == 0:
        return []
    # The parquet columns are a superset of what the LSST normalizer needs:
    # band, midpointMjdTai, psfFlux, psfFluxErr, reliability, etc.
    return df.sort_values("midpointMjdTai").to_dict("records")


def _mjd_to_jd(mjd: float) -> float:
    return float(mjd) + 2400000.5


def _local_outputs_to_events(
    expert_key: str,
    class_probabilities: dict[str, float],
    object_id: str,
    alert_jd: float,
    n_det: int,
) -> list[dict[str, Any]]:
    """Convert ExpertOutput.class_probabilities → events suitable for projector.

    Matches the shape produced by `ingest.gold._local_record_to_events` so that
    `project_expert_events(expert_key, events)` returns a proper ternary dict.
    """
    out: list[dict[str, Any]] = []
    for class_name, probability in (class_probabilities or {}).items():
        if probability is None or not math.isfinite(float(probability)):
            continue
        out.append({
            "object_id": object_id,
            "broker": "local",
            "classifier": expert_key,
            "expert_key": expert_key,
            "class_name": class_name,
            "semantic_type": "probability",
            "event_scope": "rerun_local",
            "event_time_jd": float(alert_jd),
            "n_det": int(n_det),
            "raw_label_or_score": float(probability),
            "canonical_projection": float(probability),
            "availability": True,
            "temporal_exactness": "rerun_exact",
            "survey": "LSST",
        })
    return out


def _init_experts() -> dict[str, Any]:
    """Instantiate the 4 working local experts. Heavy (torch load) — call once."""
    from debass_meta.experts.local.supernnova import SuperNNovaExpert
    from debass_meta.experts.local.alerce_lc import AlerceLCExpert
    from debass_meta.experts.local.lc_features import LcFeaturesExpert
    from debass_meta.experts.local.salt3_fit import Salt3Chi2Expert
    from debass_meta.experts.local.oracle_local import OracleLocalExpert

    return {
        "supernnova": SuperNNovaExpert(),
        "alerce_lc": AlerceLCExpert(),
        "lc_features_bv": LcFeaturesExpert(),
        "salt3_chi2": Salt3Chi2Expert(),
        "oracle_lsst": OracleLocalExpert(),
    }


def _expert_output_to_projection_cols(
    expert_key: str,
    out: Any,
    object_id: str,
    epoch_jd: float,
    n_det: int,
) -> dict[str, Any]:
    """Convert an ExpertOutput → snapshot projection columns."""
    san = sanitize_expert_key(expert_key)
    cols: dict[str, Any] = {}
    if out is None or not getattr(out, "available", False) or not getattr(out, "class_probabilities", None):
        cols[f"avail__{san}"] = 0.0
        cols[f"exact__{san}"] = 0.0
        cols[f"reason__{san}"] = "stub mode (weights missing)" if (out is None or not getattr(out, "available", True)) else "no probabilities"
        return cols

    events = _local_outputs_to_events(
        expert_key, out.class_probabilities, object_id, epoch_jd, n_det
    )
    projected = project_expert_events(expert_key, events)
    cols[f"avail__{san}"] = 1.0
    cols[f"exact__{san}"] = 1.0
    cols[f"temporal_exactness__{san}"] = "rerun_exact"
    cols[f"source_event_time_jd__{san}"] = float(epoch_jd)
    cols[f"event_count__{san}"] = 1.0
    cols[f"prediction_type__{san}"] = projected.get("prediction_type")
    cols[f"mapped_pred_class__{san}"] = projected.get("mapped_pred_class")
    for key in ("p_snia", "p_nonIa_snlike", "p_other", "top1_prob", "margin", "entropy"):
        value = projected.get(key)
        if value is not None:
            cols[f"proj__{san}__{key}"] = float(value)
    return cols


def _error_expert_cols(expert_key: str, reason: str) -> dict[str, Any]:
    san = sanitize_expert_key(expert_key)
    return {
        f"avail__{san}": 0.0,
        f"exact__{san}": 0.0,
        f"reason__{san}": reason,
    }


def _score_one_epoch_serial(
    experts: dict[str, Any],
    object_id: str,
    lightcurve: list[dict[str, Any]],
    epoch_jd: float,
    n_det: int,
    skip_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Serial per-expert inference for the non-batch experts."""
    cols: dict[str, Any] = {}
    for expert_key in LOCAL_EXPERT_KEYS:
        if expert_key in skip_keys:
            continue
        expert = experts.get(expert_key)
        if expert is None:
            cols.update(_error_expert_cols(expert_key, "expert not loaded"))
            continue
        try:
            out = expert.predict_epoch(object_id, lightcurve, epoch_jd)
        except Exception as exc:
            cols.update(_error_expert_cols(expert_key, f"inference error: {type(exc).__name__}"))
            continue
        cols.update(_expert_output_to_projection_cols(
            expert_key, out, object_id, epoch_jd, n_det,
        ))
    return cols


def _unavailable_expert_cols(expert_key: str) -> dict[str, Any]:
    """Mark a non-local expert as unavailable (DP1 isn't indexed by brokers)."""
    san = sanitize_expert_key(expert_key)
    return {
        f"avail__{san}": 0.0,
        f"exact__{san}": 0.0,
        f"temporal_exactness__{san}": None,
        f"source_event_time_jd__{san}": None,
        f"reason__{san}": "broker does not index DP1",
    }


def build_dp1_snapshots(
    lc_dir: Path,
    truth_path: Path,
    out_path: Path,
    *,
    max_n_det: int = 20,
    only_object_ids: list[int] | None = None,
    limit: int | None = None,
    snn_batch_size: int = 2000,
) -> Path:
    """Build the DP1 gold snapshot table.

    Implementation is a three-pass pipeline:
      1. Walk every cached DP1 lightcurve, run `extract_features_at_each_epoch`
         and collect per-epoch (object, epoch_jd, n_det) tuples. No expert
         calls yet.
      2. Batch-call `SNN.predict_epoch_batch` across every collected tuple —
         this loads the torch model once per chunk (vs once per call) and
         gives ~50-100× speedup.
      3. Walk the per-epoch tuples again, run the three non-batch local
         experts (`alerce_lc`, `lc_features_bv`, `salt3_chi2`) serially,
         assemble the snapshot row.

    Parameters
    ----------
    lc_dir : Path
        Directory of per-object DP1 DiaSource parquets
        (produced by scripts/fetch_dp1_lightcurves.py).
    truth_path : Path
        dp1_truth.parquet (produced by scripts/build_dp1_truth.py).
    out_path : Path
        Output parquet.
    max_n_det : int
        Cap on how many per-epoch snapshots per object (default 20).
    only_object_ids : list[int] or None
        Restrict to specific diaObjectIds (for smoke tests).
    limit : int or None
        Hard cap on the number of objects processed.
    snn_batch_size : int
        How many (object,epoch) tuples to feed SNN at once. 2000 is a safe
        default on the ELAsTiCC model (RNN inference fits in <2 GB).
    """
    import time

    truth = pd.read_parquet(truth_path)
    truth_by_id = {int(r["diaObjectId"]): r.to_dict() for _, r in truth.iterrows()}
    queue: list[int] = []
    for p in sorted(lc_dir.glob("*.parquet")):
        try:
            oid = int(p.stem)
        except ValueError:
            continue
        if only_object_ids and oid not in set(only_object_ids):
            continue
        queue.append(oid)
    if limit is not None:
        queue = queue[:limit]
    print(f"DP1 snapshot builder: {len(queue):,} objects in queue")

    print("Loading 4 local experts (torch, sncosmo, light-curve) ...")
    t0 = time.time()
    experts = _init_experts()
    for name, ex in experts.items():
        meta = ex.metadata()
        print(f"  [{name}] available={meta.get('available')} "
              f"model_loaded={meta.get('model_loaded', 'n/a')}")
    print(f"  expert load took {time.time()-t0:.1f}s")

    # ---- Pass 1: collect per-epoch tuples ----
    print("\nPass 1/3: extracting per-epoch features + building SNN batch queue")
    t1 = time.time()
    # Each entry: dict(oid, lightcurve, alert_jd, n_det, feats, truth_row)
    per_epoch_items: list[dict[str, Any]] = []
    skipped = 0
    for i, oid in enumerate(queue):
        if i % 500 == 0 and i > 0:
            print(f"  pass1: {i:,}/{len(queue):,} objects "
                  f"({len(per_epoch_items):,} epoch tuples)", flush=True)
        lc_path = lc_dir / f"{oid}.parquet"
        detections = _load_lightcurve(lc_path)
        if not detections:
            skipped += 1
            continue
        normalized = normalize_lightcurve(detections, survey="LSST")
        try:
            per_epoch = extract_features_at_each_epoch(normalized, max_n_det=max_n_det)
        except Exception as exc:
            print(f"  [{oid}] feature-extraction error: {type(exc).__name__}: {exc}")
            skipped += 1
            continue
        if not per_epoch:
            skipped += 1
            continue
        truth_row = truth_by_id.get(int(oid), {})
        for feats in per_epoch:
            n_det = int(feats.get("n_det", 0))
            alert_mjd = float(feats.pop("alert_mjd", 0.0))
            alert_jd = _mjd_to_jd(alert_mjd)
            per_epoch_items.append({
                "oid": int(oid),
                "normalized": normalized,
                "alert_jd": alert_jd,
                "n_det": n_det,
                "feats": feats,
                "truth_row": truth_row,
            })
    print(f"  pass1 done: {len(per_epoch_items):,} (obj, epoch) tuples, "
          f"{skipped:,} objects skipped, took {time.time()-t1:.1f}s")

    # ---- Pass 2: batch-SNN inference ----
    print(f"\nPass 2/3: batch SNN inference (chunk size {snn_batch_size:,})")
    t2 = time.time()
    snn_expert = experts.get("supernnova")
    snn_outputs: list[Any] = [None] * len(per_epoch_items)
    if snn_expert is not None and getattr(snn_expert, "_snn_available", False):
        for start in range(0, len(per_epoch_items), snn_batch_size):
            end = min(start + snn_batch_size, len(per_epoch_items))
            batch_items = [
                (str(it["oid"]), it["normalized"], it["alert_jd"], it["n_det"])
                for it in per_epoch_items[start:end]
            ]
            try:
                chunk_outs = snn_expert.predict_epoch_batch(batch_items)
            except Exception as exc:
                print(f"  SNN batch [{start}:{end}] FAILED: {type(exc).__name__}: {exc}")
                chunk_outs = [None] * (end - start)
            for j, out in enumerate(chunk_outs):
                snn_outputs[start + j] = out
            print(f"  snn: {end:,}/{len(per_epoch_items):,} "
                  f"({(time.time()-t2):.1f}s elapsed)", flush=True)
    else:
        print("  [supernnova] unavailable — all rows will have avail__supernnova=0")
    print(f"  pass2 done in {time.time()-t2:.1f}s")

    # ---- Pass 3: assemble rows with serial experts ----
    print(f"\nPass 3/3: serial inference on alerce_lc + lc_features_bv + salt3_chi2")
    t3 = time.time()
    rows: list[dict[str, Any]] = []
    for idx, it in enumerate(per_epoch_items):
        if idx % 1000 == 0 and idx > 0:
            print(f"  pass3: {idx:,}/{len(per_epoch_items):,} rows "
                  f"({(time.time()-t3):.1f}s elapsed, "
                  f"{(time.time()-t3)/max(idx,1)*1000:.1f} ms/row)", flush=True)
        oid = it["oid"]
        alert_jd = it["alert_jd"]
        n_det = it["n_det"]
        normalized = it["normalized"]
        feats = it["feats"]
        truth_row = it["truth_row"]

        final_class = truth_row.get("label")
        follow_proxy = (
            1.0 if final_class == "snia"
            else (0.0 if final_class in ("nonIa_snlike", "other") else None)
        )
        label_source = (
            "published_sn" if truth_row.get("is_published_sn")
            else ("gaia_star" if truth_row.get("is_gaia_known_star")
                  else ("simbad" if truth_row.get("simbad_main_type") else None))
        )

        base: dict[str, Any] = {
            "object_id": str(oid),
            "diaObjectId": int(oid),
            "n_det": n_det,
            "alert_jd": alert_jd,
            "survey": "LSST",
            "dp1_field": truth_row.get("dp1_field"),
            "lightcurve_source_object_id": str(oid),
            "lightcurve_source_identifier_kind": "lsst_dia_object_id",
            "lightcurve_association_kind": None,
            "lightcurve_association_source": None,
            "lightcurve_association_sep_arcsec": None,
        }
        for feature_name in FEATURE_NAMES:
            base[feature_name] = feats.get(feature_name)

        base["target_class"] = final_class
        base["target_follow_proxy"] = follow_proxy
        base["label_source"] = label_source
        base["label_quality"] = (
            "photometric_published" if truth_row.get("is_published_sn")
            else ("catalog" if truth_row.get("is_gaia_known_star") else None)
        )
        base["is_published_sn"] = bool(truth_row.get("is_published_sn", False))
        base["published_sn_name"] = truth_row.get("published_sn_name")
        base["published_sn_type"] = truth_row.get("published_sn_type")
        base["published_sn_conf"] = truth_row.get("published_sn_conf")
        base["is_gaia_known_star"] = bool(truth_row.get("is_gaia_known_star", False))
        base["simbad_main_type"] = truth_row.get("simbad_main_type")
        base["gaia_var_class"] = truth_row.get("gaia_var_class")
        base["is_known_variable"] = bool(truth_row.get("is_known_variable", False))

        # SNN — use batch result
        base.update(_expert_output_to_projection_cols(
            "supernnova", snn_outputs[idx], str(oid), alert_jd, n_det,
        ))

        # Three remaining local experts — serial
        base.update(_score_one_epoch_serial(
            experts, str(oid), normalized, alert_jd, n_det,
            skip_keys=("supernnova",),
        ))

        # Broker-only experts — mark unavailable
        for expert_key in ALL_EXPERT_KEYS:
            if expert_key in LOCAL_EXPERT_KEYS:
                continue
            base.update(_unavailable_expert_cols(expert_key))

        rows.append(base)
    print(f"  pass3 done in {time.time()-t3:.1f}s")

    if not rows:
        raise ValueError(f"No snapshot rows built from {lc_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    total = time.time() - t0
    print(f"\nWrote {len(rows):,} rows × {len(df.columns)} cols → {out_path}  "
          f"(total {total:.1f}s)")
    return out_path


__all__ = ["build_dp1_snapshots", "LOCAL_EXPERT_KEYS"]
