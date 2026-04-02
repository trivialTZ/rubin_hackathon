"""Gold-layer builders."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debass_meta.access.associations import load_lsst_ztf_associations, resolve_object_reference
from debass_meta.access.identifiers import infer_identifier_kind
from debass_meta.features.lightcurve import FEATURE_NAMES, extract_features_at_each_epoch
from debass_meta.projectors import ALL_EXPERT_KEYS, PHASE1_EXPERT_KEYS, get_expert_keys, project_expert_events, sanitize_expert_key

_SILVER_DIR = Path("data/silver")
_GOLD_DIR = Path("data/gold")
_LC_DIR = Path("data/lightcurves")
_TRUTH_PATH = Path("data/truth/object_truth.parquet")

# Legacy source list used by the baseline builder.
_SOURCES = ["alerce", "fink", "lasair", "supernnova", "parsnip", "ampel"]


def silver_to_gold(
    silver_dir: Path = _SILVER_DIR,
    gold_dir: Path = _GOLD_DIR,
    label_map: dict[str, str] | None = None,
    label_source: str = "manual",
) -> Path:
    """Legacy snapshot builder retained for baseline compatibility."""
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas required") from e

    silver_path = silver_dir / "broker_outputs.parquet"
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver file not found: {silver_path}")

    df = pd.read_parquet(silver_path)
    label_map = label_map or {}
    gold_rows: list[dict[str, Any]] = []

    for object_id, grp in df.groupby("object_id"):
        grp = grp.sort_values("query_time")
        grp["epoch_bucket"] = (grp["query_time"] // 60).astype(int)
        for epoch_idx, (_bucket, epoch_grp) in enumerate(
            grp.groupby("epoch_bucket", sort=True)
        ):
            availability_mask = {
                src: bool((epoch_grp["broker"] == src).any()) for src in _SOURCES
            }
            broker_scores: dict[str, Any] = {}
            for _, r in epoch_grp.iterrows():
                key = f"{r['broker']}__{r['field']}"
                broker_scores[key] = {
                    "raw_label_or_score": r["raw_label_or_score"],
                    "semantic_type": r["semantic_type"],
                    "canonical_projection": r["canonical_projection"],
                }

            gold_rows.append({
                "object_id": object_id,
                "epoch_index": epoch_idx,
                "epoch_jd": float(epoch_grp["query_time"].mean()),
                "survey": epoch_grp["survey"].iloc[0] if "survey" in epoch_grp else "ZTF",
                "availability_mask": availability_mask,
                "broker_scores": broker_scores,
                "target_label": label_map.get(str(object_id)),
                "label_source": label_source if str(object_id) in label_map else None,
            })

    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / "snapshots.parquet"
    pd.DataFrame(gold_rows).to_parquet(out_path, index=False)
    return out_path


def build_object_epoch_snapshots(
    lc_dir: Path = _LC_DIR,
    silver_dir: Path = _SILVER_DIR,
    gold_dir: Path = _GOLD_DIR,
    truth_path: Path = _TRUTH_PATH,
    max_n_det: int = 20,
    object_ids: list[str] | None = None,
    association_path: Path | None = None,
    allow_unsafe_latest_snapshot: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Build the canonical gold v2 object-epoch snapshot table."""
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas required") from e

    truth_lookup = _load_truth_lookup(truth_path)
    events_df = _load_all_event_rows(silver_dir)
    associations = load_lsst_ztf_associations(association_path)

    # Pre-group events by (object_id, expert_key) for O(1) lookup
    # instead of O(N) full-DataFrame scan per query
    _events_index: dict[tuple[str, str], Any] = {}
    if len(events_df) > 0:
        for key, group in events_df.groupby(["object_id", "expert_key"]):
            _events_index[key] = group
    print(f"  Indexed {len(_events_index):,} (object, expert) groups "
          f"from {len(events_df):,} event rows", flush=True)

    rows: list[dict[str, Any]] = []
    object_filter = set(object_ids or [])
    if object_filter:
        object_queue = sorted(object_filter)
    else:
        object_queue = sorted(path.stem for path in Path(lc_dir).glob("*.json"))
    n_total = len(object_queue)

    for i, object_id in enumerate(object_queue):
        if i % 1000 == 0:
            print(f"  snapshot: {i:,}/{n_total:,} objects ...", flush=True)
        lc_path, lightcurve_source = _resolve_lightcurve_path(
            Path(lc_dir),
            object_id=object_id,
            associations=associations,
        )
        if lc_path is None:
            continue
        detections = _load_lightcurve(lc_path)
        if not detections:
            continue
        epoch_features = extract_features_at_each_epoch(detections, max_n_det=max_n_det)

        truth = truth_lookup.get(object_id, {})

        # Infer survey from lightcurve content or object identifier
        id_kind = infer_identifier_kind(object_id)
        survey = "LSST" if id_kind == "lsst_dia_object_id" else "ZTF"

        for feats in epoch_features:
            n_det = int(feats["n_det"])
            alert_value = float(feats.pop("alert_mjd"))
            alert_jd = _to_jd(alert_value)
            base_row: dict[str, Any] = {
                "object_id": object_id,
                "n_det": n_det,
                "alert_jd": alert_jd,
                "survey": survey,
                "lightcurve_source_object_id": lightcurve_source["object_id"],
                "lightcurve_source_identifier_kind": lightcurve_source["identifier_kind"],
                "lightcurve_association_kind": lightcurve_source["association_kind"],
                "lightcurve_association_source": lightcurve_source["association_source"],
                "lightcurve_association_sep_arcsec": lightcurve_source["association_sep_arcsec"],
            }
            for feature_name in FEATURE_NAMES:
                base_row[feature_name] = feats.get(feature_name)

            base_row["target_class"] = truth.get("final_class_ternary")
            base_row["target_follow_proxy"] = truth.get("follow_proxy")
            base_row["label_source"] = truth.get("label_source")
            base_row["label_quality"] = truth.get("label_quality")

            # Attach all registered experts (maintains consistent parquet schema).
            # For ZTF-only experts on LSST objects, avail=0 naturally.
            for expert_key in ALL_EXPERT_KEYS:
                _attach_expert_projection(
                    base_row,
                    _events_index,
                    object_id=object_id,
                    expert_key=expert_key,
                    alert_jd=alert_jd,
                    allow_unsafe_latest_snapshot=allow_unsafe_latest_snapshot,
                )

            rows.append(base_row)

    print(f"  snapshot: {n_total:,}/{n_total:,} objects done "
          f"({len(rows):,} epoch rows)", flush=True)

    if not rows:
        raise ValueError(f"No epoch rows built from {lc_dir}")

    if output_path is None:
        gold_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(gold_dir) / "object_epoch_snapshots.parquet"
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path


def select_events_asof(
    event_rows,
    *,
    alert_jd: float,
    allow_unsafe_latest_snapshot: bool = False,
) -> list[dict[str, Any]]:
    """Select the latest event-safe rows for one expert/object pair.

    Priority order:
      1. Timed events with event_time_jd <= alert_jd  (exact_alert / rerun_exact with JD)
      2. static_safe events                             (e.g. Lasair Sherlock context)
      3. rerun_exact events (without JD)                (local experts)
      4. latest_object_unsafe events                    (ALeRCE API snapshots, opt-in)

    The exact__{expert} column marks (4) as 0.0 so the trust model can
    learn appropriate weighting for temporally unsafe scores.
    """
    if len(event_rows) == 0:
        return []
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas required") from e

    if not hasattr(event_rows, "copy"):
        event_rows = pd.DataFrame(event_rows)

    if "availability" in event_rows.columns:
        available_mask = event_rows["availability"].fillna(True).astype(bool)
        event_rows = event_rows[available_mask].copy()
        if len(event_rows) == 0:
            return []

    # 1. Timed events: latest event at or before alert_jd
    timed = event_rows[event_rows["event_time_jd"].notna()].copy()
    if len(timed) > 0:
        timed = timed[timed["event_time_jd"] <= alert_jd]
        if len(timed) > 0:
            max_time = timed["event_time_jd"].max()
            return timed[timed["event_time_jd"] == max_time].to_dict(orient="records")

    # 2. Static-safe context labels (e.g. Lasair Sherlock)
    static_safe = event_rows[event_rows["temporal_exactness"] == "static_safe"]
    if len(static_safe) > 0:
        return static_safe.to_dict(orient="records")

    # 3. Local expert re-runs at exact epoch (without timed JD)
    rerun_exact = event_rows[event_rows["temporal_exactness"] == "rerun_exact"]
    if len(rerun_exact) > 0:
        return rerun_exact.to_dict(orient="records")

    # 4. Latest object snapshot (ALeRCE API — temporally unsafe and
    #    excluded by default unless explicitly requested for scoring/debugging)
    if allow_unsafe_latest_snapshot:
        unsafe = event_rows[event_rows["temporal_exactness"] == "latest_object_unsafe"]
        if len(unsafe) > 0:
            return unsafe.to_dict(orient="records")

    return []


def _attach_expert_projection(
    row: dict[str, Any],
    events_index: dict,
    *,
    object_id: str,
    expert_key: str,
    alert_jd: float,
    allow_unsafe_latest_snapshot: bool,
) -> None:
    san = sanitize_expert_key(expert_key)
    subset = events_index.get((object_id, expert_key))
    if subset is None:
        row[f"avail__{san}"] = 0.0
        row[f"exact__{san}"] = 0.0
        row[f"temporal_exactness__{san}"] = None
        row[f"source_event_time_jd__{san}"] = None
        row[f"reason__{san}"] = "expert unavailable at this epoch"
        return
    selected = select_events_asof(
        subset,
        alert_jd=alert_jd,
        allow_unsafe_latest_snapshot=allow_unsafe_latest_snapshot,
    )
    row[f"avail__{san}"] = float(bool(selected))
    if not selected:
        row[f"exact__{san}"] = 0.0
        row[f"temporal_exactness__{san}"] = None
        row[f"source_event_time_jd__{san}"] = None
        row[f"reason__{san}"] = "expert unavailable at this epoch"
        return

    temporal_exactness = selected[0].get("temporal_exactness")
    row[f"exact__{san}"] = float(temporal_exactness != "latest_object_unsafe")
    row[f"temporal_exactness__{san}"] = temporal_exactness
    row[f"source_event_time_jd__{san}"] = selected[0].get("event_time_jd")
    row[f"event_count__{san}"] = float(len(selected))

    projected = project_expert_events(expert_key, selected)
    for key, value in projected.items():
        if key in {"prediction_type", "mapped_pred_class", "context_tag", "reason"}:
            row[f"{key}__{san}"] = value
        else:
            row[f"proj__{san}__{key}"] = value


def _load_truth_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas required") from e
    df = pd.read_parquet(path)
    if len(df) == 0:
        return {}
    df = df.set_index(df["object_id"].astype(str))
    return df.to_dict(orient="index")


def _load_all_event_rows(silver_dir: Path):
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas required") from e

    broker_events_path = Path(silver_dir) / "broker_events.parquet"
    broker_df = pd.read_parquet(broker_events_path) if broker_events_path.exists() else pd.DataFrame()
    local_df = _load_local_expert_event_rows(Path(silver_dir))
    if len(broker_df) == 0:
        return local_df
    if len(local_df) == 0:
        return broker_df
    return pd.concat([broker_df, local_df], ignore_index=True, sort=False)


def _load_local_expert_event_rows(silver_dir: Path):
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas required") from e

    records: list[dict[str, Any]] = []
    parquet_files = sorted((silver_dir / "local_expert_outputs").glob("*/*.parquet"))
    if parquet_files:
        for path in parquet_files:
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                records.extend(_local_record_to_events(row.to_dict()))
    else:
        legacy_json = silver_dir / "local_expert_outputs.json"
        if legacy_json.exists():
            with open(legacy_json) as fh:
                data = json.load(fh)
            for record in data:
                records.extend(_local_record_to_events(record))
    return pd.DataFrame(records)


def _local_record_to_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    expert = str(record.get("expert") or record.get("model_name") or "")
    class_probabilities = record.get("class_probabilities") or {}
    if isinstance(class_probabilities, str):
        try:
            class_probabilities = json.loads(class_probabilities)
        except json.JSONDecodeError:
            class_probabilities = {}

    alert_jd = record.get("alert_jd")
    if alert_jd is None and record.get("alert_mjd") is not None:
        alert_jd = _to_jd(float(record["alert_mjd"]))

    out: list[dict[str, Any]] = []
    for class_name, probability in class_probabilities.items():
        out.append({
            "object_id": record.get("object_id"),
            "broker": "local",
            "classifier": expert,
            "classifier_version": record.get("model_version"),
            "expert_key": expert,
            "field": "class_probability",
            "class_name": class_name,
            "semantic_type": "probability",
            "event_scope": "rerun_local",
            "event_time_jd": float(alert_jd) if alert_jd is not None else None,
            "alert_id": None,
            "n_det": int(record["n_det"]) if record.get("n_det") is not None else None,
            "raw_label_or_score": probability,
            "canonical_projection": float(probability) if probability is not None else None,
            "ranking": None,
            "availability": bool(record.get("available", True)),
            "fixture_used": False,
            "temporal_exactness": "rerun_exact",
            "payload_hash": None,
            "provenance_json": json.dumps({"source": "local_expert_outputs"}),
            "survey": "ZTF",
            "source_endpoint": "local_infer",
            "request_params_json": None,
            "status_code": None,
            "query_time_unix": None,
        })
    return out


def _load_lightcurve(path: Path) -> list[dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return []


def _resolve_lightcurve_path(
    lc_dir: Path,
    *,
    object_id: str,
    associations: dict[str, dict[str, object]] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    reference = resolve_object_reference(
        object_id,
        broker="alerce_history",
        associations=associations,
    )
    if reference.requested_object_id is not None and reference.requested_object_id != object_id:
        requested_path = lc_dir / f"{reference.requested_object_id}.json"
        if requested_path.exists():
            return requested_path, {
                "object_id": reference.requested_object_id,
                "identifier_kind": reference.requested_identifier_kind,
                "association_kind": reference.association_kind,
                "association_source": reference.association_source,
                "association_sep_arcsec": reference.association_sep_arcsec,
            }

    direct_path = lc_dir / f"{object_id}.json"
    if direct_path.exists():
        return direct_path, {
            "object_id": object_id,
            "identifier_kind": infer_identifier_kind(object_id),
            "association_kind": None,
            "association_source": None,
            "association_sep_arcsec": None,
        }

    if not reference.can_query:
        return None, {
            "object_id": object_id,
            "identifier_kind": infer_identifier_kind(object_id),
            "association_kind": None,
            "association_source": None,
            "association_sep_arcsec": None,
        }

    if reference.requested_object_id is None:
        return None, {
            "object_id": object_id,
            "identifier_kind": infer_identifier_kind(object_id),
            "association_kind": reference.association_kind,
            "association_source": reference.association_source,
            "association_sep_arcsec": reference.association_sep_arcsec,
        }

    return None, {
        "object_id": reference.requested_object_id,
        "identifier_kind": reference.requested_identifier_kind,
        "association_kind": reference.association_kind,
        "association_source": reference.association_source,
        "association_sep_arcsec": reference.association_sep_arcsec,
    }


def _to_jd(value: float) -> float:
    return value + 2400000.5 if value < 2400000.5 else value
