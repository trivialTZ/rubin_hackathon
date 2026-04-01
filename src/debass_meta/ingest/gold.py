"""Gold-layer builders."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from debass_meta.features.lightcurve import FEATURE_NAMES, extract_features_at_each_epoch
from debass_meta.projectors import PHASE1_EXPERT_KEYS, project_expert_events, sanitize_expert_key

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

    rows: list[dict[str, Any]] = []
    object_filter = set(object_ids or [])

    for lc_path in sorted(Path(lc_dir).glob("*.json")):
        object_id = lc_path.stem
        if object_filter and object_id not in object_filter:
            continue
        detections = _load_lightcurve(lc_path)
        if not detections:
            continue
        epoch_features = extract_features_at_each_epoch(detections, max_n_det=max_n_det)
        for feats in epoch_features:
            n_det = int(feats["n_det"])
            alert_value = float(feats.pop("alert_mjd"))
            alert_jd = _to_jd(alert_value)
            base_row: dict[str, Any] = {
                "object_id": object_id,
                "n_det": n_det,
                "alert_jd": alert_jd,
            }
            for feature_name in FEATURE_NAMES:
                base_row[feature_name] = feats.get(feature_name)

            truth = truth_lookup.get(object_id, {})
            base_row["target_class"] = truth.get("final_class_ternary")
            base_row["target_follow_proxy"] = truth.get("follow_proxy")
            base_row["label_source"] = truth.get("label_source")
            base_row["label_quality"] = truth.get("label_quality")

            for expert_key in PHASE1_EXPERT_KEYS:
                _attach_expert_projection(
                    base_row,
                    events_df,
                    object_id=object_id,
                    expert_key=expert_key,
                    alert_jd=alert_jd,
                    allow_unsafe_latest_snapshot=allow_unsafe_latest_snapshot,
                )

            rows.append(base_row)

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
    """Select the latest event-safe rows for one expert/object pair."""
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

    timed = event_rows[event_rows["event_time_jd"].notna()].copy()
    if len(timed) > 0:
        timed = timed[timed["event_time_jd"] <= alert_jd]
        if len(timed) > 0:
            max_time = timed["event_time_jd"].max()
            return timed[timed["event_time_jd"] == max_time].to_dict(orient="records")

    static_safe = event_rows[event_rows["temporal_exactness"] == "static_safe"]
    if len(static_safe) > 0:
        return static_safe.to_dict(orient="records")

    rerun_exact = event_rows[event_rows["temporal_exactness"] == "rerun_exact"]
    if len(rerun_exact) > 0:
        return rerun_exact.to_dict(orient="records")

    if allow_unsafe_latest_snapshot:
        unsafe = event_rows[event_rows["temporal_exactness"] == "latest_object_unsafe"]
        if len(unsafe) > 0:
            return unsafe.to_dict(orient="records")

    return []


def _attach_expert_projection(
    row: dict[str, Any],
    events_df,
    *,
    object_id: str,
    expert_key: str,
    alert_jd: float,
    allow_unsafe_latest_snapshot: bool,
) -> None:
    san = sanitize_expert_key(expert_key)
    subset = events_df[
        (events_df["object_id"] == object_id) &
        (events_df["expert_key"] == expert_key)
    ]
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
    return {
        str(row["object_id"]): row.to_dict()
        for _, row in df.iterrows()
    }


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


def _to_jd(value: float) -> float:
    return value + 2400000.5 if value < 2400000.5 else value
