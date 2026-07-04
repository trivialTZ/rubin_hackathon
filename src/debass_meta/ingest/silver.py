"""Silver layer: normalize bronze records into event-level broker rows.

Primary output:
  data/silver/broker_events.parquet

Compatibility output:
  data/silver/broker_outputs.parquet
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_BRONZE_DIR = Path("data/bronze")
_SILVER_DIR = Path("data/silver")

EVENT_COLUMNS = [
    "object_id",
    "primary_object_id",
    "requested_object_id",
    "primary_identifier_kind",
    "requested_identifier_kind",
    "associated_object_id",
    "association_kind",
    "association_source",
    "association_sep_arcsec",
    "broker",
    "classifier",
    "classifier_version",
    "expert_key",
    "field",
    "class_name",
    "semantic_type",
    "event_scope",
    "event_time_jd",
    "alert_id",
    "n_det",
    "raw_label_or_score",
    "canonical_projection",
    "ranking",
    "availability",
    "fixture_used",
    "temporal_exactness",
    "payload_hash",
    "provenance_json",
    "survey",
    "source_endpoint",
    "request_params_json",
    "status_code",
    "query_time_unix",
]

LEGACY_COLUMNS = [
    "broker",
    "object_id",
    "query_time",
    "survey",
    "field",
    "raw_label_or_score",
    "semantic_type",
    "canonical_projection",
    "classifier",
    "class",
    "fixture_used",
    "payload_hash",
]


def _serialize_raw_label_or_score(value: Any) -> str | None:
    """Return a parquet-safe string representation preserving the original value.

    PyArrow cannot write a column that mixes floats and strings. Keep the
    verbatim payload in a single nullable string column by JSON-encoding
    non-string values.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def bronze_to_silver(
    bronze_dir: Path = _BRONZE_DIR,
    silver_dir: Path = _SILVER_DIR,
) -> Path:
    """Expand all bronze parquet files into silver broker-events tables.

    Returns the path of the written broker_events file.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas is required for silver transforms") from e

    bronze_files = sorted(bronze_dir.glob("*.parquet"))
    if not bronze_files:
        raise FileNotFoundError(f"No bronze parquet files found in {bronze_dir}")

    event_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []

    for bf in bronze_files:
        df = pd.read_parquet(bf)
        for _, row in df.iterrows():
            requested_object_id = row["requested_object_id"] if "requested_object_id" in row.index else row["object_id"]
            primary_object_id = row["primary_object_id"] if "primary_object_id" in row.index else row["object_id"]
            raw_payload = json.loads(row["raw_payload"]) if isinstance(row["raw_payload"], str) else row["raw_payload"]
            events = raw_payload.get("_events") or raw_payload.get("_fields", [])
            if isinstance(row.get("events_json"), str) and not events:
                try:
                    events = json.loads(row["events_json"])
                except json.JSONDecodeError:
                    events = []

            if not events:
                # Bronze already preserves the raw broker response. Silver is the
                # canonical event-level table, so skip broker payloads that do
                # not decode into any usable event rows.
                continue

            for ranking, event in enumerate(events, start=1):
                event_time_jd = event.get("event_time_jd")
                if event_time_jd is None:
                    event_time_jd = event.get("alert_jd")

                event_row = {
                    "object_id": row["object_id"],
                    "primary_object_id": primary_object_id,
                    "requested_object_id": requested_object_id,
                    "primary_identifier_kind": row.get("primary_identifier_kind"),
                    "requested_identifier_kind": row.get("requested_identifier_kind"),
                    "associated_object_id": row.get("associated_object_id"),
                    "association_kind": row.get("association_kind"),
                    "association_source": row.get("association_source"),
                    "association_sep_arcsec": row.get("association_sep_arcsec"),
                    "broker": row["broker"],
                    "classifier": event.get("classifier"),
                    "classifier_version": event.get("classifier_version"),
                    "expert_key": _canonical_expert_key(
                        row["broker"],
                        event.get("expert_key") or _infer_expert_key(row["broker"], event),
                    ),
                    "field": event.get("field", "_raw"),
                    "class_name": event.get("class_name") or event.get("class"),
                    "semantic_type": event.get("semantic_type", row["semantic_type"]),
                    "event_scope": event.get("event_scope") or _default_event_scope(row["broker"]),
                    "event_time_jd": float(event_time_jd) if event_time_jd is not None else None,
                    "alert_id": event.get("alert_id"),
                    "n_det": int(event["n_det"]) if event.get("n_det") is not None else None,
                    "raw_label_or_score": _serialize_raw_label_or_score(
                        event.get("raw_label_or_score")
                    ),
                    "canonical_projection": event.get("canonical_projection"),
                    "ranking": event.get("ranking", ranking),
                    "availability": bool(row.get("availability", True)),
                    "fixture_used": bool(row["fixture_used"]),
                    "temporal_exactness": event.get("temporal_exactness") or _default_exactness(row["broker"]),
                    "payload_hash": row.get("payload_hash"),
                    "provenance_json": json.dumps(
                        {
                            "broker": row["broker"],
                            "source_endpoint": row.get("source_endpoint"),
                            "request_params_json": row.get("request_params_json"),
                            "primary_object_id": primary_object_id,
                            "requested_object_id": requested_object_id,
                            "primary_identifier_kind": row.get("primary_identifier_kind"),
                            "requested_identifier_kind": row.get("requested_identifier_kind"),
                            "associated_object_id": row.get("associated_object_id"),
                            "association_kind": row.get("association_kind"),
                            "association_source": row.get("association_source"),
                            "association_sep_arcsec": row.get("association_sep_arcsec"),
                        },
                        sort_keys=True,
                    ),
                    "survey": row.get("survey", "ZTF"),
                    "source_endpoint": row.get("source_endpoint"),
                    "request_params_json": row.get("request_params_json"),
                    "status_code": row.get("status_code"),
                    "query_time_unix": row.get("query_time"),
                }
                event_rows.append(event_row)
                legacy_rows.append({
                    "broker": event_row["broker"],
                    "object_id": event_row["object_id"],
                    "query_time": event_row["query_time_unix"],
                    "survey": event_row["survey"],
                    "field": event_row["field"],
                    "raw_label_or_score": event_row["raw_label_or_score"],
                    "semantic_type": event_row["semantic_type"],
                    "canonical_projection": event_row["canonical_projection"],
                    "classifier": event_row["classifier"],
                    "class": event_row["class_name"],
                    "fixture_used": event_row["fixture_used"],
                    "payload_hash": event_row["payload_hash"],
                })

    silver_dir.mkdir(parents=True, exist_ok=True)
    out_path = silver_dir / "broker_events.parquet"
    legacy_path = silver_dir / "broker_outputs.parquet"

    silver_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    if "raw_label_or_score" in silver_df.columns:
        silver_df["raw_label_or_score"] = silver_df["raw_label_or_score"].astype("string")

    # Dedup: running backfill + normalize twice should not create duplicate silver rows.
    # Key: (object_id, expert_key, event_time_jd, field, raw_label_or_score).
    # Exclude payload_hash — re-fetching the same score from a different backfill
    # run produces a different hash but is logically the same event.
    #
    # pandas drop_duplicates treats NaN != NaN, so context experts with
    # event_time_jd=NaN are never deduped.  Fill NaN with a sentinel before
    # dedup, then restore.
    if len(silver_df) > 0:
        _SENTINEL = -999.0
        silver_df["_dedup_jd"] = silver_df["event_time_jd"].fillna(_SENTINEL)
        _dedup_on = ["object_id", "expert_key", "_dedup_jd", "field", "raw_label_or_score"]
        n_before = len(silver_df)
        silver_df = silver_df.drop_duplicates(subset=_dedup_on, keep="last")
        silver_df = silver_df.drop(columns=["_dedup_jd"])
        n_dropped = n_before - len(silver_df)
        if n_dropped > 0:
            print(f"  silver dedup: {n_before:,} → {len(silver_df):,} ({n_dropped:,} duplicates removed)")

    # Ensure classifier_version is uniform string type — broker APIs return
    # a mix of str ("2.1.0"), int, and None which pyarrow rejects as mixed.
    if "classifier_version" in silver_df.columns:
        silver_df["classifier_version"] = silver_df["classifier_version"].astype("string")
    silver_df.to_parquet(out_path, index=False)

    legacy_df = pd.DataFrame(legacy_rows, columns=LEGACY_COLUMNS)
    if len(legacy_df) > 0:
        legacy_df["raw_label_or_score"] = legacy_df["raw_label_or_score"].astype("string")
    legacy_df.to_parquet(legacy_path, index=False)
    return out_path


def _canonical_expert_key(broker: str, expert_key: str | None) -> str | None:
    """Canonicalize drift-prone expert keys at normalization time.

    Bronze payloads fetched under pre-canonicalization code carry dated
    ALeRCE Rubin-stamp keys (``alerce/stamp_classifier_rubin_beta_YYYYMMDD``)
    verbatim; the gold builder looks events up by the exact registry key, so
    without this rewrite those events are silently dropped from every
    rebuild.  Fetch-time canonicalization (access/alerce.py) only helps NEW
    bronze — this seam fixes history on re-normalization."""
    if not expert_key or broker != "alerce":
        return expert_key
    if not str(expert_key).startswith("alerce/"):
        return expert_key
    from debass_meta.access.alerce import canonical_alerce_expert_key

    return canonical_alerce_expert_key(str(expert_key).split("/", 1)[1])


def _infer_expert_key(broker: str, event: dict[str, Any]) -> str:
    classifier = event.get("classifier") or ""
    field = event.get("field") or ""
    if broker == "fink":
        if str(field).startswith("snn_"):
            return "fink/snn"
        if field == "rf_snia_vs_nonia":
            return "fink/rf_ia"
        return "fink/aux"
    if broker == "alerce":
        if not classifier:
            return "alerce/unknown"
        from debass_meta.access.alerce import canonical_alerce_expert_key

        return canonical_alerce_expert_key(str(classifier))
    if broker == "lasair":
        return "lasair/sherlock"
    return f"{broker}/{classifier}" if classifier else broker


def _default_event_scope(broker: str) -> str:
    return {
        "fink": "alert",
        "alerce": "object_snapshot",
        "lasair": "static_context",
        "babamul": "static_context",
    }.get(broker, "object_snapshot")


def _default_exactness(broker: str) -> str:
    return {
        "fink": "exact_alert",
        "alerce": "latest_object_unsafe",
        "lasair": "static_safe",
        "babamul": "static_safe",
    }.get(broker, "latest_object_unsafe")
