"""ANTARES broker adapter — Phase 1 integration.

ANTARES is NOIRLab's alert-filter hosting broker for ZTF (now) and LSST.
We treat it as a container for:
  • `antares/oracle`        — Shah+ 2025 hierarchical 19-class GRU classifier
  • `antares/superphot_plus` — de Soto+ 2024 5-class SN classifier

Both emit per-alert tags on a locus; we retrieve the locus history and map
each tag to a DEBASS-ternary projection.  When `antares-client` is not
installed or credentials are missing, the adapter gracefully returns an
empty BrokerOutput with availability=False.

Docs: https://nsf-noirlab.gitlab.io/csdc/antares/client/
Web:  https://antares.noirlab.edu/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import BrokerAdapter, BrokerOutput, SemanticType
from .identifiers import infer_identifier_kind

_FIXTURE_DIR = Path("fixtures/raw/antares")

# Subset of tags DEBASS cares about.  Exact tag names are stable per ANTARES
# filter registration; see RTN-090 for operational filters.
_DEBASS_TAGS: dict[str, str] = {
    # Superphot+ class tags (one per alert with score)
    "superphot_plus_class": "antares/superphot_plus",
    # ORACLE 19-class: one top-level tag for the argmax leaf
    "oracle_class": "antares/oracle",
    # Legacy ZTF tags that can inform truth tiers
    "nuclear_transient": "antares/context",
    "refitt_newsources_snrcut": "antares/context",
}


class AntaresAdapter(BrokerAdapter):
    name = "antares"
    phase = 1
    semantic_type: SemanticType = "mixed"

    def __init__(self, timeout: int = 20) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        try:
            import antares_client  # noqa: F401

            return {"broker": self.name, "status": "ok", "client": "antares-client"}
        except Exception as exc:
            return {"broker": self.name, "status": "client_missing", "reason": str(exc)}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        """Fetch ANTARES locus by ZTF object ID (or LSST diaObjectId if supported).

        Strategy: retrieve the full locus with alerts and tags, then emit one
        event per (tag, alert) tuple so that the silver layer can index by n_det.
        """
        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False
        availability = False
        reason: str | None = None

        try:
            raw = self._query_locus(object_id)
            if raw:
                self.save_fixture(raw, fixture_path)
                availability = True
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if fixture_path.exists():
                raw = self.load_fixture(fixture_path)
                fixture_used = True

        events = self._extract_events(raw)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey=self._infer_survey(object_id),
            fields=events,
            events=events,
            availability=availability,
            fixture_used=fixture_used,
            request_params={"identifier": object_id, "reason": reason},
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        """ANTARES lightcurves come from the underlying survey — return empty."""
        return {}

    # ------------------------------------------------------------------ #
    # Locus retrieval (lazy antares-client)                                #
    # ------------------------------------------------------------------ #

    def _query_locus(self, object_id: str) -> dict[str, Any]:
        try:
            from antares_client.search import (  # type: ignore[import-not-found]
                get_by_id,
                get_by_ztf_object_id,
            )
        except ImportError:
            raise RuntimeError(
                "antares-client not installed. Add to env/requirements.txt."
            )

        kind = infer_identifier_kind(object_id)
        locus = None
        if kind == "ztf_object_id":
            locus = get_by_ztf_object_id(object_id)
        elif kind == "lsst_dia_object_id":
            try:
                locus = get_by_id(str(object_id))
            except Exception:
                return {}
        else:
            return {}
        if locus is None:
            return {}
        return _locus_to_dict(locus)

    # ------------------------------------------------------------------ #
    # Event extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_events(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        tags = raw.get("tags") or []
        alerts = raw.get("alerts") or []
        if not isinstance(alerts, list) or not alerts:
            return events

        def _alert_time(alert: dict[str, Any]) -> float:
            return float(alert.get("mjd") or alert.get("jd") or 0.0)

        sorted_alerts = sorted(
            [a for a in alerts if isinstance(a, dict)], key=_alert_time
        )

        # Group tags by alert_id for fast lookup when scores are alert-local.
        alert_tags: dict[Any, list[dict[str, Any]]] = {}
        for tag in tags or []:
            if not isinstance(tag, dict):
                continue
            alert_id = tag.get("alert_id")
            alert_tags.setdefault(alert_id, []).append(tag)

        for idx, alert in enumerate(sorted_alerts, start=1):
            mjd = alert.get("mjd") or alert.get("jd")
            jd = (float(mjd) + 2400000.5) if mjd is not None and float(mjd) < 2400000.5 else mjd
            alert_id = alert.get("alert_id") or alert.get("id") or alert.get("candid")
            # Include tags attached to this alert (or object-wide with alert_id=None)
            this_alert_tags = alert_tags.get(alert_id, []) + alert_tags.get(None, [])
            for tag in this_alert_tags:
                tag_name = tag.get("name") or tag.get("tag") or tag.get("locus_tag")
                if tag_name is None:
                    continue
                # Map known tags to DEBASS experts
                expert_key = _DEBASS_TAGS.get(tag_name) or _tag_family(tag_name)
                if expert_key is None:
                    continue
                score = tag.get("score") or tag.get("probability") or tag.get("confidence")
                class_label = tag.get("class") or tag.get("class_label") or tag.get("label")
                events.append({
                    "field": tag_name,
                    "raw_label_or_score": class_label if class_label is not None else score,
                    "semantic_type": "probability" if score is not None else "heuristic_class",
                    "canonical_projection": float(score) if score is not None else None,
                    "classifier": tag_name.split("_")[0] if "_" in tag_name else tag_name,
                    "classifier_version": tag.get("version"),
                    "class_name": str(class_label) if class_label is not None else None,
                    "expert_key": expert_key,
                    "event_scope": "alert",
                    "temporal_exactness": "exact_alert",
                    "alert_id": alert_id,
                    "n_det": idx,
                    "alert_jd": float(jd) if jd is not None else None,
                    "event_time_jd": float(jd) if jd is not None else None,
                    "raw_tag_properties": json.dumps(tag, default=str),
                })
        return events

    @staticmethod
    def _infer_survey(object_id: str) -> str:
        return "LSST" if infer_identifier_kind(object_id) == "lsst_dia_object_id" else "ZTF"


def _tag_family(tag_name: str) -> str | None:
    """Infer expert_key from a tag name pattern like `superphot_plus_SNIa`."""
    tag_lower = tag_name.lower()
    if tag_lower.startswith("superphot_plus") or tag_lower.startswith("superphotplus"):
        return "antares/superphot_plus"
    if tag_lower.startswith("oracle"):
        return "antares/oracle"
    return None


def _locus_to_dict(locus) -> dict[str, Any]:
    """Flatten an antares_client Locus into a JSON-safe dict.

    Handles both attribute-based locus objects and dict-like payloads.
    """
    def _get(obj, key, default=None):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    alerts_raw = _get(locus, "alerts", []) or []
    tags_raw = _get(locus, "tags", []) or []
    properties = _get(locus, "properties", {}) or {}
    lightcurve = _get(locus, "lightcurve", None)

    alerts_out: list[dict[str, Any]] = []
    for alert in alerts_raw:
        alert_dict = alert if isinstance(alert, dict) else {
            "alert_id": _get(alert, "alert_id"),
            "mjd": _get(alert, "mjd"),
            "properties": _get(alert, "properties", {}) or {},
        }
        alerts_out.append(alert_dict)

    tags_out: list[dict[str, Any]] = []
    for tag in tags_raw:
        if isinstance(tag, str):
            tags_out.append({"name": tag})
            continue
        tags_out.append({
            "name": _get(tag, "name") or _get(tag, "tag"),
            "score": _get(tag, "score") or _get(tag, "probability"),
            "class": _get(tag, "class") or _get(tag, "class_label"),
            "alert_id": _get(tag, "alert_id"),
            "version": _get(tag, "version"),
        })

    return {
        "locus_id": _get(locus, "locus_id") or _get(locus, "id"),
        "alerts": alerts_out,
        "tags": tags_out,
        "properties": properties,
        "lightcurve": lightcurve if isinstance(lightcurve, (dict, list)) else None,
    }
