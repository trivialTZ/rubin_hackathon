"""Lasair broker adapter.

Sherlock output is a crossmatch-based context label (crossmatch_tag),
NOT a calibrated posterior. Treat as a categorical feature only.
Falls back to saved fixture when LASAIR_TOKEN is absent.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import BrokerAdapter, BrokerOutput, SemanticType

_FIXTURE_DIR = Path("fixtures/raw/lasair")

# Valid Sherlock context labels
_SHERLOCK_CLASSES = ["SN", "NT", "AGN", "CV", "BS", "ORPHAN", "VS", "UNCLEAR"]


class LasairAdapter(BrokerAdapter):
    name = "lasair"
    phase = 1
    semantic_type: SemanticType = "crossmatch_tag"

    def __init__(self) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        self._token = os.environ.get("LASAIR_TOKEN", "")
        self._client = None
        if self._token:
            try:
                import lasair
                self._client = lasair.LasairError  # import check only
                self._client = lasair.lasair_client(self._token)
            except ImportError:
                self._client = None
            except Exception:
                self._client = None

    def _has_live_access(self) -> bool:
        return self._client is not None and bool(self._token)

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        if not self._token:
            return {
                "broker": self.name,
                "status": "fixture_fallback",
                "reason": "LASAIR_TOKEN not set — live access unavailable",
            }
        if self._client is None:
            return {"broker": self.name, "status": "unavailable", "reason": "lasair package not installed or auth failed"}
        try:
            result = self._client.cone(ra=0.0, dec=0.0, radius=1.0, requestType="nearest")
            return {"broker": self.name, "status": "ok", "cone_result": result}
        except Exception as exc:
            return {"broker": self.name, "status": "error", "reason": str(exc)}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False

        if self._has_live_access():
            try:
                obj = self._client.objects([object_id])
                raw = obj[0] if obj else {}
                self.save_fixture(raw, fixture_path)
            except Exception:
                raw, fixture_used = self._load_fixture_or_empty(fixture_path)
        else:
            raw, fixture_used = self._load_fixture_or_empty(fixture_path)

        fields = self._extract_fields(raw)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey="ZTF",
            source_endpoint="lasair.objects",
            request_params={"object_id": object_id},
            fields=fields,
            events=fields,
            availability=self._has_live_access(),
            fixture_used=fixture_used,
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        fixture_path = _FIXTURE_DIR / f"{object_id}_lc.json"
        if self._has_live_access():
            try:
                lc = self._client.lightcurves([object_id])
                data = lc[0] if lc else {}
                self.save_fixture(data, fixture_path)
                return data
            except Exception:
                pass
        if fixture_path.exists():
            return self.load_fixture(fixture_path)
        return {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_fixture_or_empty(self, path: Path) -> tuple[dict, bool]:
        if path.exists():
            return self.load_fixture(path), True
        return {}, True

    def _extract_fields(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        sherlock = raw.get("sherlock_classification") or raw.get("classification")
        if sherlock is None:
            return []
        return [{
            "field": "sherlock_classification",
            "raw_label_or_score": sherlock,
            "semantic_type": "crossmatch_tag",
            "canonical_projection": None,  # never interpret as probability
            "classifier": "sherlock",
            "class_name": str(sherlock),
            "expert_key": "lasair/sherlock",
            "event_scope": "static_context",
            "temporal_exactness": "static_safe",
            "event_time_jd": None,
            "note": "context label only — not a calibrated posterior",
        }]
