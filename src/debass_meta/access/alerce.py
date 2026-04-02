"""ALeRCE broker adapter.

Uses the public alerce Python client. No credentials required.
Returns explicit multiclass posteriors from both stamp and LC classifiers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BrokerAdapter, BrokerOutput, SemanticType
from .identifiers import infer_identifier_kind

_FIXTURE_DIR = Path("fixtures/raw/alerce")

# Canonical probability fields from ALeRCE LC classifier
_LC_CLASSES = ["SNIa", "SNIbc", "SNII", "SLSN", "AGN", "VS", "asteroid", "bogus"]
_STAMP_CLASSES = ["AGN", "SN", "VS", "asteroid", "bogus"]


class AlerceAdapter(BrokerAdapter):
    name = "alerce"
    phase = 1
    semantic_type: SemanticType = "probability"

    def __init__(self) -> None:
        try:
            from alerce.core import Alerce
            self._client = Alerce()
        except ImportError:
            self._client = None

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        if self._client is None:
            return {"broker": self.name, "status": "unavailable", "reason": "alerce not installed"}
        try:
            # Query a known ZTF object to verify connectivity
            result = self._client.query_objects(format="pandas", page=1, page_size=1)
            ok = result is not None and len(result) > 0
            return {"broker": self.name, "status": "ok" if ok else "empty", "rows": len(result) if result is not None else 0}
        except Exception as exc:
            return {"broker": self.name, "status": "error", "reason": str(exc)}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        identifier_kind = infer_identifier_kind(object_id)
        if identifier_kind == "lsst_dia_object_id":
            return self.unsupported_identifier_output(
                object_id,
                source_endpoint="alerce.query_probabilities",
                request_params={"oid": object_id},
                survey="LSST",
                identifier_kind=identifier_kind,
                expected_identifier_kind="ztf_object_id",
                reason="unsupported_identifier_for_broker",
            )

        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False
        status_code: int | None = None

        if self._client is not None:
            try:
                probs_df = self._client.query_probabilities(oid=object_id, format="pandas")
                probs = probs_df.to_dict(orient="records")
                raw = {"probabilities": probs}
                self.save_fixture(raw, fixture_path)
            except Exception:
                if fixture_path.exists():
                    raw = self.load_fixture(fixture_path)
                    fixture_used = True
                else:
                    raw = {}
                    fixture_used = True
        elif fixture_path.exists():
            raw = self.load_fixture(fixture_path)
            fixture_used = True

        fields = self._extract_fields(raw)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey="ZTF",
            source_endpoint="alerce.query_probabilities",
            request_params={"oid": object_id},
            status_code=status_code,
            fields=fields,
            events=fields,
            availability=not fixture_used,
            fixture_used=fixture_used,
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        if infer_identifier_kind(object_id) == "lsst_dia_object_id":
            return {}
        fixture_path = _FIXTURE_DIR / f"{object_id}_lc.json"
        if self._client is not None:
            try:
                lc = self._client.query_lightcurve(oid=object_id, format="json")
                self.save_fixture(lc, fixture_path)
                return lc
            except Exception:
                pass
        if fixture_path.exists():
            return self.load_fixture(fixture_path)
        return {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_fields(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        fields = []
        probs = raw.get("probabilities", [])
        if isinstance(probs, list):
            for entry in probs:
                cls_name = entry.get("class_name", "unknown")
                prob_val = entry.get("probability")
                classifier = entry.get("classifier_name", "unknown")
                classifier_version = entry.get("classifier_version")
                fields.append({
                    "field": f"{classifier}_{cls_name}",
                    "raw_label_or_score": prob_val,
                    "semantic_type": "probability",
                    "canonical_projection": float(prob_val) if prob_val is not None else None,
                    "classifier": classifier,
                    "class_name": cls_name,
                    "classifier_version": classifier_version,
                    "expert_key": f"alerce/{classifier}",
                    "event_scope": "object_snapshot",
                    "temporal_exactness": "latest_object_unsafe",
                    "event_time_jd": None,
                    "n_det": None,
                })
        return fields
