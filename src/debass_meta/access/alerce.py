"""ALeRCE broker adapter.

Uses the public alerce Python client (v2.3+). Supports both ZTF and LSST surveys
via the ``survey`` parameter.

Classifiers returned depend on the object era and survey:
  ZTF (older objects):  lc_classifier_transient, stamp_classifier
  ZTF (newer objects):  lc_classifier_BHRF_forced_phot_transient, ATAT, stamp_classifier_2025_beta
  LSST:                 stamp_classifier_rubin_beta (+ LC classifiers when available)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import BrokerAdapter, BrokerOutput, SemanticType
from .identifiers import infer_identifier_kind

_FIXTURE_DIR = Path("fixtures/raw/alerce")


class AlerceAdapter(BrokerAdapter):
    name = "alerce"
    phase = 1
    semantic_type: SemanticType = "probability"

    def __init__(self) -> None:
        try:
            from alerce.core import Alerce
            self._client = Alerce()
            # Check if client supports survey param (v2.1+)
            import inspect
            sig = inspect.signature(self._client.query_probabilities)
            self._has_survey_param = "survey" in sig.parameters
        except ImportError:
            self._client = None
            self._has_survey_param = False

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        if self._client is None:
            return {"broker": self.name, "status": "unavailable", "reason": "alerce not installed"}
        try:
            kwargs: dict[str, Any] = {"format": "pandas", "page": 1, "page_size": 1}
            if self._has_survey_param:
                kwargs["survey"] = "ztf"
            result = self._client.query_objects(**kwargs)
            ok = result is not None and len(result) > 0
            return {"broker": self.name, "status": "ok" if ok else "empty",
                    "rows": len(result) if result is not None else 0,
                    "lsst_support": self._has_survey_param}
        except Exception as exc:
            return {"broker": self.name, "status": "error", "reason": str(exc)}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        identifier_kind = infer_identifier_kind(object_id)
        is_lsst = identifier_kind == "lsst_dia_object_id"

        # LSST objects require survey param (alerce >=2.1)
        if is_lsst and not self._has_survey_param:
            return self.unsupported_identifier_output(
                object_id,
                source_endpoint="alerce.query_probabilities",
                request_params={"oid": object_id},
                survey="LSST",
                identifier_kind=identifier_kind,
                expected_identifier_kind="ztf_object_id",
                reason="alerce_client_too_old_for_lsst (need >=2.1.0)",
            )

        survey_str = "lsst" if is_lsst else "ztf"
        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False
        status_code: int | None = None

        if self._client is not None:
            try:
                kwargs: dict[str, Any] = {"oid": object_id, "format": "pandas"}
                if self._has_survey_param:
                    kwargs["survey"] = survey_str
                probs_df = self._client.query_probabilities(**kwargs)
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
            survey="LSST" if is_lsst else "ZTF",
            source_endpoint="alerce.query_probabilities",
            request_params={"oid": object_id, "survey": survey_str},
            status_code=status_code,
            fields=fields,
            events=fields,
            availability=not fixture_used,
            fixture_used=fixture_used,
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        identifier_kind = infer_identifier_kind(object_id)
        is_lsst = identifier_kind == "lsst_dia_object_id"
        survey_str = "lsst" if is_lsst else "ztf"

        fixture_path = _FIXTURE_DIR / f"{object_id}_lc.json"
        if self._client is not None:
            try:
                kwargs: dict[str, Any] = {"oid": object_id, "format": "json"}
                if self._has_survey_param:
                    kwargs["survey"] = survey_str
                lc = self._client.query_lightcurve(**kwargs)
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
                # Stamp classifiers score the first detection image only — they
                # don't change with more detections, so they are static_safe.
                # LC classifiers are object-level snapshots — temporally unsafe.
                is_stamp = "stamp" in classifier.lower()
                fields.append({
                    "field": f"{classifier}_{cls_name}",
                    "raw_label_or_score": prob_val,
                    "semantic_type": "probability",
                    "canonical_projection": float(prob_val) if prob_val is not None else None,
                    "classifier": classifier,
                    "class_name": cls_name,
                    "classifier_version": classifier_version,
                    "expert_key": f"alerce/{classifier}",
                    "event_scope": "object_snapshot" if not is_stamp else "static_context",
                    "temporal_exactness": "static_safe" if is_stamp else "latest_object_unsafe",
                    "event_time_jd": None,
                    "n_det": None,
                })
        return fields
