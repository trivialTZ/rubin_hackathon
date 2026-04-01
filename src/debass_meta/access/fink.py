"""Fink broker adapter.

Uses the public Fink REST API (no credentials for object queries).
Captures all score fields + finkclass string, preserving semantic types.
finkclass is a heuristic routing label — NOT a calibrated posterior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
import urllib3

from .base import BrokerAdapter, BrokerOutput, SemanticType

_FIXTURE_DIR = Path("fixtures/raw/fink")
_BASE_URL = "https://api.fink-portal.org"

# Fields that are genuine probability scores [0,1]
_PROB_FIELDS = {
    "rf_snia_vs_nonia",
    "snn_snia_vs_nonia",
    "snn_sn_vs_all",
    "rf_kn_vs_nonkn",
}
# Fields that are heuristic / non-probability scores
_SCORE_FIELDS = {
    "mulens_class_1",
    "mulens_class_2",
    "anomaly_score",
}
# Fields that are classification strings (heuristic labels)
_CLASS_FIELDS = {"finkclass", "tracklet"}


class FinkAdapter(BrokerAdapter):
    name = "fink"
    phase = 1
    semantic_type: SemanticType = "mixed"

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout
        self._allow_insecure_ssl = True

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        try:
            r = self._request(
                "GET",
                f"{_BASE_URL}/api/v1/statistics",
            )
            r.raise_for_status()
            return {"broker": self.name, "status": "ok", "http": r.status_code}
        except Exception as exc:
            return {"broker": self.name, "status": "error", "reason": str(exc)}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False
        status_code: int | None = None
        request_payload = {
            "objectId": object_id,
            "output-format": "json",
            "columns": "i:jd,i:candid,i:ndethist,d:rf_snia_vs_nonia,d:snn_snia_vs_nonia,d:snn_sn_vs_all,d:rf_kn_vs_nonkn,d:mulens_class_1,d:finkclass",
        }

        try:
            r = self._request(
                "POST",
                f"{_BASE_URL}/api/v1/objects",
                json=request_payload,
            )
            status_code = r.status_code
            r.raise_for_status()
            data = r.json()
            raw = {"alerts": data}
            self.save_fixture(raw, fixture_path)
        except Exception:
            if fixture_path.exists():
                raw = self.load_fixture(fixture_path)
                fixture_used = True
            else:
                raw = {}
                fixture_used = True

        fields = self._extract_fields(raw)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey="ZTF",
            source_endpoint=f"{_BASE_URL}/api/v1/objects",
            request_params=request_payload,
            status_code=status_code,
            fields=fields,
            events=fields,
            availability=not fixture_used,
            fixture_used=fixture_used,
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        fixture_path = _FIXTURE_DIR / f"{object_id}_lc.json"
        try:
            r = self._request(
                "POST",
                f"{_BASE_URL}/api/v1/objects",
                json={
                    "objectId": object_id,
                    "output-format": "json",
                    "columns": "i:jd,i:magpsf,i:sigmapsf,i:fid,i:ra,i:dec",
                },
            )
            r.raise_for_status()
            lc = r.json()
            self.save_fixture(lc, fixture_path)
            return lc
        except Exception:
            if fixture_path.exists():
                return self.load_fixture(fixture_path)
            return {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_fields(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract score fields from ALL alerts, keyed by n_det (ndethist).

        Each returned field dict includes `n_det` and `alert_jd` so the
        epoch table can key rows by detection count rather than query time.
        Alerts are sorted oldest-first so n_det increases monotonically.
        """
        fields = []
        alerts = raw.get("alerts", [])
        if not isinstance(alerts, list) or len(alerts) == 0:
            return fields

        # Sort oldest-first by JD (Fink returns newest-first by default)
        def _jd(alert):
            return alert.get("i:jd") or alert.get("jd") or 0.0

        sorted_alerts = sorted(
            [a for a in alerts if isinstance(a, dict)],
            key=_jd
        )

        for alert in sorted_alerts:
            # n_det: number of previous detections including this one
            ndethist = alert.get("i:ndethist") or alert.get("ndethist")
            jd = alert.get("i:jd") or alert.get("jd")
            candid = alert.get("i:candid") or alert.get("candid")
            n_det = int(ndethist) if ndethist is not None else None

            for key, val in alert.items():
                short = key.split(":")[-1]
                if short in _PROB_FIELDS:
                    stype = "probability"
                    proj = float(val) if val is not None else None
                elif short in _SCORE_FIELDS:
                    stype = "binary_score"
                    proj = None
                elif short in _CLASS_FIELDS:
                    stype = "heuristic_class"
                    proj = None
                else:
                    continue
                fields.append({
                    "field": short,
                    "raw_label_or_score": val,
                    "semantic_type": stype,
                    "canonical_projection": proj,
                    "classifier": "snn" if short.startswith("snn_") else "rf" if short.startswith("rf_") else "fink",
                    "classifier_version": None,
                    "expert_key": self._expert_key_for_field(short),
                    "event_scope": "alert",
                    "temporal_exactness": "exact_alert",
                    "alert_id": int(candid) if candid is not None else None,
                    "n_det": n_det,
                    "alert_jd": float(jd) if jd is not None else None,
                    "event_time_jd": float(jd) if jd is not None else None,
                })
        return fields

    @staticmethod
    def _expert_key_for_field(field_name: str) -> str:
        if field_name.startswith("snn_"):
            return "fink/snn"
        if field_name == "rf_snia_vs_nonia":
            return "fink/rf_ia"
        return "fink/aux"

    def _request(self, method: str, url: str, **kwargs):
        kwargs.setdefault("timeout", self._timeout)
        try:
            return requests.request(method, url, **kwargs)
        except requests.exceptions.SSLError:
            if not self._allow_insecure_ssl:
                raise
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            return requests.request(method, url, verify=False, **kwargs)
