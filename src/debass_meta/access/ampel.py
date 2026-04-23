"""AMPEL broker adapter — Phase 1 integration.

AMPEL (Nordin et al. 2025) is a state-indexed broker from DESY/Stockholm.
Every T2 classifier result is stored per-state (per-alert), so retrieval of
"the classifier output at the state with N detections" is native — this is
the cleanest per-epoch semantics of any full-stream broker.

DEBASS targets two T2 units:
  • SNGuess (T2MultiXGBClassifier) — binary young-extragalactic-transient P
  • FollowMe (T2RunParsnip)         — 14-class ParSNIP + priors

Access: DESY live-query + archive REST APIs.  Requires an AMPEL channel
token (`AMPEL_ARCHIVE_TOKEN`) for full archive access.  For public live
queries a token may not be required; the adapter gracefully degrades.

Docs: https://ampelproject.github.io/
Paper: Nordin+ 2025 A&A (arXiv 2501.16511)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import urllib3

from .base import BrokerAdapter, BrokerOutput, SemanticType
from .identifiers import infer_identifier_kind

_FIXTURE_DIR = Path("fixtures/raw/ampel")
_DEFAULT_BASE_URL = "https://ampel.zeuthen.desy.de"

# Map T2 unit name → DEBASS expert key
_T2_TO_EXPERT: dict[str, str] = {
    "T2MultiXGBClassifier": "ampel/snguess",
    "T2SNGuess": "ampel/snguess",
    "T2RunParsnip": "ampel/parsnip_followme",
    "T2ElasticcReport": "ampel/parsnip_followme",
}


class AmpelAdapter(BrokerAdapter):
    name = "ampel"
    phase = 1
    semantic_type: SemanticType = "mixed"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 20,
        token: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("AMPEL_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._token = token or os.environ.get("AMPEL_ARCHIVE_TOKEN")
        self._allow_insecure_ssl = True

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        endpoint = f"{self._base_url}/api/live/status"
        try:
            r = self._request("GET", endpoint)
            return {
                "broker": self.name,
                "status": "ok" if r.status_code == 200 else "error",
                "endpoint": endpoint,
                "has_token": bool(self._token),
            }
        except Exception as exc:
            return {
                "broker": self.name,
                "status": "unreachable",
                "endpoint": endpoint,
                "reason": str(exc),
                "has_token": bool(self._token),
            }

    def fetch_object(self, object_id: str) -> BrokerOutput:
        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False
        availability = False
        status_code: int | None = None
        source_endpoint = f"{self._base_url}/api/archive/object/{object_id}/t2"
        reason: str | None = None

        try:
            r = self._request("GET", source_endpoint)
            status_code = r.status_code
            if r.status_code == 200:
                raw = r.json()
                self.save_fixture(raw, fixture_path)
                availability = True
            else:
                reason = f"HTTP {r.status_code}"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if fixture_path.exists():
                raw = self.load_fixture(fixture_path)
                fixture_used = True

        events = self._extract_events(raw, survey=self._infer_survey(object_id))
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey=self._infer_survey(object_id),
            source_endpoint=source_endpoint,
            status_code=status_code,
            fields=events,
            events=events,
            availability=availability,
            fixture_used=fixture_used,
            request_params={"identifier": object_id, "reason": reason},
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------ #
    # Event extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_events(
        self, raw: dict[str, Any], *, survey: str = "ZTF"
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        t2_docs = raw.get("t2_documents") or raw.get("t2") or raw.get("documents") or []
        if not isinstance(t2_docs, list):
            return events

        for doc in t2_docs:
            if not isinstance(doc, dict):
                continue
            unit = doc.get("unit") or doc.get("t2_unit") or doc.get("unit_name")
            expert_key = _T2_TO_EXPERT.get(unit)
            if expert_key is None:
                continue
            # State info: AMPEL stores per-state snapshots; state_id ≈ alert_id
            state = doc.get("state") or doc.get("state_id") or doc.get("link")
            # n_det per state (explicit in some payloads)
            n_det = doc.get("n_det") or doc.get("ndet") or doc.get("n_detections")
            mjd = doc.get("mjd") or doc.get("event_mjd")
            jd = (float(mjd) + 2400000.5) if mjd is not None and float(mjd) < 2400000.5 else mjd
            body = doc.get("body") or doc.get("result") or {}
            if isinstance(body, list) and body:
                body = body[-1]

            # SNGuess binary P(young extragalactic transient)
            if expert_key == "ampel/snguess":
                score = (
                    body.get("prob") if isinstance(body, dict) else None
                ) or body.get("probability") if isinstance(body, dict) else None
                if isinstance(body, dict):
                    score = (
                        body.get("prob")
                        or body.get("probability")
                        or body.get("snguess_prob")
                        or body.get("snguess_probability")
                    )
                events.append({
                    "field": "snguess_prob",
                    "raw_label_or_score": float(score) if score is not None else None,
                    "semantic_type": "probability",
                    "canonical_projection": float(score) if score is not None else None,
                    "classifier": unit,
                    "classifier_version": doc.get("version"),
                    "class_name": None,
                    "expert_key": expert_key,
                    "event_scope": "alert",
                    "temporal_exactness": "exact_alert",
                    "alert_id": state,
                    "n_det": int(n_det) if n_det is not None else None,
                    "alert_jd": float(jd) if jd is not None else None,
                    "event_time_jd": float(jd) if jd is not None else None,
                })
            else:
                # ParSNIP FollowMe: 14-class probability vector in body
                probs: dict[str, float] = {}
                if isinstance(body, dict):
                    for key in ("class_probabilities", "probs", "parsnip_probs"):
                        if key in body and isinstance(body[key], dict):
                            probs = {
                                str(k): float(v) for k, v in body[key].items()
                            }
                            break
                if not probs and isinstance(body, dict):
                    # Flat `p_<class>` keys
                    probs = {
                        k.replace("p_", ""): float(v)
                        for k, v in body.items()
                        if k.startswith("p_") and isinstance(v, (int, float))
                    }
                events.append({
                    "field": "parsnip_probs",
                    "raw_label_or_score": max(probs.values()) if probs else None,
                    "semantic_type": "probability",
                    "canonical_projection": (
                        max(probs.values()) if probs else None
                    ),
                    "classifier": unit,
                    "classifier_version": doc.get("version"),
                    "class_name": max(probs, key=probs.get) if probs else None,
                    "expert_key": expert_key,
                    "event_scope": "alert",
                    "temporal_exactness": "exact_alert",
                    "alert_id": state,
                    "n_det": int(n_det) if n_det is not None else None,
                    "alert_jd": float(jd) if jd is not None else None,
                    "event_time_jd": float(jd) if jd is not None else None,
                    "raw_probs_json": __import__("json").dumps(probs),
                })
        return events

    @staticmethod
    def _infer_survey(object_id: str) -> str:
        return "LSST" if infer_identifier_kind(object_id) == "lsst_dia_object_id" else "ZTF"

    # ------------------------------------------------------------------ #
    # HTTP                                                                 #
    # ------------------------------------------------------------------ #

    def _request(self, method: str, url: str, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        if self._token:
            headers.setdefault("Authorization", f"Bearer {self._token}")
        kwargs.setdefault("timeout", self._timeout)
        try:
            return requests.request(method, url, headers=headers, **kwargs)
        except requests.exceptions.SSLError:
            if not self._allow_insecure_ssl:
                raise
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            return requests.request(method, url, headers=headers, verify=False, **kwargs)
