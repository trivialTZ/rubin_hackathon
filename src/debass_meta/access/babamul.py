"""Babamul broker adapter — Caltech/UMN, BOOM backend.

Verified architecture (2026-04-26, see ``project_babamul_verification.md``):

  - DP1 NOT backfilled (live-stream only, Feb 2026+).
  - ZTF retro-ingested across 2017-2023 (50/50 bulk-XM hits in probe).
  - REST returns JSON; Kafka returns Avro Object Container Files.
  - AppleCiDEr is gating logic (Kafka topic name = class), NOT a per-message
    score. 6-state encoding: 3-way ``applecider_class`` × 2-way
    ``cross_survey_match``.

This adapter exposes the **REST historical** path. The live Kafka stream
path (with topic-as-AppleCiDEr-class) is a separate daemon and not needed
for v7-on-existing-ZTF-cohort training.

Per-object features extracted (6):
  babamul_star_flag             properties.star
  babamul_near_brightstar_flag  properties.near_brightstar
  babamul_rock_flag             properties.rock
  babamul_stationary_flag       properties.stationary
  babamul_xmatch_lsst           1 if survey_matches.lsst is non-null else 0
  babamul_xmatch_ztf            1 if survey_matches.ztf is non-null else 0
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from .base import BrokerAdapter, BrokerOutput, SemanticType
from .identifiers import infer_identifier_kind

_FIXTURE_DIR = Path("fixtures/raw/babamul")

_API_BASE_BY_ENV = {
    "production": "https://babamul.caltech.edu/api/babamul",
    "staging": "https://staging.babamul.caltech.edu/api/babamul",
}

# Fields the adapter emits per object. Names must stay in sync with
# silver/gold downstream consumers.
_FEATURE_NAMES = (
    "babamul_star_flag",
    "babamul_near_brightstar_flag",
    "babamul_rock_flag",
    "babamul_stationary_flag",
    "babamul_xmatch_lsst",
    "babamul_xmatch_ztf",
)


def _api_base() -> str:
    env = os.environ.get("BABAMUL_ENV", "production")
    return _API_BASE_BY_ENV.get(env, env)


class BabamulAdapter(BrokerAdapter):
    name = "babamul"
    phase = 1
    # context_class fits — Babamul carries alert-time computed flags
    # (star/near_brightstar/rock/stationary) that act as contaminant tags,
    # not classification probabilities.
    semantic_type: SemanticType = "context_class"

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout
        self._token = os.environ.get("BABAMUL_API_TOKEN", "")
        self._base = _api_base()

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        if not self._token:
            return {"broker": self.name, "status": "error", "reason": "no BABAMUL_API_TOKEN"}
        try:
            r = self._get(f"{self._base}/profile")
            ok = r.status_code == 200
            return {"broker": self.name,
                    "status": "ok" if ok else "error",
                    "endpoint": self._base,
                    "http": r.status_code}
        except Exception as exc:
            return {"broker": self.name, "status": "error", "reason": str(exc)}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        identifier_kind = infer_identifier_kind(object_id)
        survey = "LSST" if identifier_kind == "lsst_dia_object_id" else "ZTF"
        endpoint = f"{self._base}/surveys/{survey}/objects/{object_id}"

        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False
        status_code: int | None = None

        try:
            r = self._get(endpoint)
            status_code = r.status_code
            if r.status_code == 200:
                raw = r.json().get("data", r.json())
                if not isinstance(raw, dict):
                    raw = {"_data": raw}
                self.save_fixture(raw, fixture_path)
            elif r.status_code == 404:
                # Honest "no coverage" — record it as unavailable rather than
                # silently retry. Babamul retro-ingest is uneven across years.
                return self.unavailable_output(
                    object_id,
                    source_endpoint=endpoint,
                    survey=survey,
                    identifier_kind=identifier_kind,
                    reason="not_indexed_by_babamul",
                    raw_payload_extra={"http_status": 404},
                )
            else:
                if fixture_path.exists():
                    raw = self.load_fixture(fixture_path)
                    fixture_used = True
                else:
                    return self.unavailable_output(
                        object_id,
                        source_endpoint=endpoint,
                        survey=survey,
                        identifier_kind=identifier_kind,
                        reason=f"http_{r.status_code}",
                    )
        except Exception:
            if fixture_path.exists():
                raw = self.load_fixture(fixture_path)
                fixture_used = True
            else:
                raw = {}
                fixture_used = True

        events = self._extract_event(raw, object_id, survey)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey=survey,
            source_endpoint=endpoint,
            request_params={},
            status_code=status_code,
            fields=events,
            events=events,
            availability=bool(events) and not fixture_used,
            fixture_used=fixture_used,
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        """Return the Babamul-cached lightcurve from the per-object endpoint.

        Babamul embeds prv_candidates / fp_hists in the object payload, so
        we reuse fetch_object's cache when available.
        """
        identifier_kind = infer_identifier_kind(object_id)
        survey = "LSST" if identifier_kind == "lsst_dia_object_id" else "ZTF"
        endpoint = f"{self._base}/surveys/{survey}/objects/{object_id}"
        try:
            r = self._get(endpoint)
            if r.status_code == 200:
                payload = r.json().get("data", r.json())
                if isinstance(payload, dict):
                    return {
                        "candidate": payload.get("candidate"),
                        "prv_candidates": payload.get("prv_candidates", []),
                        "prv_nondetections": payload.get("prv_nondetections", []),
                        "fp_hists": payload.get("fp_hists", []),
                    }
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------ #
    # Bulk REST helpers — used by scripts/backfill.py for cohort pulls    #
    # ------------------------------------------------------------------ #

    def bulk_cross_match(
        self,
        object_ids: list[str],
        survey: str = "ZTF",
        batch_size: int = 100,
        n_threads: int = 8,
    ) -> dict[str, dict[str, Any] | None]:
        """Bulk catalog cross-match lookup. Returns {object_id: catalog_dict|None}.

        IMPORTANT: this returns CATALOG ENRICHMENT (Gaia, TNS, 2MASS, NED, VSX,
        Milliquas, ...) — NOT the alert payload. For per-object alert features
        (``properties.star``, ``survey_matches``, etc.) use ``bulk_fetch_objects``.

        Coverage is high (~100% on Babamul-indexed Babamul-known IDs); used
        for negative-class enrichment beyond the existing Gaia/SIMBAD baseline.

        Endpoint: ``POST /surveys/{survey}/objects/cross-matches``. Response
        shape: ``{"status","message","data": {object_id: catalog_dict, ...}}``.
        Caps batch_size to 100 per api.py guards in boom-astro/babamul.
        """
        if not object_ids:
            return {}
        url = f"{self._base}/surveys/{survey}/objects/cross-matches"
        out: dict[str, dict[str, Any] | None] = {oid: None for oid in object_ids}
        for start in range(0, len(object_ids), min(batch_size, 100)):
            chunk = object_ids[start: start + batch_size]
            try:
                r = self._post(url, json={"object_ids": chunk, "n_threads": n_threads})
                if r.status_code != 200:
                    continue
                body = r.json()
                # Per the live endpoint: data is a dict keyed by object_id.
                # Older versions may return a list; handle both shapes.
                data = body.get("data", body) if isinstance(body, dict) else body
                if isinstance(data, dict):
                    for oid in chunk:
                        item = data.get(oid)
                        out[oid] = item if item else None
                elif isinstance(data, list) and len(data) == len(chunk):
                    for oid, item in zip(chunk, data):
                        out[oid] = item if item else None
            except Exception:
                continue
        return out

    def bulk_fetch_objects(
        self,
        object_ids: list[str],
        survey: str = "ZTF",
        n_threads: int = 16,
    ) -> dict[str, dict[str, Any] | None]:
        """Threaded per-object alert fetch. Returns {object_id: alert_dict|None}.

        This is the right endpoint for the 6 Babamul features
        (``babamul_star_flag``, ``babamul_xmatch_*``, etc.) — those live in
        the per-object alert payload, not in the cross-match catalog.

        Uses the per-object GET endpoint with a thread pool. 404s become
        ``None`` (Babamul did not retro-ingest that object).
        """
        from concurrent.futures import ThreadPoolExecutor

        out: dict[str, dict[str, Any] | None] = {oid: None for oid in object_ids}

        def _one(oid: str) -> tuple[str, dict[str, Any] | None]:
            url = f"{self._base}/surveys/{survey}/objects/{oid}"
            try:
                r = self._get(url)
                if r.status_code != 200:
                    return oid, None
                body = r.json()
                data = body.get("data", body) if isinstance(body, dict) else body
                return oid, (data if isinstance(data, dict) else None)
            except Exception:
                return oid, None

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for oid, payload in ex.map(_one, object_ids):
                out[oid] = payload
        return out

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _get(self, url: str, **kw):
        return requests.get(url, headers=self._headers(), timeout=self._timeout, **kw)

    def _post(self, url: str, **kw):
        return requests.post(url, headers=self._headers(), timeout=self._timeout, **kw)

    def _extract_event(
        self,
        raw: dict[str, Any],
        object_id: str,
        survey: str,
    ) -> list[dict[str, Any]]:
        """Pull the 6 Babamul features into a single event row.

        One event per object; per-epoch evolution would need the live Kafka
        stream (bronze records the topic per message). For REST historical
        backfill, properties + survey_matches are object-level snapshots.
        """
        if not raw:
            return []
        props = raw.get("properties") or {}
        sm = raw.get("survey_matches") or {}
        features = {
            "babamul_star_flag": _bool(props.get("star")),
            "babamul_near_brightstar_flag": _bool(props.get("near_brightstar")),
            "babamul_rock_flag": _bool(props.get("rock")),
            "babamul_stationary_flag": _bool(props.get("stationary")),
            "babamul_xmatch_lsst": 1 if sm.get("lsst") else 0,
            "babamul_xmatch_ztf": 1 if sm.get("ztf") else 0,
        }
        candidate = raw.get("candidate") or {}
        event_jd = candidate.get("jd") if isinstance(candidate, dict) else None
        # expert_key="babamul" routes through the babamul projector;
        # temporal_exactness=static_safe matches lasair/sherlock — the per-object
        # endpoint gives us alert-time computed flags, not per-epoch evolution.
        return [
            {
                "expert_key": "babamul",
                "classifier": "babamul",
                "field": fname,
                "raw_label_or_score": value,
                "semantic_type": "binary_score",
                "canonical_projection": float(value) if value is not None else None,
                "alert_jd": event_jd,
                "survey": survey,
                "temporal_exactness": "static_safe",
                "event_scope": "static_context",
            }
            for fname, value in features.items()
        ]


def _bool(v: Any) -> int | None:
    """Normalise Babamul booleans to 0/1 (None if missing)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if v else 0
    if isinstance(v, str):
        return 1 if v.lower() in ("true", "1", "yes") else 0
    return None
