"""Lasair broker adapter.

Sherlock output is a crossmatch-based context label (crossmatch_tag),
not a calibrated posterior. Treat as a categorical feature only.

Two Lasair deployments:
  1. ZTF Lasair  — ``https://lasair-ztf.lsst.ac.uk``  (string ZTF IDs)
  2. LSST Lasair — ``https://lasair.lsst.ac.uk``       (numeric diaObjectIds)

The adapter is **dual-mode**: it queries the right endpoint based on the
object's identifier type. Both endpoints require ``LASAIR_TOKEN``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from .base import BrokerAdapter, BrokerOutput, SemanticType
from .identifiers import (
    build_lasair_api_endpoint,
    infer_identifier_kind,
)

_FIXTURE_DIR = Path("fixtures/raw/lasair")
_SITE_BASE_ZTF = "https://lasair-ztf.lsst.ac.uk"
_SITE_BASE_LSST = "https://lasair.lsst.ac.uk"

_SHERLOCK_CLASSES = ["SN", "NT", "AGN", "CV", "BS", "ORPHAN", "VS", "UNCLEAR"]


def extract_sherlock_label(raw: dict[str, Any]) -> str | None:
    direct = raw.get("sherlock_classification") or raw.get("classification")
    if direct is not None and str(direct).strip():
        return str(direct)

    sherlock = ((raw.get("lasairData") or {}).get("sherlock"))
    if isinstance(sherlock, str) and sherlock.strip():
        return sherlock.strip()
    if not isinstance(sherlock, dict) or not sherlock:
        return None

    for key in (
        "classification",
        "class",
        "class_name",
        "top_class",
        "best_class",
        "sherlock_classification",
        "label",
    ):
        value = sherlock.get(key)
        if value is not None and str(value).strip():
            return str(value)

    scored_labels = [
        (label, float(score))
        for label, score in sherlock.items()
        if label in _SHERLOCK_CLASSES and isinstance(score, (int, float))
    ]
    if scored_labels:
        return max(scored_labels, key=lambda item: item[1])[0]

    for label in _SHERLOCK_CLASSES:
        if sherlock.get(label):
            return label

    for value in sherlock.values():
        if not isinstance(value, dict):
            continue
        for key in ("classification", "class", "class_name", "label"):
            nested = value.get(key)
            if nested is not None and str(nested).strip():
                return str(nested)

    return None


class LasairAdapter(BrokerAdapter):
    name = "lasair"
    phase = 1
    semantic_type: SemanticType = "crossmatch_tag"

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        # Token: shared or per-endpoint
        self._ztf_token = os.environ.get("LASAIR_ZTF_TOKEN", "") or os.environ.get("LASAIR_TOKEN", "")
        self._lsst_token = os.environ.get("LASAIR_LSST_TOKEN", "") or os.environ.get("LASAIR_TOKEN", "")

        # Endpoints: user can set both explicitly
        self._ztf_base = os.environ.get("LASAIR_ZTF_ENDPOINT", _SITE_BASE_ZTF).rstrip("/")
        self._lsst_base = os.environ.get("LASAIR_LSST_ENDPOINT",
                          os.environ.get("LASAIR_ENDPOINT", _SITE_BASE_LSST)).rstrip("/")
        self._ztf_api = build_lasair_api_endpoint(self._ztf_base)
        self._lsst_api = build_lasair_api_endpoint(self._lsst_base)

        self._client = None
        if self._ztf_token:
            try:
                import lasair
                self._client = lasair.lasair_client(self._ztf_token, endpoint=self._ztf_api)
            except TypeError:
                try:
                    self._client = lasair.lasair_client(self._ztf_token)
                except Exception:
                    self._client = None
            except Exception:
                self._client = None

    def _auth_headers(self, *, accept: str = "application/json", survey: str = "ztf") -> dict[str, str]:
        headers = {"Accept": accept}
        token = self._lsst_token if survey == "lsst" else self._ztf_token
        if token:
            headers["Authorization"] = f"Token {token}"
        return headers

    def _query_api(
        self,
        *,
        selected: str,
        tables: str,
        conditions: str,
        limit: int = 1000,
        offset: int = 0,
        api_base: str | None = None,
        survey: str = "ztf",
    ) -> list[dict[str, Any]]:
        if api_base is None:
            api_base = self._ztf_api if survey == "ztf" else self._lsst_api
        response = requests.post(
            f"{api_base}/query/",
            headers=self._auth_headers(survey=survey),
            data={
                "selected": selected,
                "tables": tables,
                "conditions": conditions,
                "limit": str(limit),
                "offset": str(offset),
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        results = {}
        # Probe ZTF
        try:
            r = requests.get(
                self._ztf_base + "/",
                headers=self._auth_headers(accept="text/html", survey="ztf"),
                timeout=self._timeout,
            )
            results["ztf"] = {"status": "ok" if r.status_code == 200 else "error",
                              "endpoint": self._ztf_base,
                              "has_token": bool(self._ztf_token)}
        except Exception as exc:
            results["ztf"] = {"status": "error", "reason": str(exc)}

        # Probe LSST
        if self._lsst_token:
            try:
                rows = self._query_api(
                    selected="diaObjectId",
                    tables="objects",
                    conditions="1=1",
                    limit=1,
                    api_base=self._lsst_api,
                    survey="lsst",
                )
                results["lsst"] = {"status": "ok" if rows else "empty",
                                   "endpoint": self._lsst_base,
                                   "has_token": True, "rows": len(rows)}
            except Exception as exc:
                results["lsst"] = {"status": "error", "reason": str(exc)}
        else:
            results["lsst"] = {"status": "no_token", "endpoint": self._lsst_base}

        return {"broker": self.name, "mode": "dual", **results}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        identifier_kind = infer_identifier_kind(object_id)
        is_lsst_object = identifier_kind == "lsst_dia_object_id"
        survey = "LSST" if is_lsst_object else "ZTF"

        if is_lsst_object:
            source_endpoint = f"{self._lsst_api}/object/"
            request_params = {"objectId": object_id}
        else:
            source_endpoint = f"{self._ztf_base}/objects/{object_id}/"
            request_params = {"object_id": object_id}

        # LSST API requires token
        if is_lsst_object and not self._lsst_token:
            return self.unsupported_identifier_output(
                object_id,
                source_endpoint=source_endpoint,
                request_params=request_params,
                survey=survey,
                identifier_kind=identifier_kind,
                expected_identifier_kind=identifier_kind,
                reason="lasair_no_lsst_token (set LASAIR_LSST_TOKEN or LASAIR_TOKEN)",
            )

        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False
        status_code: int | None = None

        try:
            if is_lsst_object:
                raw, status_code = self._fetch_lsst_object(object_id)
            else:
                raw, status_code = self._fetch_ztf_object(object_id)
            if raw:
                self.save_fixture(raw, fixture_path)
        except Exception:
            raw = {}

        if not raw:
            raw, fixture_used = self._load_fixture_or_empty(fixture_path)

        fields = self._extract_fields(raw)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey=survey,
            source_endpoint=source_endpoint,
            request_params=request_params,
            status_code=status_code,
            fields=fields,
            events=fields,
            availability=bool(raw) and not fixture_used,
            fixture_used=fixture_used,
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        identifier_kind = infer_identifier_kind(object_id)
        is_lsst_object = identifier_kind == "lsst_dia_object_id"

        fixture_path = _FIXTURE_DIR / f"{object_id}_lc.json"
        try:
            if is_lsst_object:
                data, _ = self._fetch_lsst_object(object_id)
                if data:
                    self.save_fixture(data, fixture_path)
                    return data
            elif self._client is not None:
                lightcurves = self._client.lightcurves([object_id])
                data = lightcurves[0] if lightcurves else {}
                if data:
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

    def _fetch_ztf_object(self, object_id: str) -> tuple[dict[str, Any], int | None]:
        """Fetch from ZTF Lasair. Try REST page first, fall back to query API."""
        # Try direct object page
        try:
            response = requests.get(
                f"{self._ztf_base}/objects/{object_id}/",
                headers=self._auth_headers(survey="ztf"),
                timeout=self._timeout,
            )
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "json" in content_type:
                    data = response.json()
                    if isinstance(data, dict) and data:
                        return data, response.status_code
        except Exception:
            pass

        # Fall back to query API (needs token)
        if self._ztf_token:
            try:
                rows = self._query_api(
                    selected="objectId, sherlock_classification",
                    tables="objects",
                    conditions=f'objectId="{object_id}"',
                    limit=1,
                    api_base=self._ztf_api,
                    survey="ztf",
                )
                if rows:
                    return rows[0], 200
            except Exception:
                pass

        return {}, None

    def _fetch_lsst_object(self, object_id: str) -> tuple[dict[str, Any], int | None]:
        response = requests.post(
            f"{self._lsst_api}/object/",
            headers=self._auth_headers(survey="lsst"),
            data={"objectId": str(object_id)},
            timeout=self._timeout,
        )
        if response.status_code in (400, 404):
            return {}, response.status_code
        response.raise_for_status()
        data = response.json()
        return (data if isinstance(data, dict) else {}), response.status_code

    def _load_fixture_or_empty(self, path: Path) -> tuple[dict[str, Any], bool]:
        if path.exists():
            return self.load_fixture(path), True
        return {}, False

    def _extract_fields(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        sherlock = extract_sherlock_label(raw)
        if sherlock is None:
            return []
        return [{
            "field": "sherlock_classification",
            "raw_label_or_score": sherlock,
            "semantic_type": "crossmatch_tag",
            "canonical_projection": None,
            "classifier": "sherlock",
            "class_name": str(sherlock),
            "expert_key": "lasair/sherlock",
            "event_scope": "static_context",
            "temporal_exactness": "static_safe",
            "event_time_jd": None,
            "note": "context label only - not a calibrated posterior",
        }]
