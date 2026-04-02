"""Lasair broker adapter.

Sherlock output is a crossmatch-based context label (crossmatch_tag),
not a calibrated posterior. Treat as a categorical feature only.

Two Lasair deployments matter here:
  1. ZTF Lasair uses string object IDs such as ``ZTF21abbzjeq``.
  2. LSST Lasair uses numeric ``diaObjectId`` values exposed via ``/api/object/``.
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
    is_lsst_lasair_endpoint,
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

        self._token = os.environ.get("LASAIR_TOKEN", "")
        self._site_base = os.environ.get("LASAIR_ENDPOINT", _SITE_BASE_ZTF).rstrip("/")
        self._api_base = build_lasair_api_endpoint(self._site_base)
        self._lsst_mode = is_lsst_lasair_endpoint(self._site_base)
        self._client = None
        if self._token:
            try:
                import lasair

                self._client = lasair.lasair_client(self._token, endpoint=self._api_base)
            except TypeError:
                try:
                    self._client = lasair.lasair_client(self._token)
                except Exception:
                    self._client = None
            except Exception:
                self._client = None

    def _auth_headers(self, *, accept: str = "application/json") -> dict[str, str]:
        headers = {"Accept": accept}
        if self._token:
            headers["Authorization"] = f"Token {self._token}"
        return headers

    def _query_api(
        self,
        *,
        selected: str,
        tables: str,
        conditions: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        response = requests.post(
            f"{self._api_base}/query/",
            headers=self._auth_headers(),
            data={
                "selected": selected,
                "tables": tables,
                "conditions": conditions,
                "limit": str(limit),
                "offset": str(offset),
            },
            timeout=self._timeout,
        )
        if response.status_code in (400, 401, 403, 404):
            response.raise_for_status()
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        if self._lsst_mode:
            if not self._token:
                try:
                    response = requests.get(
                        self._site_base + "/",
                        headers=self._auth_headers(accept="text/html"),
                        timeout=self._timeout,
                    )
                    response.raise_for_status()
                    return {
                        "broker": self.name,
                        "status": "reachable_no_token",
                        "mode": "html_root",
                        "endpoint": self._site_base,
                        "note": "LSST Lasair requires LASAIR_TOKEN for API queries",
                    }
                except Exception as exc:
                    return {"broker": self.name, "status": "error", "endpoint": self._site_base, "reason": str(exc)}

            try:
                rows = self._query_api(
                    selected="diaObjectId",
                    tables="objects",
                    conditions="1=1",
                    limit=1,
                )
                return {
                    "broker": self.name,
                    "status": "ok" if rows else "empty",
                    "mode": "query_api",
                    "endpoint": self._api_base,
                    "rows": len(rows),
                    "identifier_kind": "lsst_dia_object_id",
                }
            except Exception as exc:
                return {"broker": self.name, "status": "error", "endpoint": self._api_base, "reason": str(exc)}

        try:
            root_response = requests.get(
                self._site_base + "/",
                headers=self._auth_headers(),
                timeout=self._timeout,
            )
            if root_response.status_code >= 500:
                return {
                    "broker": self.name,
                    "status": "rest_api_error",
                    "http": root_response.status_code,
                    "endpoint": self._site_base,
                }

            object_response = requests.get(
                f"{self._site_base}/objects/ZTF21abbzjeq/",
                headers=self._auth_headers(),
                timeout=self._timeout,
            )
            if object_response.status_code == 200:
                return {
                    "broker": self.name,
                    "status": "ok",
                    "mode": "object_api",
                    "endpoint": self._site_base,
                    "identifier_kind": "ztf_object_id",
                }
            return {
                "broker": self.name,
                "status": "rest_api_error",
                "http": object_response.status_code,
                "endpoint": self._site_base,
            }
        except Exception as exc:
            return {
                "broker": self.name,
                "status": "fixture_fallback",
                "endpoint": self._site_base,
                "reason": f"REST API unreachable: {exc}",
            }

    def fetch_object(self, object_id: str) -> BrokerOutput:
        identifier_kind = infer_identifier_kind(object_id)
        survey = "LSST" if self._lsst_mode else "ZTF"
        request_params = {"objectId": object_id} if self._lsst_mode else {"object_id": object_id}
        source_endpoint = f"{self._api_base}/object/" if self._lsst_mode else f"{self._site_base}/objects/{object_id}/"

        if self._lsst_mode and identifier_kind != "lsst_dia_object_id":
            return self.unsupported_identifier_output(
                object_id,
                source_endpoint=source_endpoint,
                request_params=request_params,
                survey=survey,
                identifier_kind=identifier_kind,
                expected_identifier_kind="lsst_dia_object_id",
                reason="unsupported_identifier_for_endpoint",
            )

        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False
        status_code: int | None = None

        try:
            if self._lsst_mode:
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
        if self._lsst_mode and identifier_kind != "lsst_dia_object_id":
            return {}

        fixture_path = _FIXTURE_DIR / f"{object_id}_lc.json"
        try:
            if self._lsst_mode:
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
        response = requests.get(
            f"{self._site_base}/objects/{object_id}/",
            headers=self._auth_headers(),
            timeout=self._timeout,
        )
        if response.status_code in (404, 500):
            return {}, response.status_code
        response.raise_for_status()
        data = response.json()
        return (data if isinstance(data, dict) else {}), response.status_code

    def _fetch_lsst_object(self, object_id: str) -> tuple[dict[str, Any], int | None]:
        response = requests.post(
            f"{self._api_base}/object/",
            headers=self._auth_headers(),
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
