"""Fink broker adapter.

Supports both ZTF and LSST surveys:
  ZTF:  ``api.fink-portal.org`` — per-alert SNN/RF scores via ``d:`` columns
  LSST: ``api.lsst.fink-portal.org`` — per-alert SNN/CATS/EarlySNIa via ``f:clf_*`` columns

LSST Fink returns PER-ALERT classification scores that genuinely evolve
with detection count — this is the richest per-epoch signal for trust training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
import urllib3

from .base import BrokerAdapter, BrokerOutput, SemanticType
from .identifiers import infer_identifier_kind

_FIXTURE_DIR = Path("fixtures/raw/fink")
_ZTF_BASE_URL = "https://api.fink-portal.org"
_LSST_BASE_URL = "https://api.lsst.fink-portal.org"

# ZTF: Fields that are genuine probability scores [0,1]
_PROB_FIELDS = {
    "rf_snia_vs_nonia",
    "snn_snia_vs_nonia",
    "snn_sn_vs_all",
    "rf_kn_vs_nonkn",
}
# ZTF: Fields that are heuristic / non-probability scores
_SCORE_FIELDS = {
    "mulens_class_1",
    "mulens_class_2",
    "anomaly_score",
}
# ZTF: Fields that are classification strings (heuristic labels)
_CLASS_FIELDS = {"finkclass", "tracklet"}

# LSST: Fink classifier columns (f:clf_*)
_LSST_CLF_FIELDS = {
    "f:clf_snnSnVsOthers_score": ("probability", "fink_lsst/snn"),
    "f:clf_cats_class": ("heuristic_class", "fink_lsst/cats"),
    "f:clf_cats_score": ("probability", "fink_lsst/cats"),
    "f:clf_earlySNIa_score": ("probability", "fink_lsst/early_snia"),
}


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
        results = {}
        for label, base_url in [("ztf", _ZTF_BASE_URL), ("lsst", _LSST_BASE_URL)]:
            try:
                r = self._request("GET", f"{base_url}/api/v1/statistics")
                results[label] = {"status": "ok" if r.status_code == 200 else "error",
                                  "endpoint": base_url}
            except Exception as exc:
                results[label] = {"status": "error", "reason": str(exc)}
        return {"broker": self.name, "mode": "dual", **results}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        identifier_kind = infer_identifier_kind(object_id)
        is_lsst = identifier_kind == "lsst_dia_object_id"

        if is_lsst:
            return self._fetch_lsst_object(object_id)
        else:
            return self._fetch_ztf_object(object_id)

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        if infer_identifier_kind(object_id) == "lsst_dia_object_id":
            return {}  # LSST lightcurves come from ALeRCE/Lasair
        fixture_path = _FIXTURE_DIR / f"{object_id}_lc.json"
        try:
            r = self._request(
                "POST",
                f"{_ZTF_BASE_URL}/api/v1/objects",
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
    # ZTF path (existing)                                                  #
    # ------------------------------------------------------------------ #

    def _fetch_ztf_object(self, object_id: str) -> BrokerOutput:
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
            r = self._request("POST", f"{_ZTF_BASE_URL}/api/v1/objects", json=request_payload)
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

        fields = self._extract_ztf_fields(raw)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey="ZTF",
            source_endpoint=f"{_ZTF_BASE_URL}/api/v1/objects",
            request_params=request_payload,
            status_code=status_code,
            fields=fields,
            events=fields,
            availability=not fixture_used,
            fixture_used=fixture_used,
        )

    # ------------------------------------------------------------------ #
    # LSST path (new)                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_lsst_object(self, object_id: str) -> BrokerOutput:
        fixture_path = _FIXTURE_DIR / f"{object_id}_object.json"
        raw: dict[str, Any] = {}
        fixture_used = False
        status_code: int | None = None
        source_endpoint = f"{_LSST_BASE_URL}/api/v1/sources"
        request_payload = {"diaObjectId": object_id, "output-format": "json"}

        try:
            r = self._request("POST", source_endpoint, json=request_payload)
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

        fields = self._extract_lsst_fields(raw)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey="LSST",
            source_endpoint=source_endpoint,
            request_params=request_payload,
            status_code=status_code,
            fields=fields,
            events=fields,
            availability=not fixture_used,
            fixture_used=fixture_used,
        )

    # ------------------------------------------------------------------ #
    # Field extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_ztf_fields(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract score fields from ALL ZTF alerts, keyed by n_det."""
        fields = []
        alerts = raw.get("alerts", [])
        if not isinstance(alerts, list) or len(alerts) == 0:
            return fields

        def _jd(alert):
            return alert.get("i:jd") or alert.get("jd") or 0.0

        sorted_alerts = sorted(
            [a for a in alerts if isinstance(a, dict)],
            key=_jd
        )

        for alert in sorted_alerts:
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
                    "expert_key": self._ztf_expert_key_for_field(short),
                    "event_scope": "alert",
                    "temporal_exactness": "exact_alert",
                    "alert_id": int(candid) if candid is not None else None,
                    "n_det": n_det,
                    "alert_jd": float(jd) if jd is not None else None,
                    "event_time_jd": float(jd) if jd is not None else None,
                })
        return fields

    def _extract_lsst_fields(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract per-alert classifier scores from Fink LSST sources.

        Fink LSST returns per-alert rows with ``f:clf_*`` columns that
        genuinely evolve with detection count.
        """
        fields = []
        alerts = raw.get("alerts", [])
        if not isinstance(alerts, list) or len(alerts) == 0:
            return fields

        sorted_alerts = sorted(
            [a for a in alerts if isinstance(a, dict)],
            key=lambda a: a.get("r:midpointMjdTai") or 0,
        )

        for alert_idx, alert in enumerate(sorted_alerts):
            mjd = alert.get("r:midpointMjdTai")
            jd = (float(mjd) + 2400000.5) if mjd is not None and float(mjd) < 2400000.5 else mjd
            dia_source_id = alert.get("r:diaSourceId")
            band = alert.get("r:band")
            n_det = alert_idx + 1  # approximate — sorted by time

            # SNN: SN vs Others score (probability-like)
            snn_score = alert.get("f:clf_snnSnVsOthers_score")
            if snn_score is not None:
                snn_val = float(snn_score)
                if snn_val >= 0:  # -1 means not computed
                    fields.append({
                        "field": "clf_snnSnVsOthers_score",
                        "raw_label_or_score": snn_val,
                        "semantic_type": "probability",
                        "canonical_projection": snn_val,
                        "classifier": "snn_lsst",
                        "classifier_version": alert.get("f:fink_science_version"),
                        "expert_key": "fink_lsst/snn",
                        "event_scope": "alert",
                        "temporal_exactness": "exact_alert",
                        "alert_id": dia_source_id,
                        "n_det": n_det,
                        "alert_jd": float(jd) if jd is not None else None,
                        "event_time_jd": float(jd) if jd is not None else None,
                    })

            # CATS: class + score
            cats_class = alert.get("f:clf_cats_class")
            cats_score = alert.get("f:clf_cats_score")
            if cats_class is not None and cats_class != -1:
                fields.append({
                    "field": "clf_cats_class",
                    "raw_label_or_score": cats_class,
                    "semantic_type": "heuristic_class",
                    "canonical_projection": float(cats_score) if cats_score is not None else None,
                    "classifier": "cats_lsst",
                    "classifier_version": alert.get("f:fink_science_version"),
                    "class_name": str(cats_class),
                    "expert_key": "fink_lsst/cats",
                    "event_scope": "alert",
                    "temporal_exactness": "exact_alert",
                    "alert_id": dia_source_id,
                    "n_det": n_det,
                    "alert_jd": float(jd) if jd is not None else None,
                    "event_time_jd": float(jd) if jd is not None else None,
                })

            # EarlySNIa: score (-1 = not triggered)
            early_snia = alert.get("f:clf_earlySNIa_score")
            if early_snia is not None:
                early_val = float(early_snia)
                if early_val >= 0:  # -1 means not triggered
                    fields.append({
                        "field": "clf_earlySNIa_score",
                        "raw_label_or_score": early_val,
                        "semantic_type": "probability",
                        "canonical_projection": early_val,
                        "classifier": "early_snia_lsst",
                        "classifier_version": alert.get("f:fink_science_version"),
                        "expert_key": "fink_lsst/early_snia",
                        "event_scope": "alert",
                        "temporal_exactness": "exact_alert",
                        "alert_id": dia_source_id,
                        "n_det": n_det,
                        "alert_jd": float(jd) if jd is not None else None,
                        "event_time_jd": float(jd) if jd is not None else None,
                    })

        # Extract crossmatch fields from the latest alert for truth metadata.
        # These are static per-object (don't change with alerts), so take from last.
        if sorted_alerts:
            latest = sorted_alerts[-1]
            xm_fields = {}
            for key in ("f:xm_tns_fullname", "f:xm_tns_type", "f:xm_tns_redshift",
                        "f:xm_simbad_otype",
                        "f:xm_legacydr8_zphot", "f:xm_legacydr8_pstar", "f:xm_legacydr8_fqual",
                        "f:xm_gaiadr3_Plx", "f:xm_gaiadr3_VarFlag", "f:xm_gaiadr3_DR3Name",
                        "f:xm_mangrove_HyperLEDA_name", "f:xm_mangrove_lum_dist",
                        "f:xm_vsx_Type"):
                val = latest.get(key)
                if val is not None and str(val) not in ("None", "nan", "", "Fail"):
                    short_key = key.replace("f:xm_", "fink_xm_")
                    xm_fields[short_key] = val
            if xm_fields:
                # Store as a special "crossmatch" field so downstream truth
                # builders can access it from the silver events.
                fields.append({
                    "field": "fink_lsst_crossmatch",
                    "raw_label_or_score": json.dumps(xm_fields),
                    "semantic_type": "crossmatch_tag",
                    "canonical_projection": None,
                    "classifier": "fink_xm",
                    "classifier_version": latest.get("f:fink_science_version"),
                    "expert_key": "fink_lsst/crossmatch",
                    "event_scope": "static_context",
                    "temporal_exactness": "static_safe",
                    "alert_id": None,
                    "n_det": None,
                    "alert_jd": None,
                    "event_time_jd": None,
                })

        return fields

    @staticmethod
    def _ztf_expert_key_for_field(field_name: str) -> str:
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
