"""Pitt-Google Broker adapter — BigQuery-backed SuperNNova queries.

Queries via ``pittgoogle-client`` (Google Cloud BigQuery):
  LSST: ``pitt-alert-broker.lsst.supernnova``   → binary P(Ia) per alert
  ZTF:  ``ardent-cycling-243415.ztf.SuperNNova`` → binary P(Ia) per alert

Authentication — set these env vars before use::

    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    export GOOGLE_CLOUD_PROJECT=your-gcp-billing-project

If credentials are absent (or ``pittgoogle-client`` is not installed), the
adapter returns ``availability=False`` without raising, so the rest of the
pipeline proceeds.  Set ``PITTGOOGLE_SKIP=1`` to suppress the import warning.

Install::

    pip install pittgoogle-client>=0.3

GCP resources (read-only public datasets — no PGB token needed):
  LSST topics:   projects/pitt-alert-broker/topics/lsst-*
  ZTF topics:    projects/ardent-cycling-243415/topics/ztf-*
  LSST BQ:       pitt-alert-broker.lsst.<table>
  ZTF BQ:        ardent-cycling-243415.ztf.<table>
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from .base import BrokerAdapter, BrokerOutput, SemanticType
from .identifiers import infer_identifier_kind

# ---------------------------------------------------------------------------
# Optional pittgoogle import
# ---------------------------------------------------------------------------

try:
    import pittgoogle  # type: ignore[import-untyped]

    _PITTGOOGLE_AVAILABLE = True
except ImportError:
    _PITTGOOGLE_AVAILABLE = False
    if not os.getenv("PITTGOOGLE_SKIP"):
        warnings.warn(
            "pittgoogle-client not installed — PittAdapter will return "
            "availability=False.  Run: pip install pittgoogle-client>=0.3",
            stacklevel=2,
        )

# ---------------------------------------------------------------------------
# GCP resource identifiers
# ---------------------------------------------------------------------------

_PITTGOOGLE_GCP_PROJECT = "pitt-alert-broker"
_ZTF_GCP_PROJECT = "ardent-cycling-243415"

_LSST_SUPERNNOVA_TABLE = f"`{_PITTGOOGLE_GCP_PROJECT}.lsst.supernnova`"
_ZTF_SUPERNNOVA_TABLE = f"`{_ZTF_GCP_PROJECT}.ztf.SuperNNova`"

_FIXTURE_DIR = Path("fixtures/raw/pittgoogle")

# SuperNNova probability column — PGB uses these names
# class 0 = SN Ia, class 1 = non-Ia.  We try each in order.
_SNN_IA_PROB_CANDIDATES = [
    "SuperNNova_prob_class0",   # ZTF confirmed column name
    "supernnova_prob_class0",   # LSST likely variant
    "prob_class0",
    "pIa",
    "p_Ia",
    "ia_prob",
]
_SNN_NONIA_PROB_CANDIDATES = [
    "SuperNNova_prob_class1",
    "supernnova_prob_class1",
    "prob_class1",
    "pnonIa",
    "p_nonIa",
]
_SNN_PRED_CLASS_CANDIDATES = [
    "SuperNNova_predicted_class",
    "supernnova_predicted_class",
    "predicted_class",
]
_TIMESTAMP_CANDIDATES = ["jd", "midpointMjdTai", "mjd", "timestamp", "jd_alert"]
_OBJECT_ID_CANDIDATES = {
    "lsst": ["diaObjectId", "dia_object_id"],
    "ztf": ["objectId", "object_id"],
}


class PittAdapter(BrokerAdapter):
    """Pitt-Google Broker adapter.

    Expert keys produced:
    - ``pittgoogle/supernnova_lsst`` — LSST SuperNNova P(Ia) (per alert,
      latest_unsafe because BQ query returns aggregate object view)
    - ``pittgoogle/supernnova_ztf``  — ZTF SuperNNova P(Ia)
    """

    name = "pitt_google"
    phase = 1
    semantic_type: SemanticType = "probability"

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def probe(self) -> dict[str, Any]:
        if not _PITTGOOGLE_AVAILABLE:
            return {"broker": self.name, "status": "unavailable",
                    "reason": "pittgoogle-client not installed"}
        if not self._has_credentials():
            return {"broker": self.name, "status": "unavailable",
                    "reason": "no GCP credentials (set GOOGLE_APPLICATION_CREDENTIALS)"}
        try:
            client = pittgoogle.bigquery.Client()
            df = client.query(
                f"SELECT COUNT(*) AS n FROM {_LSST_SUPERNNOVA_TABLE} LIMIT 1"
            )
            n = int(df["n"].iloc[0]) if len(df) > 0 else 0
            return {"broker": self.name, "status": "ok",
                    "lsst_supernnova_row_count": n}
        except Exception as exc:
            return {"broker": self.name, "status": "error", "reason": str(exc)}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        identifier_kind = infer_identifier_kind(object_id)
        if identifier_kind == "lsst_dia_object_id":
            return self._fetch_lsst(object_id)
        if identifier_kind == "ztf_object_id":
            return self._fetch_ztf(object_id)
        return self.unsupported_identifier_output(
            object_id,
            source_endpoint="bigquery",
            identifier_kind=identifier_kind,
            expected_identifier_kind="lsst_dia_object_id",
            reason="unsupported_identifier",
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        # PGB does not serve lightcurves via this adapter;
        # use the ALeRCE or Fink LSST adapter instead.
        return {}

    # ------------------------------------------------------------------ #
    # LSST path                                                            #
    # ------------------------------------------------------------------ #

    def _fetch_lsst(self, object_id: str) -> BrokerOutput:
        fixture_path = _FIXTURE_DIR / f"{object_id}_supernnova.json"
        source_endpoint = _LSST_SUPERNNOVA_TABLE
        request_params = {"diaObjectId": object_id}

        raw, fixture_used = self._query_or_fixture(
            sql=f"SELECT * FROM {_LSST_SUPERNNOVA_TABLE} WHERE diaObjectId = {object_id} ORDER BY 1 LIMIT 500",
            fixture_path=fixture_path,
            survey="LSST",
        )

        fields = self._extract_supernnova_fields(raw, survey="LSST",
                                                  expert_key="pittgoogle/supernnova_lsst")
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey="LSST",
            source_endpoint=source_endpoint,
            request_params=request_params,
            primary_object_id=object_id,
            requested_object_id=object_id,
            primary_identifier_kind="lsst_dia_object_id",
            requested_identifier_kind="lsst_dia_object_id",
            fields=fields,
            events=fields,
            availability=not fixture_used and bool(fields),
            fixture_used=fixture_used,
        )

    # ------------------------------------------------------------------ #
    # ZTF path                                                             #
    # ------------------------------------------------------------------ #

    def _fetch_ztf(self, object_id: str) -> BrokerOutput:
        fixture_path = _FIXTURE_DIR / f"{object_id}_supernnova.json"
        source_endpoint = _ZTF_SUPERNNOVA_TABLE
        request_params = {"objectId": object_id}

        raw, fixture_used = self._query_or_fixture(
            sql=f"SELECT * FROM {_ZTF_SUPERNNOVA_TABLE} WHERE objectId = '{object_id}' ORDER BY jd LIMIT 500",
            fixture_path=fixture_path,
            survey="ZTF",
        )

        fields = self._extract_supernnova_fields(raw, survey="ZTF",
                                                  expert_key="pittgoogle/supernnova_ztf")
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw,
            semantic_type=self.semantic_type,
            survey="ZTF",
            source_endpoint=source_endpoint,
            request_params=request_params,
            primary_object_id=object_id,
            requested_object_id=object_id,
            primary_identifier_kind="ztf_object_id",
            requested_identifier_kind="ztf_object_id",
            fields=fields,
            events=fields,
            availability=not fixture_used and bool(fields),
            fixture_used=fixture_used,
        )

    # ------------------------------------------------------------------ #
    # BigQuery + fixture helpers                                           #
    # ------------------------------------------------------------------ #

    def _query_or_fixture(
        self,
        sql: str,
        fixture_path: Path,
        survey: str,
    ) -> tuple[dict[str, Any], bool]:
        """Run a BigQuery SQL query; fall back to fixture on any error.

        Returns (raw_payload_dict, fixture_used_bool).
        """
        if not _PITTGOOGLE_AVAILABLE or not self._has_credentials():
            if fixture_path.exists():
                return self.load_fixture(fixture_path), True
            return {"rows": [], "error": "no_credentials"}, True

        try:
            client = pittgoogle.bigquery.Client()
            df = client.query(sql)
            # Use pandas JSON serialisation — handles Timestamp/numpy types.
            import json as _json
            rows = _json.loads(df.to_json(orient="records"))
            raw = {"rows": rows, "survey": survey}
            self.save_fixture(raw, fixture_path)
            return raw, False
        except Exception as exc:
            if fixture_path.exists():
                return self.load_fixture(fixture_path), True
            return {"rows": [], "error": str(exc)}, True

    # ------------------------------------------------------------------ #
    # Field extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_supernnova_fields(
        self,
        raw: dict[str, Any],
        survey: str,
        expert_key: str,
    ) -> list[dict[str, Any]]:
        """Parse BigQuery rows into per-detection field dicts.

        Handles both ZTF (``SuperNNova_prob_class0``) and LSST column names.
        """
        rows = raw.get("rows", [])
        if not rows:
            return []

        # Detect column names from the first row
        first = rows[0]
        ia_col = _first_match(first, _SNN_IA_PROB_CANDIDATES)
        nonia_col = _first_match(first, _SNN_NONIA_PROB_CANDIDATES)
        pred_col = _first_match(first, _SNN_PRED_CLASS_CANDIDATES)
        ts_col = _first_match(first, _TIMESTAMP_CANDIDATES)

        if ia_col is None:
            # No recognisable SuperNNova probability column — not yet populated
            return []

        fields = []
        for idx, row in enumerate(rows):
            ia_raw = row.get(ia_col)
            if ia_raw is None:
                continue
            p_ia = float(ia_raw)
            nonia_raw = row.get(nonia_col) if nonia_col else None
            p_nonia = float(nonia_raw) if nonia_raw is not None else None

            # Resolve timestamp
            ts_raw = row.get(ts_col) if ts_col else None
            jd: float | None = None
            if ts_raw is not None:
                ts_f = float(ts_raw)
                # Convert MJD → JD if value looks like MJD (< 2400000)
                jd = ts_f + 2400000.5 if ts_f < 2400000.5 else ts_f

            # Resolve object / alert identifiers
            obj_id_col = _first_match(row, _OBJECT_ID_CANDIDATES.get(survey.lower(), []))
            alert_id_raw = row.get("candid") or row.get("diaSourceId") or row.get("sourceid")

            fields.append({
                "field": "supernnova_prob_class0",
                "raw_label_or_score": p_ia,
                "semantic_type": "probability",
                "canonical_projection": p_ia,
                "canonical_complement": p_nonia,  # P(non-Ia) if available
                "classifier": "pittgoogle_supernnova",
                "classifier_version": row.get("supernnova_version") or row.get("model_version"),
                "expert_key": expert_key,
                "event_scope": "alert",
                # Per-alert rows in BigQuery: mark as exact_alert
                "temporal_exactness": "exact_alert",
                "alert_id": int(alert_id_raw) if alert_id_raw is not None else None,
                "n_det": idx + 1,  # approximate — rows ordered by timestamp
                "alert_jd": jd,
                "event_time_jd": jd,
            })

        return fields

    # ------------------------------------------------------------------ #
    # Auth helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _has_credentials() -> bool:
        """Return True if GCP credentials are likely configured."""
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            return True
        # Application Default Credentials (gcloud auth login)
        adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if adc_path.exists():
            return True
        return False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _first_match(row: dict[str, Any], candidates: list[str]) -> str | None:
    """Return the first candidate key that exists in *row*, or None."""
    for c in candidates:
        if c in row:
            return c
    return None
