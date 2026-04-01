"""TNS (Transient Name Server) access adapter.

Adapted from decat_pointings/tz_debass_auto/debass_tns/tns_client.py.
Designed for bulk crossmatch: concurrent fetching with rate-limit backoff.

Credentials are loaded exclusively from environment variables:
  TNS_API_KEY       — required
  TNS_TNS_ID        — required (your bot ID)
  TNS_MARKER_TYPE   — default "bot"
  TNS_MARKER_NAME   — required (your registered bot name)

Security:
  - No hardcoded keys or personal identifiers.
  - __repr__ on credential objects redacts secrets.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Literal

import requests

# Load .env from repo root if python-dotenv is available (no-op otherwise)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

# ------------------------------------------------------------------ #
# Credentials                                                         #
# ------------------------------------------------------------------ #


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


@dataclass(frozen=True)
class TNSCredentials:
    api_key: str
    tns_id: str
    marker_type: str
    marker_name: str
    sandbox: bool = False

    @property
    def user_agent(self) -> str:
        marker = {
            "tns_id": str(self.tns_id),
            "type": str(self.marker_type),
            "name": str(self.marker_name),
        }
        return f"tns_marker{json.dumps(marker, separators=(',', ':'))}"

    def __repr__(self) -> str:
        return (
            f"TNSCredentials(api_key='***', tns_id={self.tns_id!r}, "
            f"marker_name={self.marker_name!r}, sandbox={self.sandbox})"
        )


def load_tns_credentials(*, sandbox: bool = False) -> TNSCredentials:
    api_key = _env("TNS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing TNS_API_KEY. Set env vars: TNS_API_KEY, TNS_TNS_ID, "
            "TNS_MARKER_TYPE, TNS_MARKER_NAME (see .env.example)."
        )
    tns_id = _env("TNS_TNS_ID")
    if not tns_id:
        raise RuntimeError(
            "Missing TNS_TNS_ID. Set to your TNS user or bot ID "
            "(visible at https://www.wis-tns.org/user)."
        )
    marker_name = _env("TNS_MARKER_NAME")
    if not marker_name:
        raise RuntimeError(
            "Missing TNS_MARKER_NAME. Set to your TNS username or registered bot name."
        )
    marker_type = _env("TNS_MARKER_TYPE", "user")
    return TNSCredentials(
        api_key=api_key,
        tns_id=tns_id,
        marker_type=marker_type,
        marker_name=marker_name,
        sandbox=sandbox,
    )


# ------------------------------------------------------------------ #
# TNS client — single-object + bulk                                   #
# ------------------------------------------------------------------ #

TNS_TERNARY_MAP: dict[str, str] = {
    # SN Ia variants → "snia"
    "SN Ia": "snia",
    "SN Ia-91T-like": "snia",
    "SN Ia-91bg-like": "snia",
    "SN Ia-CSM": "snia",
    "SN Ia-pec": "snia",
    "SN Ia-SC": "snia",
    "SN Iax[02cx-like]": "snia",
    # Core-collapse / thermonuclear non-Ia → "nonIa_snlike"
    "SN Ib": "nonIa_snlike",
    "SN Ib-pec": "nonIa_snlike",
    "SN Ib/c": "nonIa_snlike",
    "SN Ic": "nonIa_snlike",
    "SN Ic-BL": "nonIa_snlike",
    "SN Ic-pec": "nonIa_snlike",
    "SN II": "nonIa_snlike",
    "SN IIb": "nonIa_snlike",
    "SN IIP": "nonIa_snlike",
    "SN IIL": "nonIa_snlike",
    "SN IIn": "nonIa_snlike",
    "SN IIn-pec": "nonIa_snlike",
    "SN II-pec": "nonIa_snlike",
    "SLSN-I": "nonIa_snlike",
    "SLSN-II": "nonIa_snlike",
    "SLSN-R": "nonIa_snlike",
    "SN": "nonIa_snlike",
    # Everything else → "other"
    "TDE": "other",
    "AGN": "other",
    "CV": "other",
    "LBV": "other",
    "Nova": "other",
    "Varstar": "other",
    "Galaxy": "other",
    "QSO": "other",
    "ILRT": "other",
    "Kilonova": "other",
    "FRB": "other",
    "Impostor-SN": "other",
    "Other": "other",
}


class TNSError(RuntimeError):
    """Raised for TNS protocol/transport failures."""


@dataclass(frozen=True)
class TNSResult:
    """Result from a single TNS object query."""
    object_id: str          # ZTF object_id we queried for
    tns_name: str | None    # e.g. "2023abc"
    tns_prefix: str | None  # "SN", "AT"
    type_name: str | None   # e.g. "SN Ia"
    has_spectra: bool
    redshift: float | None
    ra: float | None
    dec: float | None
    discovery_date: str | None
    raw: dict[str, Any]


class TNSClient:
    """Rate-limit-aware TNS API client for bulk crossmatch."""

    def __init__(
        self,
        creds: TNSCredentials,
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ):
        self._creds = creds
        self._timeout_s = timeout_s
        self._max_retries = max_retries

        host = "sandbox.wis-tns.org" if creds.sandbox else "www.wis-tns.org"
        self._search_url = f"https://{host}/api/get/search"
        self._object_url = f"https://{host}/api/get/object"
        self._session = requests.Session()
        self._session.headers["User-Agent"] = creds.user_agent

        # Rate limit state — shared across threads via GIL-safe ops
        self._rate_wait_until: float = 0.0

    def _wait_for_rate_limit(self) -> None:
        """Block until any globally observed rate limit has expired."""
        now = time.monotonic()
        if self._rate_wait_until > now:
            time.sleep(self._rate_wait_until - now)

    def _handle_rate_limit(self, response: requests.Response) -> float:
        """Parse rate limit headers and return seconds to wait."""
        reset_hdr = response.headers.get("x-rate-limit-reset")
        if reset_hdr:
            try:
                v = int(float(reset_hdr))
                now_epoch = int(time.time())
                wait = (v - now_epoch) if v > (now_epoch + 5) else v
                return max(1.0, float(wait))
            except (ValueError, TypeError):
                pass
        return 60.0

    def _post(
        self,
        url: str,
        data_obj: dict[str, Any],
        *,
        allow_no_results: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "api_key": self._creds.api_key,
            "data": json.dumps(data_obj, ensure_ascii=False),
        }
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            self._wait_for_rate_limit()
            try:
                r = self._session.post(url, data=payload, timeout=self._timeout_s)
                try:
                    raw = r.json()
                except Exception as e:
                    raise TNSError(f"Failed to parse JSON from TNS. HTTP {r.status_code}.") from e

                id_code = int(raw.get("id_code", -1))
                id_message = str(raw.get("id_message", ""))
                data = raw.get("data", None)

                if r.status_code == 429 or id_code == 429:
                    wait_s = self._handle_rate_limit(r)
                    self._rate_wait_until = time.monotonic() + wait_s + 1
                    time.sleep(wait_s + 1)
                    continue

                if id_code == 110 and allow_no_results:
                    return raw
                if id_code != 200:
                    raise TNSError(f"TNS error id_code={id_code} message={id_message}")
                return raw
            except TNSError:
                raise
            except Exception as e:
                last_exc = e
                if attempt < self._max_retries:
                    time.sleep(1.0 * attempt)
                    continue
                raise TNSError(f"TNS request failed after {self._max_retries} attempts") from e
        raise TNSError("TNS request failed") from last_exc

    # ------------------------------------------------------------------ #
    # Single-object operations                                             #
    # ------------------------------------------------------------------ #

    def search_by_name(self, name: str) -> list[dict[str, Any]]:
        """Search TNS for an object by internal name (e.g. ZTF ID)."""
        data_obj = {"internal_name": name}
        raw = self._post(self._search_url, data_obj, allow_no_results=True)
        if int(raw.get("id_code", -1)) == 110:
            return []
        data = raw.get("data", None)
        if isinstance(data, dict):
            reply = data.get("reply", [])
            return reply if isinstance(reply, list) else []
        if isinstance(data, list):
            return data
        return []

    def get_object(
        self,
        objname: str,
        *,
        spectra: bool = True,
        photometry: bool = False,
    ) -> dict[str, Any]:
        """Fetch full object detail from TNS."""
        data_obj: dict[str, Any] = {"objname": objname}
        if spectra:
            data_obj["spectra"] = "1"
        if photometry:
            data_obj["photometry"] = "1"
        raw = self._post(self._object_url, data_obj)
        data = raw.get("data", {})
        if isinstance(data, dict):
            reply = data.get("reply", data)
            return reply if isinstance(reply, dict) else data
        return data if isinstance(data, dict) else {}

    def crossmatch_object(self, object_id: str) -> TNSResult:
        """Full crossmatch pipeline for one ZTF object ID.

        1. Search TNS by internal name
        2. If found, fetch object detail (with spectra flag)
        3. Return structured TNSResult
        """
        hits = self.search_by_name(object_id)
        if not hits:
            return TNSResult(
                object_id=object_id,
                tns_name=None,
                tns_prefix=None,
                type_name=None,
                has_spectra=False,
                redshift=None,
                ra=None,
                dec=None,
                discovery_date=None,
                raw={},
            )

        # Take first match
        hit = hits[0] if isinstance(hits[0], dict) else {}
        tns_name = str(hit.get("objname") or hit.get("name") or "").strip()
        tns_prefix = str(hit.get("prefix") or "").strip() or None

        if not tns_name:
            return TNSResult(
                object_id=object_id,
                tns_name=None,
                tns_prefix=tns_prefix,
                type_name=None,
                has_spectra=False,
                redshift=None,
                ra=None,
                dec=None,
                discovery_date=None,
                raw=hit,
            )

        detail = self.get_object(tns_name, spectra=True)
        type_name = _extract_type(detail)
        has_spectra = _has_spectra(detail)
        redshift = _extract_redshift(detail)
        ra = _safe_float(detail.get("ra") or detail.get("radeg"))
        dec = _safe_float(detail.get("dec") or detail.get("decdeg"))
        discovery_date = str(detail.get("discoverydate") or "") or None

        return TNSResult(
            object_id=object_id,
            tns_name=tns_name,
            tns_prefix=tns_prefix,
            type_name=type_name,
            has_spectra=has_spectra,
            redshift=redshift,
            ra=ra,
            dec=dec,
            discovery_date=discovery_date,
            raw=detail,
        )

    # ------------------------------------------------------------------ #
    # Bulk crossmatch — concurrent with shared rate limiter               #
    # ------------------------------------------------------------------ #

    def crossmatch_bulk(
        self,
        object_ids: list[str],
        *,
        max_workers: int = 4,
        progress: bool = True,
    ) -> list[TNSResult]:
        """Crossmatch many objects against TNS concurrently.

        Uses ThreadPoolExecutor with a shared rate-limit backoff.
        The TNS API rate limit is global per key, so all threads
        respect the same _rate_wait_until timestamp.
        """
        results: dict[str, TNSResult] = {}
        total = len(object_ids)

        if progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total, desc="TNS crossmatch", unit="obj")
            except ImportError:
                pbar = None
        else:
            pbar = None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._safe_crossmatch, oid): oid
                for oid in object_ids
            }
            for future in as_completed(futures):
                oid = futures[future]
                results[oid] = future.result()
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Preserve input order
        return [results[oid] for oid in object_ids]

    def _safe_crossmatch(self, object_id: str) -> TNSResult:
        """Crossmatch with error handling — never raises."""
        try:
            return self.crossmatch_object(object_id)
        except Exception as exc:
            import sys
            print(f"WARNING: TNS crossmatch failed for {object_id}: {exc}", file=sys.stderr)
            return TNSResult(
                object_id=object_id,
                tns_name=None,
                tns_prefix=None,
                type_name=None,
                has_spectra=False,
                redshift=None,
                ra=None,
                dec=None,
                discovery_date=None,
                raw={"error": "crossmatch_failed", "detail": str(exc)},
            )


# ------------------------------------------------------------------ #
# TNS response parsing helpers                                        #
# ------------------------------------------------------------------ #


def _extract_type(detail: dict[str, Any]) -> str | None:
    """Extract the most specific type classification from TNS detail."""
    # object_type.name is the main classification
    obj_type = detail.get("object_type")
    if isinstance(obj_type, dict):
        name = str(obj_type.get("name") or "").strip()
        if name:
            return name
    # Fallback: type string at top level
    for key in ("type", "object_type"):
        val = detail.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _has_spectra(detail: dict[str, Any]) -> bool:
    """Check if the TNS object has spectroscopic classification."""
    spectra = detail.get("spectra")
    if isinstance(spectra, list) and len(spectra) > 0:
        return True
    # Some responses use nested structure
    if isinstance(spectra, dict) and spectra:
        return True
    # Check classification spectra specifically
    class_spectra = detail.get("classification_spectra")
    if isinstance(class_spectra, list) and len(class_spectra) > 0:
        return True
    return False


def _extract_redshift(detail: dict[str, Any]) -> float | None:
    """Extract redshift from TNS detail, preferring host redshift."""
    for key in ("redshift", "host_redshift"):
        val = _safe_float(detail.get(key))
        if val is not None:
            return val
    return None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None


def map_tns_type_to_ternary(type_name: str | None) -> str | None:
    """Map a TNS type string to the DEBASS ternary label.

    Returns None if the type is unknown or unmappable.
    """
    if not type_name:
        return None
    cleaned = type_name.strip()
    if cleaned in TNS_TERNARY_MAP:
        return TNS_TERNARY_MAP[cleaned]
    # Fuzzy fallback: if it starts with "SN Ia", it's snia
    if cleaned.startswith("SN Ia"):
        return "snia"
    # Any SN-like prefix
    if cleaned.startswith("SN ") or cleaned.startswith("SLSN"):
        return "nonIa_snlike"
    return None
