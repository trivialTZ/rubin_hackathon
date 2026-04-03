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
# TNS type → ternary mapping                                          #
# ------------------------------------------------------------------ #
#
# Science notes on contested types:
#   - SN Ia-CSM: thermonuclear + circumstellar medium. Some debate
#     whether these are true Ia or IIn mimics, but TNS consensus
#     (and most published analyses) treats them as Ia-family.
#   - SN Iax/02cx-like: low-luminosity thermonuclear. Definitely
#     NOT core-collapse. Keep as snia for follow-up purposes.
#   - "SN" (generic, no subtype): observer saw SN features but
#     couldn't determine type. Could be Ia OR CC — ambiguous.
#     Mapped to nonIa_snlike as conservative prior (CCSNe are
#     ~3x more common), BUT flagged in TNS_AMBIGUOUS_TYPES so
#     quality assignment can downgrade from spectroscopic.

TNS_TERNARY_MAP: dict[str, str] = {
    # ── SN Ia variants → "snia" ──────────────────────────────
    "SN Ia": "snia",
    "SN Ia-91T-like": "snia",     # overluminous Ia
    "SN Ia-91bg-like": "snia",    # subluminous Ia
    "SN Ia-CSM": "snia",          # thermonuclear + CSM (see note)
    "SN Ia-pec": "snia",          # peculiar Ia
    "SN Ia-SC": "snia",           # super-Chandrasekhar
    "SN Iax[02cx-like]": "snia",  # low-luminosity thermonuclear
    # ── Core-collapse / non-Ia SN → "nonIa_snlike" ──────────
    "SN Ib": "nonIa_snlike",
    "SN Ib-pec": "nonIa_snlike",
    "SN Ib/c": "nonIa_snlike",
    "SN Ic": "nonIa_snlike",
    "SN Ic-BL": "nonIa_snlike",   # broad-lined, GRB-associated
    "SN Ic-pec": "nonIa_snlike",
    "SN Ibn": "nonIa_snlike",     # He-rich CSM interaction
    "SN Icn": "nonIa_snlike",     # C/O-rich CSM interaction
    "SN II": "nonIa_snlike",
    "SN IIb": "nonIa_snlike",     # transitional H→He
    "SN IIP": "nonIa_snlike",     # plateau
    "SN IIL": "nonIa_snlike",     # linear decline
    "SN IIn": "nonIa_snlike",     # narrow lines (CSM)
    "SN IIn-pec": "nonIa_snlike",
    "SN II-pec": "nonIa_snlike",
    "SLSN-I": "nonIa_snlike",     # superluminous H-poor
    "SLSN-II": "nonIa_snlike",    # superluminous H-rich
    "SLSN-R": "nonIa_snlike",     # superluminous radioactive
    "SN": "nonIa_snlike",         # generic SN (AMBIGUOUS — see below)
    # ── Everything else → "other" ────────────────────────────
    "TDE": "other",               # tidal disruption event
    "AGN": "other",               # active galactic nucleus
    "CV": "other",                # cataclysmic variable
    "LBV": "other",               # luminous blue variable
    "Nova": "other",
    "Varstar": "other",           # generic variable star
    "Galaxy": "other",            # host galaxy (not transient)
    "QSO": "other",               # quasar
    "ILRT": "other",              # intermediate-luminosity red transient
    "Kilonova": "other",          # neutron star merger
    "FRB": "other",               # fast radio burst
    "Impostor-SN": "other",       # SN impostor (LBV eruption etc.)
    "Other": "other",
    # SN subtypes seen in real TNS data
    "SN Ib-Ca-rich": "nonIa_snlike",  # calcium-rich transient
    "SN I": "nonIa_snlike",       # stripped-envelope, subtype uncertain
    # TDE subtypes (real TNS classifications)
    "TDE-H": "other",             # TDE with hydrogen features
    "TDE-He": "other",            # TDE with helium features
    "TDE-H-He": "other",          # TDE with both H and He
    "TDE-featureless": "other",   # featureless TDE
    # Rare/emerging types
    "FBOT": "other",              # fast blue optical transient
    "LRN": "other",               # luminous red nova (stellar merger)
    "Afterglow": "other",         # GRB afterglow
    "M dwarf": "other",           # stellar flare
    "WR": "other",                # Wolf-Rayet star
    "Gap": "other",               # luminosity gap transient
    "PISN": "nonIa_snlike",       # pair-instability SN
    "Blazar": "other",            # blazar / BL Lac
    "Light-Echo": "other",        # light echo (not a transient)
    "NA/Unknown": "other",        # explicitly unknown
}

# Types where the ternary class is AMBIGUOUS even with a spectrum.
# "SN" means the observer saw SN features but couldn't determine Ia vs CC.
# These should NOT get "spectroscopic" quality — downgrade to "consensus"
# unless corroborated by broker agreement.
TNS_AMBIGUOUS_TYPES: set[str] = {"SN"}


# ------------------------------------------------------------------ #
# Source verification framework                                        #
# ------------------------------------------------------------------ #
#
# Recognized survey programs with documented spectroscopic follow-up
# pipelines. Classifications from these groups have well-characterized
# reduction and typing procedures.
#
# Groups NOT in this set may still produce correct classifications,
# but we lack the metadata to verify the pipeline programmatically.
# Their labels are assigned a lower quality tier unless corroborated
# by independent broker consensus.

TNS_CREDIBLE_GROUPS: set[str] = {
    # ── Major time-domain surveys (discovery + classification) ────
    "ZTF", "ZTF/CLU", "BTS",           # Zwicky Transient Facility
    "iPTF", "PTF",                      # (intermediate) Palomar Transient Factory
    "ATLAS",                            # Asteroid Terrestrial-impact Last Alert System
    "Pan-STARRS", "PS1", "PS2",        # Panoramic Survey Telescope
    "ASAS-SN",                          # All-Sky Automated Survey for Supernovae
    "GaiaAlerts",                       # ESA Gaia Science Alerts
    "MASTER",                           # Mobile Astronomical System of TElescope-Robots
    "OGLE",                             # Optical Gravitational Lensing Experiment
    "SkyMapper",                        # SkyMapper Southern Survey (ANU)
    "GOTO",                             # Gravitational-wave Optical Transient Observer
    "BlackGEM",                         # BlackGEM array (ESO/Radboud)
    "WFST",                             # Wide Field Survey Telescope (USTC)
    "Rubin",                            # Vera C. Rubin Observatory
    "LAST",                             # Large Array Survey Telescope
    "SDSS", "SDSS-II",                  # Sloan Digital Sky Survey
    "DES", "DES-SN", "DESGW",          # Dark Energy Survey
    "SNLS",                             # Supernova Legacy Survey
    # ── Spectroscopic classification programs ─────────────────────
    "SCAT", "UCSC",                     # Santa Cruz / UCSC
    "ePESSTO", "ePESSTO+", "PESSTO",   # Public ESO Spectroscopic Survey
    "YSE",                              # Young Supernova Experiment
    "DEBASS",                           # Discovery and Early Broadband Analysis of SNe
    "SNFactory", "SNf",                 # Nearby Supernova Factory
    "CSP",                              # Carnegie Supernova Project
    "LCO", "LCO Global SN Project",    # Las Cumbres Observatory
    "CfA",                              # Center for Astrophysics
    "Asiago",                           # Asiago classification program
    "GSP",                              # Global SN Project
    "SGLF",                             # Singapore Gravitational Lensing
    "SNe in ACT",                       # SNe in ACT survey
    "GSNST",                            # Global SN Science Team
    "DESIRT",                           # DESI Transients program
    # ── Alert brokers ─────────────────────────────────────────────
    "ALeRCE",                           # ALeRCE broker
    "Fink",                             # Fink broker
    "Lasair",                           # Lasair broker
    "AMPEL",                            # Alert Management Platform (AIP)
    "ANTARES",                          # ANTARES broker (NOIRLab)
    # ── Multi-messenger / GW follow-up ────────────────────────────
    "ENGRAVE",                          # EM counterpart to GW
    "GROWTH", "DECam-GROWTH",           # Global Relay of Observatories
    "GW-MMADS",                         # GW Multi-Messenger Astronomy
    # ── Radio transient surveys ───────────────────────────────────
    "CHIMEFRB",                         # CHIME Fast Radio Burst
    "CRAFT",                            # Commensal Real-time ASKAP Fast Transients
    "FRBCAT",                           # FRB Catalogue project
    # ── Observatory / telescope groups ────────────────────────────
    "SOAR",                             # SOAR telescope group
    "NOIR",                             # NOIRLab
    "GTC",                              # Gran Telescopio Canarias
    "NOT",                              # Nordic Optical Telescope
    "WHT",                              # William Herschel Telescope
    "Keck",                             # Keck Observatory
    "Gemini",                           # Gemini Observatory
    "VLT", "ESO",                       # Very Large Telescope / ESO
    "Magellan",                         # Magellan Telescopes
    "AAT",                              # Anglo-Australian Telescope
    "NTT",                              # New Technology Telescope
    "SALT",                             # Southern African Large Telescope
    "HET",                              # Hobby-Eberly Telescope
    "APO",                              # Apache Point Observatory
    "Lick",                             # Lick Observatory
    "MMT",                              # MMT Observatory
    "TCD", "GREAT",                     # Trinity College Dublin / GREAT survey
    # ── Instrument-named groups ───────────────────────────────────
    "FLOYDS",                           # FLOYDS spectrograph (LCO)
    "SNIFS",                            # SuperNova Integral Field Spectrograph
    # ── Specialized classification programs ───────────────────────
    "TDE",                              # TDE classification group
    "SNHunt",                           # Catalina Real-Time SN Hunt
    "BraTS",                            # Brazilian Transient Search
    "TNTS",                             # Tsinghua-NAOC Transient Survey
    "XOSS",                             # Xingming Observatory Sky Survey
    "DLT40",                            # Distance Less Than 40 Mpc Survey (UC Davis/Tartaglia)
    "FLEET",                            # Finding Luminous & Exotic Extragalactic Transients (CfA)
    "Supernova Alliance",               # Supernova Alliance (ANU/Swinburne)
    "YSSN",                             # Yonsei SN group
    "NUTS", "NUTS2",                    # NOT Unbiased Transient Survey
    "CBET",                             # IAU Central Bureau
}

# Recognized spectrographs — an instrument in this set provides
# strong provenance for the classification even if the group name
# is not in the registry (e.g. a PI-led program on a known telescope).
TNS_CREDIBLE_INSTRUMENTS: set[str] = {
    # Robotic classifiers
    "SEDM", "SPRAT",
    # Workhorse spectrographs on 2-4m telescopes
    "ALFOSC", "EFOSC2", "FLOYDS", "WiFeS", "SNIFS",
    "DBSP", "DIS", "Goodman", "GMOS",
    "Binospec", "LDSS3", "IMACS", "MODS",
    # Large telescope spectrographs (4-10m)
    "DEIMOS", "LRIS", "ESI",     # Keck
    "FORS2", "X-shooter",         # VLT
    "GMOS-N", "GMOS-S",           # Gemini
    "FIRE", "MagE", "LDSS-3",     # Magellan
    "RSS",                         # SALT
    "LRS2", "VIRUS",               # HET
    "Kastner", "Kast",             # Lick
}

# Build case-insensitive lookup sets at import time.
# TNS API returns inconsistent casing (e.g. "BINOSPEC" vs "Binospec").
_CREDIBLE_GROUPS_LOWER = {g.lower() for g in TNS_CREDIBLE_GROUPS}
_CREDIBLE_INSTRUMENTS_LOWER = {i.lower() for i in TNS_CREDIBLE_INSTRUMENTS}


@dataclass(frozen=True)
class SpectrumCredibility:
    """Credibility assessment of a TNS spectroscopic classification."""
    tier: str  # "professional", "likely_professional", "unverified"
    source_group: str | None
    instrument: str | None
    reason: str


def extract_classification_credibility(
    raw_detail: dict[str, Any],
) -> SpectrumCredibility:
    """Assess verification status of TNS classification from API response metadata.

    Checks the classification spectrum's source group and instrument
    against recognized survey programs and instruments.

    Returns
    -------
    SpectrumCredibility
        tier="professional" → recognized survey program, use as spectroscopic truth
        tier="likely_professional" → recognized instrument, group not in registry
        tier="unverified" → insufficient metadata to verify, lower quality tier
    """
    spectra = raw_detail.get("spectra") or raw_detail.get("classification_spectra")
    if not spectra:
        return SpectrumCredibility(
            tier="unverified",
            source_group=None,
            instrument=None,
            reason="no_spectra_in_response",
        )

    if isinstance(spectra, dict):
        spectra = list(spectra.values())
    if not isinstance(spectra, list) or len(spectra) == 0:
        return SpectrumCredibility(
            tier="unverified",
            source_group=None,
            instrument=None,
            reason="empty_spectra_list",
        )

    # Check ALL spectra — if ANY is from a credible source, trust it.
    # (Object may have multiple spectra from different groups.)
    best_tier = "unverified"
    best_group = None
    best_instrument = None
    best_reason = "no_recognized_source"

    for spec in spectra:
        if not isinstance(spec, dict):
            continue

        group = str(spec.get("source_group_name") or "").strip()
        inst_obj = spec.get("instrument")
        instrument = ""
        if isinstance(inst_obj, dict):
            instrument = str(inst_obj.get("name") or "").strip()
        elif isinstance(inst_obj, str):
            instrument = inst_obj.strip()

        # Check group against credible list (case-insensitive)
        if group and group.lower() in _CREDIBLE_GROUPS_LOWER:
            return SpectrumCredibility(
                tier="professional",
                source_group=group,
                instrument=instrument or None,
                reason=f"credible_group:{group}",
            )

        # Check instrument against credible list (case-insensitive)
        if instrument and instrument.lower() in _CREDIBLE_INSTRUMENTS_LOWER:
            if best_tier != "professional":
                best_tier = "likely_professional"
                best_group = group or None
                best_instrument = instrument
                best_reason = f"credible_instrument:{instrument}"

        # Track whatever we found
        if best_group is None and group:
            best_group = group
        if best_instrument is None and instrument:
            best_instrument = instrument

    return SpectrumCredibility(
        tier=best_tier,
        source_group=best_group,
        instrument=best_instrument,
        reason=best_reason,
    )


def is_credible_classification(
    raw_detail: dict[str, Any],
) -> bool:
    """Quick check: is the TNS classification from a recognized source?

    Returns True for "professional" or "likely_professional" tiers.
    """
    cred = extract_classification_credibility(raw_detail)
    return cred.tier in ("professional", "likely_professional")


# ------------------------------------------------------------------ #
# TNS client — single-object + bulk                                   #
# ------------------------------------------------------------------ #


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
    # Credibility metadata (populated from API responses, None for bulk CSV)
    source_group: str | None = None
    classification_instrument: str | None = None
    credibility_tier: str | None = None  # "professional"/"likely_professional"/"unverified"
    credibility_reason: str | None = None


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
        return self._parse_search_reply(
            self._post(self._search_url, data_obj, allow_no_results=True)
        )

    def search_by_position(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 3.0,
    ) -> list[dict[str, Any]]:
        """Search TNS by sky position (cone search).

        TNS ``/api/get/search`` accepts ``ra``, ``dec``, ``radius``, ``units``
        parameters.  Coordinates are decimal degrees; radius in arcseconds.
        Cone searches are heavier on the TNS side — respect rate limits.
        """
        data_obj = {
            "ra": str(ra_deg),
            "dec": str(dec_deg),
            "radius": str(radius_arcsec),
            "units": "arcsec",
        }
        return self._parse_search_reply(
            self._post(self._search_url, data_obj, allow_no_results=True)
        )

    @staticmethod
    def _parse_search_reply(raw: dict[str, Any]) -> list[dict[str, Any]]:
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

    def crossmatch_object(
        self,
        object_id: str,
        *,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        positional_radius_arcsec: float = 3.0,
    ) -> TNSResult:
        """Full crossmatch pipeline for one object.

        1. Search TNS by internal name (fast, exact)
        2. If no match and ``ra_deg``/``dec_deg`` are provided, try cone search
        3. If found, fetch object detail (with spectra flag)
        4. Return structured TNSResult with match method metadata
        """
        hits = self.search_by_name(object_id)
        match_method = "name"

        if not hits and ra_deg is not None and dec_deg is not None:
            hits = self.search_by_position(ra_deg, dec_deg, positional_radius_arcsec)
            match_method = "positional"

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
                raw={"match_method": match_method},
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

        # Assess classification credibility from spectrum metadata
        cred = extract_classification_credibility(detail)

        detail["match_method"] = match_method
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
            source_group=cred.source_group,
            classification_instrument=cred.instrument,
            credibility_tier=cred.tier,
            credibility_reason=cred.reason,
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


def is_ambiguous_type(type_name: str | None) -> bool:
    """Check if the TNS type is ambiguous (class uncertain even with spectrum).

    Generic "SN" means the observer saw SN features but couldn't distinguish
    Ia from core-collapse. This should not get full spectroscopic quality.
    """
    if not type_name:
        return True
    return type_name.strip() in TNS_AMBIGUOUS_TYPES
