"""Rubin Science Platform (RSP) TAP client for DP1 data access.

Thin wrapper around `pyvo.dal.TAPService` with Gafaelfawr Bearer auth,
batched diaObjectId→DiaSource pulling, and a few ADQL helpers tuned for
USDF. Not a `BrokerAdapter` — DP1 ingest is bulk TAP, not per-object
classifier fetch.

Verified access 2026-04-24 at https://data.lsst.cloud/api/tap
(see docs/execplan_v6_dp1_validation.md §2c).

Usage
-----
    from debass_meta.access.rubin_rsp import RSPClient

    client = RSPClient()  # auto-loads RSP_TOKEN from env or .env
    # Pool query (93K objects with n>=5)
    pool = client.pool_diaobjects(min_n_det=5, limit=10000)
    # Batched DiaSource pull for a set of diaObjectIds
    lcs = client.fetch_diasources(pool["diaObjectId"].tolist(), batch_size=500)

DP1 column gotchas (measured):
  - RA/Dec: `ra`, `dec` (NOT `decl`)
  - DiaSource time: `midpointMjdTai`
  - Band: single-char string `band` in {u,g,r,i,z,y}
  - Flux: `psfFlux`, `psfFluxErr` in nanojansky (DIA differential; can be negative)
  - Reliability: `reliability` (commissioning-tuning, 85% of rows <0.1)
  - ADQL dialect: no CAST, no arithmetic inside GROUP BY.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

_DEFAULT_TAP_URL = "https://data.lsst.cloud/api/tap"
_DEFAULT_REPO_ENV = "DEBASS_ROOT"


def _find_repo_root() -> Path:
    """Resolve the DEBASS repo root from env or this module's path."""
    env = os.environ.get(_DEFAULT_REPO_ENV)
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def load_rsp_token(repo_root: Path | None = None) -> str:
    """Read the RSP bearer token from $RSP_TOKEN or .env."""
    env_tok = os.environ.get("RSP_TOKEN")
    if env_tok:
        return env_tok
    root = repo_root or _find_repo_root()
    env_path = root / ".env"
    if not env_path.exists():
        raise RuntimeError(
            f"RSP_TOKEN not set and {env_path} does not exist — "
            "save your RSP token first"
        )
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if s.startswith("RSP_TOKEN="):
            return s.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(f"RSP_TOKEN= line missing from {env_path}")


class RSPClient:
    """Minimal USDF TAP client with DP1 helpers.

    One TAP service is constructed per client. Thread-safe for ADQL SELECTs
    (pyvo uses a requests.Session internally), but not for mutation operations.
    """

    def __init__(
        self,
        token: str | None = None,
        tap_url: str = _DEFAULT_TAP_URL,
        repo_root: Path | None = None,
    ) -> None:
        import pyvo
        import requests

        self.tap_url = tap_url
        self.token = token or load_rsp_token(repo_root)
        sess = requests.Session()
        sess.headers["Authorization"] = f"Bearer {self.token}"
        self._svc = pyvo.dal.TAPService(tap_url, session=sess)

    def query(self, adql: str) -> pd.DataFrame:
        """Run an ADQL SELECT and return a pandas DataFrame."""
        return self._svc.search(adql).to_table().to_pandas()

    def count(self, where: str = "", table: str = "dp1.DiaObject") -> int:
        """SELECT COUNT(*) with optional WHERE clause."""
        where_clause = f" WHERE {where}" if where else ""
        r = self.query(f"SELECT COUNT(*) AS n FROM {table}{where_clause}")
        return int(r["n"].iloc[0])

    # ------------------------------------------------------------------ #
    # DP1-specific helpers                                                #
    # ------------------------------------------------------------------ #

    def pool_diaobjects(
        self,
        min_n_det: int = 5,
        limit: int | None = None,
        ra_range: tuple[float, float] | None = None,
        dec_range: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        """Pull a DP1 DiaObject sample.

        Parameters
        ----------
        min_n_det : int
            Minimum nDiaSources cut. The v6 plan uses n>=5 (93,336 objects).
        limit : int or None
            TOP N cap. Use None for full pool; note that single queries above
            ~60K time out — use batched/field queries instead.
        ra_range, dec_range : optional float tuples
            Pre-filter to a sky box.
        """
        where = [f"nDiaSources >= {int(min_n_det)}"]
        if ra_range is not None:
            where.append(f"ra BETWEEN {ra_range[0]} AND {ra_range[1]}")
        if dec_range is not None:
            where.append(f"dec BETWEEN {dec_range[0]} AND {dec_range[1]}")
        cap = f"TOP {int(limit)} " if limit else ""
        adql = (
            f"SELECT {cap}diaObjectId, ra, dec, nDiaSources "
            f"FROM dp1.DiaObject WHERE {' AND '.join(where)}"
        )
        return self.query(adql)

    def fetch_diasources(
        self,
        dia_object_ids: Iterable[int],
        batch_size: int = 500,
        min_reliability: float | None = None,
        columns: tuple[str, ...] = (
            "diaObjectId", "diaSourceId", "band",
            "midpointMjdTai", "psfFlux", "psfFluxErr",
            "reliability", "snr", "ra", "dec",
        ),
        progress_every: int = 20,
    ) -> pd.DataFrame:
        """Batched DiaSource pull via `diaObjectId IN (...)`.

        Per v6 plan §2c: 500-ID batches return ~6.4K rows in one query
        (median 13 detections/obj). Use batch_size=500 unless you have a reason.
        """
        ids = [int(x) for x in dia_object_ids if x is not None]
        if not ids:
            return pd.DataFrame(columns=list(columns))
        col_sql = ", ".join(columns)
        filters = []
        if min_reliability is not None:
            filters.append(f"reliability >= {float(min_reliability)}")

        frames: list[pd.DataFrame] = []
        n_batches = (len(ids) + batch_size - 1) // batch_size
        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i + batch_size]
            id_sql = ",".join(str(x) for x in chunk)
            where = [f"diaObjectId IN ({id_sql})"] + filters
            adql = (
                f"SELECT {col_sql} FROM dp1.DiaSource "
                f"WHERE {' AND '.join(where)}"
            )
            df = self.query(adql)
            frames.append(df)
            batch_idx = i // batch_size + 1
            if progress_every and batch_idx % progress_every == 0:
                total_rows = sum(len(f) for f in frames)
                print(
                    f"  TAP batch {batch_idx}/{n_batches}: "
                    f"{total_rows:,} rows cumulative",
                    flush=True,
                )
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=list(columns))

    def cone_search_diaobject(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float = 3.0,
    ) -> pd.DataFrame:
        """Cone search dp1.DiaObject around (ra, dec)."""
        r_deg = radius_arcsec / 3600.0
        adql = (
            "SELECT diaObjectId, ra, dec, nDiaSources FROM dp1.DiaObject "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {r_deg})) = 1"
        )
        return self.query(adql)


def normalize_diasource_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a TAP DiaSource row dict into the DEBASS detection schema.

    Passes through the keys `detection.normalize_detection()` recognizes as
    LSST. This keeps `midpointMjdTai`, `psfFlux`, `psfFluxErr`, `band`,
    `reliability` intact so the normalizer does the flux→mag conversion.
    """
    return dict(row)


def lightcurve_from_diasource_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build a list-of-dict lightcurve from a DiaSource DataFrame.

    Input: DataFrame with `band, midpointMjdTai, psfFlux, psfFluxErr,
    reliability` at minimum (other columns pass through as provenance).
    Output: list of dicts sorted by time, ready for
    `features.lightcurve.extract_features` after normalization.
    """
    if len(df) == 0:
        return []
    ordered = df.sort_values("midpointMjdTai")
    return [row.to_dict() for _, row in ordered.iterrows()]
