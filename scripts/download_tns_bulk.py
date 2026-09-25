#!/usr/bin/env python3
"""Download the TNS bulk public-objects catalog for local crossmatching.

Two output paths:

  * Legacy CSV (unchanged): ``--output-dir data`` writes
    ``data/tns_public_objects.csv[.zip]``.
  * v11 parquet master (new): ``--out data/truth/tns_public.parquet`` with a
    daily-diff **upsert** refresh (``--mode {auto,full,diff}``).

The master export has no ``Content-Length`` header, so byte-range resume is
impossible (fusion_v11 spec §1). Refresh = full master re-download OR a small
dated diff ``tns_public_objects_YYYYMMDD.csv.zip`` (only that day's modified
rows) upserted keyed ``objid`` with the newest ``lastmodified`` winning.

Run on SCC (or locally) where TNS credentials are configured. CSV parsing is
delegated to ``crossmatch_tns._load_tns_bulk_csv`` (handles the leading
timestamp line via ``skiprows`` and the ``name``→``objname`` rename).
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

# Load .env from repo root if present (silent if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

from debass_meta.access.tns import (
    TNSCredentials,
    fetch_tns_bulk,
    load_tns_credentials,
)


# ------------------------------------------------------------------ #
# Legacy CSV download (kept byte-compatible with prior callers)       #
# ------------------------------------------------------------------ #


def download_tns_bulk_csv(output_dir: Path) -> Path:
    """Download and extract TNS public objects CSV (legacy behaviour)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    creds = load_tns_credentials()
    url = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"

    zip_path = output_dir / "tns_public_objects.csv.zip"
    csv_path = output_dir / "tns_public_objects.csv"

    print(f"Downloading TNS bulk CSV from {url}")
    headers = {"User-Agent": creds.user_agent}
    r = requests.get(url, headers=headers, timeout=300)
    r.raise_for_status()

    print(f"Writing {len(r.content)} bytes to {zip_path}")
    zip_path.write_bytes(r.content)

    print(f"Extracting to {csv_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    print(f"OK downloaded {csv_path}")
    return csv_path


# ------------------------------------------------------------------ #
# v11 parquet master + daily-diff upsert                              #
# ------------------------------------------------------------------ #


def _extract_bulk_csv(zip_bytes: bytes, dest_csv: Path) -> Path:
    """Extract the single CSV member of a TNS bulk ZIP to ``dest_csv``."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"TNS bulk ZIP has no .csv member (got {zf.namelist()})")
        dest_csv.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(members[0]) as src, open(dest_csv, "wb") as out:
            out.write(src.read())
    return dest_csv


def bulk_zip_to_df(zip_bytes: bytes, *, scratch_csv: Path):
    """Parse a TNS bulk ZIP into a DataFrame via ``_load_tns_bulk_csv``.

    Parsing is reused (by import) from ``scripts/crossmatch_tns.py`` so the
    ``skiprows`` timestamp-header handling stays in one place.
    """
    from scripts.crossmatch_tns import _load_tns_bulk_csv

    csv_path = _extract_bulk_csv(zip_bytes, scratch_csv)
    return _load_tns_bulk_csv(csv_path)


def upsert_tns_bulk(master, diff):
    """Upsert a daily diff into the master TNS table.

    Keyed on ``objid``; the row with the newest ``lastmodified`` wins. Pure
    function (no I/O) — unit-tested with synthetic frames.
    """
    import pandas as pd

    frames = [f for f in (master, diff) if f is not None and len(f) > 0]
    if not frames:
        return master if master is not None else diff
    combined = pd.concat(frames, ignore_index=True)
    if "objid" not in combined.columns:
        raise ValueError("TNS bulk frame missing 'objid' column; cannot upsert")
    if "lastmodified" in combined.columns:
        # Stable sort by lastmodified so keep='last' retains the newest row.
        order = pd.to_datetime(combined["lastmodified"], errors="coerce")
        combined = combined.assign(_ts=order).sort_values(
            "_ts", kind="stable", na_position="first"
        )
        combined = combined.drop_duplicates(subset="objid", keep="last")
        combined = combined.drop(columns="_ts")
    else:
        combined = combined.drop_duplicates(subset="objid", keep="last")
    return combined.reset_index(drop=True)


def _yesterday_utc() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")


def refresh_tns_parquet(
    *,
    out_path: Path,
    mode: str = "auto",
    diff_date: str | None = None,
    creds: TNSCredentials | None = None,
) -> Path:
    """Download/refresh the TNS master parquet.

    * ``full``  — download master, overwrite parquet.
    * ``diff``  — download ``diff_date`` (default: yesterday UTC), upsert into
      the existing parquet.
    * ``auto``  — ``full`` when parquet is absent, else ``diff``.
    """
    import pandas as pd

    creds = creds or load_tns_credentials()
    out_path = Path(out_path)
    scratch_csv = out_path.parent / "_tns_bulk_scratch.csv"

    effective = mode
    if mode == "auto":
        effective = "diff" if out_path.exists() else "full"

    if effective == "full":
        print("TNS refresh: downloading full master ...")
        zip_bytes = fetch_tns_bulk(creds, date=None)
        df = bulk_zip_to_df(zip_bytes, scratch_csv=scratch_csv)
        print(f"  master rows: {len(df)}")
    elif effective == "diff":
        date = diff_date or _yesterday_utc()
        print(f"TNS refresh: downloading daily diff {date} ...")
        if not out_path.exists():
            raise SystemExit(
                f"--mode diff needs an existing master at {out_path}; run --mode full first"
            )
        master = pd.read_parquet(out_path)
        try:
            zip_bytes = fetch_tns_bulk(creds, date=date)
        except requests.HTTPError as exc:
            # Current-day diff 404s until the day closes (spec §1).
            raise SystemExit(
                f"TNS diff {date} unavailable ({exc}). The current UTC day 404s "
                f"until it closes; pass an earlier --diff-date or use --mode full."
            ) from exc
        diff = bulk_zip_to_df(zip_bytes, scratch_csv=scratch_csv)
        print(f"  diff rows: {len(diff)}; master rows before: {len(master)}")
        df = upsert_tns_bulk(master, diff)
        print(f"  master rows after upsert: {len(df)}")
    else:
        raise ValueError(f"unknown mode {mode!r} (expected auto|full|diff)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    if scratch_csv.exists():
        scratch_csv.unlink()
    typed = int(df["type"].notna().sum()) if "type" in df.columns else 0
    print(f"OK wrote {len(df)} TNS rows ({typed} typed) -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Legacy CSV output dir (writes tns_public_objects.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="v11 parquet master path (e.g. data/truth/tns_public.parquet)")
    parser.add_argument("--mode", choices=("auto", "full", "diff"), default="auto",
                        help="Refresh mode for --out (default: auto)")
    parser.add_argument("--diff-date", default=None,
                        help="YYYYMMDD diff date for --mode diff (default: yesterday UTC)")
    args = parser.parse_args()

    if args.out is not None:
        refresh_tns_parquet(out_path=args.out, mode=args.mode, diff_date=args.diff_date)
        return

    # Legacy path
    output_dir = args.output_dir or Path("data")
    csv_path = download_tns_bulk_csv(output_dir)
    print(f"\nTo use: python scripts/crossmatch_tns.py --bulk-csv {csv_path}")


if __name__ == "__main__":
    main()
