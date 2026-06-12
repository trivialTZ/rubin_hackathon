#!/usr/bin/env python3
"""Probe the Rubin Science Platform TAP service.

Reads RSP_TOKEN from .env (or env var) and inventories visible schemas + tables.
Usage:
    python scripts/_exp_rsp_tap_probe.py                      # schema + tables
    python scripts/_exp_rsp_tap_probe.py --schema dp02_dc2_catalogs
    python scripts/_exp_rsp_tap_probe.py --url https://data-int.lsst.cloud/api/tap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyvo
import requests


def load_token(repo_root: Path) -> str:
    tok = os.environ.get("RSP_TOKEN")
    if tok:
        return tok
    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            s = line.strip()
            if s.startswith("RSP_TOKEN="):
                return s.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("RSP_TOKEN not found in $RSP_TOKEN or .env")


def tap_service(url: str, token: str) -> pyvo.dal.TAPService:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    session.headers["User-Agent"] = "debass-rsp-probe/0.1"
    return pyvo.dal.TAPService(url, session=session)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://data.lsst.cloud/api/tap",
                    help="TAP endpoint (default: USDF)")
    ap.add_argument("--schema", help="Limit to one schema (else list all)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    token = load_token(repo_root)
    print(f"TAP endpoint: {args.url}")
    print(f"Token: {len(token)}-char bearer (hidden)")

    svc = tap_service(args.url, token)

    print("\n=== Schemas ===")
    try:
        rows = list(svc.search(
            "SELECT schema_name, description FROM tap_schema.schemas"
        ))
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {e}")
        return 1
    for r in rows:
        desc = (r['description'] or '').strip()[:80]
        print(f"  {r['schema_name']:<40}  {desc}")

    schema_names = [args.schema] if args.schema else [r['schema_name'] for r in rows]

    for sn in schema_names:
        print(f"\n=== tables in {sn} ===")
        try:
            tr = list(svc.search(
                f"SELECT TOP 60 table_name, description FROM tap_schema.tables "
                f"WHERE schema_name = '{sn}'"
            ))
            if not tr:
                print("  (no tables)")
            for r in tr:
                desc = (r['description'] or '').strip()[:60]
                print(f"  {r['table_name']:<40}  {desc}")
        except Exception as e:
            print(f"  ERR: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
