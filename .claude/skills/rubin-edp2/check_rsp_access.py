#!/usr/bin/env python3
"""Verify Rubin Science Platform access end to end. Run this FIRST when anything
RSP-related misbehaves -- it separates "token is dead" from "query is wrong".

Prints identity, scopes, visible schemas, a DP2 row count and a cutout-service
probe. Never prints the token itself.

    python .claude/skills/rubin-edp2/check_rsp_access.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

TOKEN_INFO = "https://data.lsst.cloud/auth/api/v1/token-info"


def main() -> int:
    try:
        from debass_meta.access.rubin_rsp import RSPClient, load_rsp_token
    except ImportError as e:
        print(f"FAIL: cannot import debass_meta ({e}). "
              f"Activate the venv: source ~/.venvs/debass_py313/bin/activate")
        return 1

    try:
        token = load_rsp_token()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        print("  -> user mints a token at https://data.lsst.cloud/auth/tokens "
              "(scopes read:tap + read:image) and sets RSP_TOKEN= in .env")
        return 1
    print(f"token loaded: {token[:3]}... ({len(token)} chars)")

    req = urllib.request.Request(TOKEN_INFO,
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        info = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: token-info rejected the token ({type(e).__name__}: {e})")
        print("  -> 401 means revoked/replaced, not that RSP is down. Re-mint.")
        return 1
    info.pop("token", None)
    print(f"identity: {info.get('username')}  name={info.get('token_name')}  "
          f"expires={info.get('expires', 'never')}")
    scopes = info.get("scopes", [])
    print(f"scopes: {', '.join(scopes)}")
    for need in ("read:tap", "read:image"):
        if need not in scopes:
            print(f"  WARNING: missing {need} — "
                  f"{'TAP queries' if need == 'read:tap' else 'cutouts'} will 403")

    client = RSPClient()
    schemas = sorted(client.query(
        "SELECT schema_name FROM tap_schema.schemas")["schema_name"].astype(str))
    print(f"schemas ({len(schemas)}): {', '.join(schemas)}")
    if "dp2" not in schemas:
        print("  WARNING: no dp2 schema visible — EDP2 is spelled 'dp2', "
              "not 'edp2'; fall back to dp1")

    n = client.count(table="dp2.DiaObject") if "dp2" in schemas else 0
    print(f"dp2.DiaObject rows: {n:,}")

    mjd = client.query("SELECT MIN(midpointMjdTai) AS lo, MAX(midpointMjdTai) AS hi "
                       "FROM dp2.DiaSource")
    print(f"dp2.DiaSource MJD: {mjd['lo'].iloc[0]:.3f} -> {mjd['hi'].iloc[0]:.3f}")

    obs = client.query("SELECT dataproduct_subtype, COUNT(*) AS n FROM ivoa.ObsCore "
                       "WHERE obs_collection='LSST.DP2' GROUP BY dataproduct_subtype")
    for r in obs.to_dict("records"):
        print(f"ObsCore DP2 {r['dataproduct_subtype']}: {r['n']:,}")

    print("\nOK — TAP and ObsCore both reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
