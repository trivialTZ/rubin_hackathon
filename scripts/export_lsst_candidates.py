"""Export LSST Lasair candidates to a local CSV.

The output is intentionally named like a labels file for convenience, but it
contains candidate metadata, not science labels.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def export_lsst_candidates(
    *,
    output_path: Path,
    lookback_days: int = 2,
    max_candidates: int = 100,
    require_sherlock: bool = False,
) -> Path:
    import pandas as pd

    now_mjd = time.time() / 86400.0 + 40587.0
    since_mjd = now_mjd - lookback_days
    if require_sherlock:
        from scripts.discover_nightly import query_lasair_recent

        df = query_lasair_recent(since_mjd=since_mjd, max_candidates=max_candidates)
    else:
        df = _query_recent_lsst_candidates(since_mjd=since_mjd, max_candidates=max_candidates)

    if len(df) == 0:
        df = pd.DataFrame(
            columns=[
                "object_id",
                "ra",
                "dec",
                "n_det",
                "last_mjd",
                "discovery_broker",
            ]
        )

    df = df.copy()
    df["object_id"] = df["object_id"].astype("string")
    df["survey"] = "LSST"
    df["identifier_kind"] = "lsst_dia_object_id"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LSST Lasair candidates to CSV")
    parser.add_argument("--output", default="data/labels_lsst.csv")
    parser.add_argument("--lookback-days", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument(
        "--require-sherlock",
        action="store_true",
        help="Restrict export to LSST objects with Sherlock SN/NT context",
    )
    args = parser.parse_args()

    path = export_lsst_candidates(
        output_path=Path(args.output),
        lookback_days=args.lookback_days,
        max_candidates=args.max_candidates,
        require_sherlock=args.require_sherlock,
    )
    print(f"Wrote LSST candidate CSV → {path}")


def _query_recent_lsst_candidates(*, since_mjd: float, max_candidates: int):
    import pandas as pd
    from dotenv import load_dotenv

    from debass_meta.access.identifiers import build_lasair_api_endpoint, is_lsst_lasair_endpoint

    load_dotenv()
    token = os.environ.get("LASAIR_TOKEN")
    site_base = os.environ.get("LASAIR_ENDPOINT", "https://lasair.lsst.ac.uk").rstrip("/")
    if not token or not is_lsst_lasair_endpoint(site_base):
        return pd.DataFrame()

    try:
        from lasair import lasair_client as lasair_mod
    except ImportError:
        return pd.DataFrame()

    api_base = build_lasair_api_endpoint(site_base)
    try:
        client = lasair_mod(token, endpoint=api_base)
    except TypeError:
        client = lasair_mod(token)

    try:
        results = client.query(
            "diaObjectId,ra,decl,nDiaSources,lastDiaSourceMjdTai",
            "objects",
            f"lastDiaSourceMjdTai > {since_mjd} ORDER BY lastDiaSourceMjdTai DESC",
            limit=max_candidates,
        )
    except Exception:
        return pd.DataFrame()

    rows = [
        {
            "object_id": str(obj.get("diaObjectId", "")).strip(),
            "ra": float(obj.get("ra")),
            "dec": float(obj.get("decl")),
            "n_det": int(obj.get("nDiaSources", 0)),
            "last_mjd": float(obj.get("lastDiaSourceMjdTai")),
            "discovery_broker": "lasair",
        }
        for obj in results
        if str(obj.get("diaObjectId", "")).strip()
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
