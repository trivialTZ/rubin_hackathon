"""Crossmatch LSST candidates to nearby ZTF objects via ALeRCE conesearch."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2.0))
    ddec = dec1 - dec2
    return math.hypot(dra, ddec) * 3600.0


def _normalize_hits(df) -> Any:
    if df is None or len(df) == 0:
        return df
    if "oid" in df.columns:
        return df.reset_index(drop=True)
    return df.reset_index().rename(columns={"index": "oid"})


def crossmatch_lsst_to_ztf(
    *,
    candidates_path: Path,
    output_path: Path,
    radius_arcsec: float = 2.0,
    page_size: int = 10,
) -> Path:
    import pandas as pd
    from alerce.core import Alerce

    candidates = pd.read_csv(candidates_path, dtype={"object_id": "string"})
    required = {"object_id", "ra", "dec"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate CSV is missing required columns: {missing}")

    client = Alerce()
    rows: list[dict[str, Any]] = []

    for _, candidate in candidates.iterrows():
        lsst_object_id = str(candidate["object_id"]).strip()
        ra = float(candidate["ra"])
        dec = float(candidate["dec"])
        hits = client.query_objects(
            format="pandas",
            survey="ztf",
            ra=ra,
            dec=dec,
            radius=radius_arcsec,
            page_size=page_size,
        )
        hits = _normalize_hits(hits)

        base_row = {
            "lsst_object_id": lsst_object_id,
            "lsst_ra": ra,
            "lsst_dec": dec,
            "lsst_n_det": int(candidate["n_det"]) if "n_det" in candidate and not pd.isna(candidate["n_det"]) else None,
            "lsst_last_mjd": float(candidate["last_mjd"]) if "last_mjd" in candidate and not pd.isna(candidate["last_mjd"]) else None,
            "match_radius_arcsec": radius_arcsec,
            "association_kind": "position_crossmatch",
            "association_source": "alerce_conesearch",
        }

        if hits is None or len(hits) == 0:
            rows.append(
                {
                    **base_row,
                    "match_status": "no_match",
                    "match_count": 0,
                    "ztf_object_id": None,
                    "sep_arcsec": None,
                    "ztf_meanra": None,
                    "ztf_meandec": None,
                    "ztf_lastmjd": None,
                    "ztf_ndet": None,
                    "candidate_matches_json": "[]",
                }
            )
            continue

        hits = hits.copy()
        hits["sep_arcsec"] = hits.apply(
            lambda hit: separation_arcsec(
                ra,
                dec,
                float(hit.get("meanra", math.nan)),
                float(hit.get("meandec", math.nan)),
            ),
            axis=1,
        )
        sort_cols = ["sep_arcsec"]
        ascending = [True]
        if "lastmjd" in hits.columns:
            sort_cols.append("lastmjd")
            ascending.append(False)
        hits = hits.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

        best = hits.iloc[0]
        matches = []
        for _, hit in hits.head(page_size).iterrows():
            matches.append(
                {
                    "ztf_object_id": str(hit.get("oid")),
                    "sep_arcsec": float(hit["sep_arcsec"]),
                    "meanra": float(hit.get("meanra")) if not pd.isna(hit.get("meanra")) else None,
                    "meandec": float(hit.get("meandec")) if not pd.isna(hit.get("meandec")) else None,
                    "lastmjd": float(hit.get("lastmjd")) if "lastmjd" in hit and not pd.isna(hit.get("lastmjd")) else None,
                    "ndet": int(hit.get("ndet")) if "ndet" in hit and not pd.isna(hit.get("ndet")) else None,
                }
            )

        rows.append(
            {
                **base_row,
                "match_status": "matched",
                "match_count": int(len(hits)),
                "ztf_object_id": str(best.get("oid")),
                "sep_arcsec": float(best["sep_arcsec"]),
                "ztf_meanra": float(best.get("meanra")) if not pd.isna(best.get("meanra")) else None,
                "ztf_meandec": float(best.get("meandec")) if not pd.isna(best.get("meandec")) else None,
                "ztf_lastmjd": float(best.get("lastmjd")) if "lastmjd" in best and not pd.isna(best.get("lastmjd")) else None,
                "ztf_ndet": int(best.get("ndet")) if "ndet" in best and not pd.isna(best.get("ndet")) else None,
                "candidate_matches_json": json.dumps(matches, sort_keys=True),
            }
        )

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Crossmatch LSST candidates to ZTF objects using ALeRCE")
    parser.add_argument("--candidates", default="data/labels_lsst.csv")
    parser.add_argument("--output", default="data/crossmatch/lsst_to_ztf.csv")
    parser.add_argument("--radius-arcsec", type=float, default=2.0)
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()

    path = crossmatch_lsst_to_ztf(
        candidates_path=Path(args.candidates),
        output_path=Path(args.output),
        radius_arcsec=args.radius_arcsec,
        page_size=args.page_size,
    )
    print(f"Wrote LSST→ZTF association CSV → {path}")


if __name__ == "__main__":
    main()
