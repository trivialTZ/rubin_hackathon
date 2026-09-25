#!/usr/bin/env python3
"""Pull RSP DIA lightcurves for the 38 DEBaSS follow-up targets.

Input : combined_table_38_CORRECTED.csv (SNID + optional diaObjectId)
        data/tns_public_objects.csv     (RA/Dec for all 38 SNIDs)
Output: reports/rsp_debass38/
          schemas.csv          discovered TAP schemas/tables
          match_summary.csv    per-SNID DiaObject match (cone + declared id)
          diasources.parquet   stacked per-epoch DIA lightcurves

Schema is auto-discovered: tries DP2/EDP2 aliases first, falls back to DP1.
Nothing here is DP1-specific beyond the column names, which are shared.

Usage:
    python scripts/_exp_rsp_debass38_pull.py [--csv PATH] [--radius 2.0]
                                             [--schema dp2] [--no-sources]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from debass_meta.access.rubin_rsp import RSPClient  # noqa: E402

DEFAULT_CSV = Path.home() / "Downloads" / "combined_table_38_CORRECTED.csv"
OUT_DIR = REPO / "reports" / "rsp_debass38"

# Preference order when auto-detecting the release schema.
SCHEMA_PREFERENCE = ["dp2", "edp2", "dp1"]

SOURCE_COLS = (
    "diaSourceId", "diaObjectId", "band", "midpointMjdTai",
    "ra", "dec", "snr", "psfFlux", "psfFluxErr",
    "scienceFlux", "scienceFluxErr", "reliability", "visit", "detector",
)


def load_targets(csv_path: Path) -> pd.DataFrame:
    """CSV rows joined to TNS RA/Dec. All 38 SNIDs are expected to resolve."""
    rows = [r for r in csv.DictReader(csv_path.open()) if r.get("SNID")]
    tgt = pd.DataFrame(rows)
    # Nullable Int64, never float64: diaObjectId ~1e17 exceeds float64's exact
    # integer range (2**53), so a float round-trip silently mangles the low digits.
    # Parse straight from the string — pd.to_numeric would go via float64 first.
    tgt["declared_diaObjectId"] = pd.array(
        [int(s) if (s := str(v).strip()) not in ("—", "-", "", "nan") else None
         for v in tgt["diaObjectId"]],
        dtype="Int64",
    )

    tns = pd.read_csv(REPO / "data" / "tns_public_objects.csv",
                      skiprows=1, low_memory=False)
    tns["name"] = tns["name"].astype(str).str.strip()
    tns = tns[["name", "ra", "declination", "type", "redshift"]].rename(
        columns={"declination": "dec", "type": "tns_type", "redshift": "tns_z"}
    )
    merged = tgt.merge(tns, left_on="SNID", right_on="name", how="left")
    missing = merged.loc[merged["ra"].isna(), "SNID"].tolist()
    if missing:
        print(f"  WARNING: no TNS coords for {len(missing)}: {missing}")
    return merged


def detect_schema(client: RSPClient, override: str | None) -> str:
    """Return the DIA release schema to query, printing the full inventory."""
    schemas = client.query("SELECT schema_name FROM tap_schema.schemas")
    names = sorted(schemas["schema_name"].astype(str).tolist())
    print(f"=== {len(names)} TAP schemas visible ===")
    for n in names:
        print(f"  {n}")

    tables = client.query("SELECT schema_name, table_name FROM tap_schema.tables")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables.to_csv(OUT_DIR / "schemas.csv", index=False)

    has_dia = {
        s for s, g in tables.groupby("schema_name")
        if any(str(t).lower().endswith("diaobject") for t in g["table_name"])
    }
    print(f"\nSchemas with a DiaObject table: {sorted(has_dia) or '(none)'}")

    if override:
        if override not in has_dia:
            print(f"  NOTE: --schema {override} has no DiaObject table; trying anyway")
        return override
    for cand in SCHEMA_PREFERENCE:
        if cand in has_dia:
            return cand
    if has_dia:
        return sorted(has_dia)[0]
    raise SystemExit("No schema with a DiaObject table is visible to this token.")


def cone_match(client: RSPClient, schema: str, tgt: pd.DataFrame,
               radius_arcsec: float) -> pd.DataFrame:
    """Cone-search each target; keep the nearest DiaObject within the radius."""
    r_deg = radius_arcsec / 3600.0
    out = []
    # Column-first access throughout. `tgt.iterrows()` would collapse each row to
    # a single float64 Series and corrupt declared_diaObjectId (see load_targets).
    for i in range(len(tgt)):
        row = {c: tgt[c].iloc[i] for c in tgt.columns}
        rec = {
            "SNID": row["SNID"],
            "tns_ra": row["ra"], "tns_dec": row["dec"],
            "tns_type": row.get("tns_type"), "tns_z": row.get("tns_z"),
            "declared_diaObjectId": row["declared_diaObjectId"],
            "Status": row.get("Status"), "Days": row.get("Days"),
        }
        if pd.isna(row["ra"]):
            rec.update(matched_diaObjectId=None, sep_arcsec=None,
                       nDiaSources=None, n_cone_hits=0)
            out.append(rec)
            continue
        adql = (
            f"SELECT diaObjectId, ra, dec, nDiaSources, "
            f"DISTANCE(POINT('ICRS', ra, dec), "
            f"POINT('ICRS', {row['ra']}, {row['dec']})) AS sep_deg "
            f"FROM {schema}.DiaObject "
            f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {row['ra']}, {row['dec']}, {r_deg})) = 1 "
            f"ORDER BY sep_deg ASC"
        )
        try:
            hits = client.query(adql)
        except Exception as e:  # noqa: BLE001 — per-target failure is not fatal
            print(f"  {row['SNID']:<9} QUERY FAILED: {type(e).__name__}: {str(e)[:120]}")
            rec.update(matched_diaObjectId=None, sep_arcsec=None,
                       nDiaSources=None, n_cone_hits=-1)
            out.append(rec)
            continue

        rec["n_cone_hits"] = len(hits)
        if len(hits):
            # Column-first: hits.iloc[0]["diaObjectId"] returns float64 and drops
            # the last two digits of an ~1e17 ID.
            oid = int(hits["diaObjectId"].iloc[0])
            rec.update(
                matched_diaObjectId=oid,
                sep_arcsec=float(hits["sep_deg"].iloc[0]) * 3600.0,
                nDiaSources=int(hits["nDiaSources"].iloc[0]),
            )
            flag = ""
            if pd.notna(row["declared_diaObjectId"]):
                same = int(row["declared_diaObjectId"]) == oid
                flag = "  [matches CSV]" if same else "  [differs from CSV id]"
            print(f"  {row['SNID']:<9} -> {oid} "
                  f"sep={rec['sep_arcsec']:.2f}\" n={rec['nDiaSources']}{flag}")
        else:
            rec.update(matched_diaObjectId=None, sep_arcsec=None, nDiaSources=None)
            print(f"  {row['SNID']:<9} -> no DiaObject within {radius_arcsec}\"")
        out.append(rec)
    df = pd.DataFrame(out)
    # pd.DataFrame() turns int-or-None columns into float64. Rebuild the ID
    # columns as Int64 from the original dicts, which still hold exact ints.
    for col in ("matched_diaObjectId", "declared_diaObjectId"):
        df[col] = pd.array([r.get(col) for r in out], dtype="Int64")
    return df


def pull_sources(client: RSPClient, schema: str, ids: list[int]) -> pd.DataFrame:
    """Fetch DiaSource rows for the matched objects, in batches."""
    frames = []
    cols = ", ".join(SOURCE_COLS)
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        adql = (
            f"SELECT {cols} FROM {schema}.DiaSource "
            f"WHERE diaObjectId IN ({','.join(str(x) for x in chunk)})"
        )
        try:
            frames.append(client.query(adql))
        except Exception as e:  # noqa: BLE001
            print(f"  DiaSource batch {i // 200 + 1} FAILED: "
                  f"{type(e).__name__}: {str(e)[:160]}")
    if not frames:
        return pd.DataFrame(columns=list(SOURCE_COLS))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["diaObjectId", "midpointMjdTai"]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--radius", type=float, default=2.0,
                    help="cone radius in arcsec (default 2.0)")
    ap.add_argument("--schema", default=None,
                    help="force a release schema instead of auto-detecting")
    ap.add_argument("--no-sources", action="store_true",
                    help="match only; skip the DiaSource pull")
    args = ap.parse_args()

    client = RSPClient()
    print(f"TAP: {client.tap_url}\n")

    schema = detect_schema(client, args.schema)
    print(f"\n=== Using schema: {schema} ===")
    try:
        n_obj = client.count(table=f"{schema}.DiaObject")
        rng = client.query(
            f"SELECT MIN(midpointMjdTai) AS mjd_min, MAX(midpointMjdTai) AS mjd_max "
            f"FROM {schema}.DiaSource"
        )
        print(f"  DiaObject rows: {n_obj:,}")
        print(f"  DiaSource MJD range: {rng['mjd_min'].iloc[0]:.2f} – "
              f"{rng['mjd_max'].iloc[0]:.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"  inventory query failed: {type(e).__name__}: {str(e)[:200]}")

    tgt = load_targets(args.csv)
    print(f"\n=== Cone-matching {len(tgt)} DEBaSS targets "
          f"at {args.radius}\" ===")
    match = cone_match(client, schema, tgt, args.radius)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    match.to_csv(OUT_DIR / "match_summary.csv", index=False)
    n_hit = int(match["matched_diaObjectId"].notna().sum())
    print(f"\nMatched {n_hit}/{len(match)} targets -> "
          f"{OUT_DIR / 'match_summary.csv'}")

    if args.no_sources or n_hit == 0:
        return 0

    ids = [int(x) for x in match["matched_diaObjectId"].dropna().tolist()]
    print(f"\n=== Pulling DiaSource rows for {len(ids)} objects ===")
    src = pull_sources(client, schema, ids)
    if len(src):
        ok = match["matched_diaObjectId"].notna()
        id2snid = {int(o): s for o, s in
                   zip(match.loc[ok, "matched_diaObjectId"], match.loc[ok, "SNID"])}
        src["SNID"] = src["diaObjectId"].map(id2snid)
        missing = int(src["SNID"].isna().sum())
        if missing:
            print(f"  WARNING: {missing} rows failed to map back to a SNID")
        src.to_parquet(OUT_DIR / "diasources.parquet", index=False)
        print(f"  {len(src):,} rows, {src['diaObjectId'].nunique()} objects, "
              f"bands={sorted(set(src['band']))}")
        print(f"  -> {OUT_DIR / 'diasources.parquet'}")
        # Integrity check: rows pulled must equal the catalogue's nDiaSources.
        # A shortfall means IDs were corrupted or a batch silently truncated.
        per = src.groupby("SNID").size()
        chk = match.loc[ok, ["SNID", "nDiaSources"]].copy()
        chk["pulled"] = chk["SNID"].map(per).fillna(0).astype(int)
        chk["expected"] = chk["nDiaSources"].astype(int)
        chk = chk.sort_values("expected", ascending=False)
        bad = chk[chk["pulled"] != chk["expected"]]
        print("\n  SNID       expected  pulled")
        for _, r in chk.iterrows():
            mark = "" if r["pulled"] == r["expected"] else "   <-- MISMATCH"
            print(f"    {r['SNID']:<9} {r['expected']:>8} {r['pulled']:>7}{mark}")
        print(f"\n  integrity: {len(chk) - len(bad)}/{len(chk)} objects complete, "
              f"{int(chk['pulled'].sum())}/{int(chk['expected'].sum())} detections")
    else:
        print("  no DiaSource rows returned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
