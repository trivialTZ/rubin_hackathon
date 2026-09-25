#!/usr/bin/env python3
"""WP1 — TNS backlog sweep: secure spec labels for the LSST training universe.

We hold a TNS master parquet (``data/truth/tns_public.parquet``) with ~1,396
spectroscopically typed objects at dec < +15 discovered since 2025-06-01, but
only ~300 are matched to LSST diaObjectIds. This sweep matches the REST against
the LSST object universe via two routes:

  * Route A (``lasair_xm``): fully paginate the Lasair LSST ``crossmatch_tns``
    table joined with ``objects`` and match TNS objname -> crossmatch tns_name
    (normalized). Separation is computed with the TNS-BULK coordinates treated
    as authoritative.
  * Route B (``alerce_cone``): for filtered TNS objects NOT matched by Route A,
    cone-search the ALeRCE LSST object API within ``--max-sep-arcsec`` of the
    TNS position. Per-object disk cache makes reruns free and resumable.

Every match enforces sep <= max-sep, maps the TNS type to the metaDEBASS ternary
via the repo mapper, and is tagged ``label_quality='spectroscopic'`` /
``label_source='tns_backlog_sweep'``. Output is a standalone parquet
(``--out``); routing into the frozen benchmark manifest / truth tables is the
orchestrator's job — this script never touches them.

The pure helpers (name normalization / authoritative sep / TNS filter / xm index
build / cone-hit matching / ternary+hash finalize / cache reuse) carry no live
network and are unit-tested in tests/test_sweep_tns_backlog.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# Reuse existing repo primitives — never reimplement.
from debass_meta.access.tns import map_tns_type_to_ternary
from scripts.build_truth_lsst_live import hash_route
from scripts.crossmatch_lsst_to_ztf import separation_arcsec

LABEL_QUALITY = "spectroscopic"
LABEL_SOURCE = "tns_backlog_sweep"
ROUTE_LASAIR = "lasair_xm"
ROUTE_ALERCE = "alerce_cone"

OUTPUT_COLUMNS = [
    "object_id",
    "tns_name",
    "tns_type",
    "final_class_ternary",
    "label_quality",
    "label_source",
    "sep_arcsec",
    "route",
    "tns_redshift",
    "tns_discovery_date",
    "hash_route",
    "match_count",
]

# Route-A Lasair query. Qualified columns MUST be aliased — Lasair /query/
# flattens qualified names to bare keys (objects.ra and crossmatch_tns.ra
# collide otherwise). ORDER BY a unique key so offset pagination is
# deterministic (no repeated/skipped rows across pages).
_XM_SELECTED = (
    "objects.diaObjectId AS dia_id, objects.ra AS obj_ra, "
    "objects.decl AS obj_decl, crossmatch_tns.tns_name AS tns_name"
)
_XM_TABLES = "objects,crossmatch_tns"
_XM_CONDITIONS = "1=1 ORDER BY objects.diaObjectId"

_PREFIX_RE = re.compile(r"^(at|sn)", re.IGNORECASE)


# ------------------------------------------------------------------ #
# Pure helpers (no network; unit-tested)                              #
# ------------------------------------------------------------------ #


def normalize_tns_name(name: Any) -> str:
    """Normalize a TNS name for cross-catalog matching.

    Strip surrounding/interior whitespace, drop a leading ``AT``/``SN`` prefix,
    lowercase. ``'SN 2024xyz'``, ``'AT2024xyz'`` and ``'2024xyz'`` all collapse
    to ``'2024xyz'``. Empty / NaN -> ``''``.
    """
    if name is None:
        return ""
    text = str(name).strip()
    if not text or text.lower() == "nan":
        return ""
    compact = "".join(text.split())
    compact = _PREFIX_RE.sub("", compact, count=1)
    return compact.lower()


def authoritative_sep(
    tns_ra: Any, tns_dec: Any, other_ra: Any, other_dec: Any
) -> float | None:
    """Separation (arcsec) with the TNS-BULK position as the authoritative anchor.

    Returns ``None`` if any coordinate is missing / non-finite.
    """
    coords = []
    for value in (tns_ra, tns_dec, other_ra, other_dec):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN
            return None
        coords.append(f)
    return separation_arcsec(coords[0], coords[1], coords[2], coords[3])


def filter_tns_candidates(df, *, max_dec: float, since: str):
    """Filter the TNS master table to typed, southern, recent candidates.

    * ``type`` non-null and non-empty
    * ``declination`` < ``max_dec``
    * ``discoverydate`` >= ``since`` (lexicographic on ISO-ish strings is safe;
      discoverydate is stored ``YYYY-MM-DD HH:MM:SS.mmm``)
    """
    type_col = df["type"].astype("string")
    mask = type_col.notna() & (type_col.str.strip() != "")
    mask &= df["declination"].astype(float) < float(max_dec)
    disc = df["discoverydate"].astype("string").str.strip()
    mask &= disc.notna() & (disc >= str(since))
    return df[mask].copy()


def tns_candidates_from_df(df) -> list[dict]:
    """Convert a filtered TNS DataFrame into plain candidate dicts."""
    out: list[dict] = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        prefix = d.get("name_prefix")
        objname = d.get("objname")
        full = objname
        if prefix and str(prefix).strip() and str(prefix).lower() != "nan":
            full = f"{str(prefix).strip()} {str(objname).strip()}"
        out.append(
            {
                "objname": None if objname is None else str(objname).strip(),
                "tns_name": None if full is None else str(full).strip(),
                "tns_type": d.get("type"),
                "ra": d.get("ra"),
                "dec": d.get("declination"),
                "tns_redshift": d.get("redshift"),
                "tns_discovery_date": (
                    None
                    if d.get("discoverydate") is None
                    else str(d.get("discoverydate"))
                ),
            }
        )
    return out


def build_xm_index(records: list[dict]) -> dict[str, list[dict]]:
    """Build ``normalized tns_name -> [ {dia_id, ra, dec}, ... ]`` from xm records.

    Records come from the aliased Lasair join (``dia_id`` / ``obj_ra`` /
    ``obj_decl`` / ``tns_name``). Records without a diaObjectId or an
    un-normalizable name are skipped.
    """
    index: dict[str, list[dict]] = {}
    for rec in records:
        dia = rec.get("dia_id", rec.get("diaObjectId"))
        if dia is None or not str(dia).strip():
            continue
        key = normalize_tns_name(rec.get("tns_name"))
        if not key:
            continue
        index.setdefault(key, []).append(
            {
                "dia_id": str(dia).strip(),
                "ra": rec.get("obj_ra", rec.get("ra")),
                "dec": rec.get("obj_decl", rec.get("decl")),
            }
        )
    return index


def match_route_a(
    candidates: list[dict], xm_index: dict[str, list[dict]], *, max_sep: float
) -> tuple[list[dict], set[str]]:
    """Match candidates against the Lasair xm index (Route A).

    Returns ``(raw_matches, matched_objnames)``. ``raw_matches`` are pre-finalize
    dicts (object_id/tns_name/tns_type/sep_arcsec/route/redshift/discovery).
    ``matched_objnames`` are the objnames that produced >=1 in-radius match, so
    Route B can skip them.
    """
    raw: list[dict] = []
    matched_objnames: set[str] = set()
    for cand in candidates:
        key = normalize_tns_name(cand.get("objname"))
        if not key or key not in xm_index:
            continue
        for hit in xm_index[key]:
            sep = authoritative_sep(cand.get("ra"), cand.get("dec"), hit["ra"], hit["dec"])
            if sep is None or sep > max_sep:
                continue
            matched_objnames.add(cand["objname"])
            raw.append(
                {
                    "object_id": hit["dia_id"],
                    "tns_name": cand.get("tns_name"),
                    "tns_type": cand.get("tns_type"),
                    "sep_arcsec": sep,
                    "route": ROUTE_LASAIR,
                    "tns_redshift": cand.get("tns_redshift"),
                    "tns_discovery_date": cand.get("tns_discovery_date"),
                }
            )
    return raw, matched_objnames


def hits_to_matches(cand: dict, hits: list[dict], *, max_sep: float) -> list[dict]:
    """Turn ALeRCE cone hits into raw match dicts (Route B).

    Separation is TNS-authoritative: TNS position vs each hit's mean position.
    """
    raw: list[dict] = []
    for hit in hits:
        dia = hit.get("oid", hit.get("diaObjectId", hit.get("dia_id")))
        if dia is None or not str(dia).strip():
            continue
        sep = authoritative_sep(
            cand.get("ra"),
            cand.get("dec"),
            hit.get("meanra", hit.get("ra")),
            hit.get("meandec", hit.get("dec")),
        )
        if sep is None or sep > max_sep:
            continue
        raw.append(
            {
                "object_id": str(dia).strip(),
                "tns_name": cand.get("tns_name"),
                "tns_type": cand.get("tns_type"),
                "sep_arcsec": sep,
                "route": ROUTE_ALERCE,
                "tns_redshift": cand.get("tns_redshift"),
                "tns_discovery_date": cand.get("tns_discovery_date"),
            }
        )
    return raw


def finalize_matches(raw_matches: list[dict], *, max_sep: float) -> list[dict]:
    """Map ternary, tag labels/hash_route, dedup by object_id, flag match_count.

    * Rows whose TNS type does not map to a ternary label are dropped (cannot
      mint a usable spectroscopic label).
    * Rows separated by more than ``max_sep`` are dropped (belt-and-braces).
    * De-dup: one row per ``object_id``, keeping the smallest ``sep_arcsec``.
    * ``match_count``: number of distinct object_ids sharing a ``tns_name``
      (LSST re-detections of one transient) — all kept, each flagged.
    """
    best_by_id: dict[str, dict] = {}
    for m in raw_matches:
        sep = m.get("sep_arcsec")
        if sep is None or float(sep) > max_sep:
            continue
        ternary = map_tns_type_to_ternary(
            str(m["tns_type"]) if m.get("tns_type") is not None else None
        )
        if ternary is None:
            continue
        object_id = str(m["object_id"]).strip()
        row = {
            "object_id": object_id,
            "tns_name": m.get("tns_name"),
            "tns_type": m.get("tns_type"),
            "final_class_ternary": ternary,
            "label_quality": LABEL_QUALITY,
            "label_source": LABEL_SOURCE,
            "sep_arcsec": float(sep),
            "route": m.get("route"),
            "tns_redshift": m.get("tns_redshift"),
            "tns_discovery_date": m.get("tns_discovery_date"),
            "hash_route": hash_route(object_id),
        }
        prev = best_by_id.get(object_id)
        if prev is None or row["sep_arcsec"] < prev["sep_arcsec"]:
            best_by_id[object_id] = row

    rows = list(best_by_id.values())
    # match_count per tns_name (distinct object_ids sharing the label).
    counts: dict[str, int] = {}
    for r in rows:
        counts[str(r["tns_name"])] = counts.get(str(r["tns_name"]), 0) + 1
    for r in rows:
        r["match_count"] = counts[str(r["tns_name"])]
    rows.sort(key=lambda r: (r["object_id"]))
    return rows


def cone_hits_cached(
    objname: str,
    ra: Any,
    dec: Any,
    radius: float,
    *,
    cache_dir: Path,
    fetch: Callable[[float, float, float], list[dict]],
    sleep_fn: Callable[[float], None] = time.sleep,
    sleep_s: float = 0.2,
) -> tuple[list[dict], bool]:
    """Cone-search hits for one object, disk-cached per objname (resumable).

    Returns ``(hits, from_cache)``. A cache HIT never calls ``fetch`` and never
    sleeps. A cache MISS calls ``fetch(ra, dec, radius)``, persists the result,
    then sleeps ``sleep_s`` to rate-limit live calls. ``fetch`` may raise — the
    caller is responsible for tolerating/recording per-object failures.
    """
    cache_path = cache_dir / f"{_safe_filename(objname)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text()), True
    hits = fetch(float(ra), float(dec), float(radius))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(hits))
    sleep_fn(sleep_s)
    return hits, False


def _safe_filename(name: str) -> str:
    """File-system-safe token for a per-object cache file."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name).strip()) or "unnamed"


# ------------------------------------------------------------------ #
# Live I/O (network; cached + resumable)                              #
# ------------------------------------------------------------------ #


def harvest_xm_records(
    *, cache_dir: Path, page_size: int = 1000, max_pages: int = 100
) -> list[dict]:
    """Fully paginate the Lasair LSST ``objects,crossmatch_tns`` join (Route A).

    One JSON page per file, tagged with a hash of the exact query so a changed
    SELECT/conditions never reuses a stale page. A short/empty page ends the
    harvest.
    """
    from debass_meta.access.lasair import LasairAdapter

    cache_dir.mkdir(parents=True, exist_ok=True)
    adapter = LasairAdapter()
    query_tag = hashlib.sha1(
        f"{_XM_SELECTED}|{_XM_TABLES}|{_XM_CONDITIONS}".encode()
    ).hexdigest()[:8]
    records: list[dict] = []
    for page in range(max_pages):
        page_path = cache_dir / f"xm_{query_tag}_{page:04d}.json"
        if page_path.exists():
            rows = json.loads(page_path.read_text())
        else:
            rows = adapter._query_api(
                selected=_XM_SELECTED,
                tables=_XM_TABLES,
                conditions=_XM_CONDITIONS,
                limit=page_size,
                offset=page * page_size,
                survey="lsst",
            )
            page_path.write_text(json.dumps(rows))
        records.extend(rows)
        print(f"  Route A page {page}: {len(rows)} rows (total {len(records)})")
        if len(rows) < page_size:
            break
    return records


def _make_alerce_fetch(client) -> Callable[[float, float, float], list[dict]]:
    """Build an ALeRCE LSST cone fetch: (ra, dec, radius) -> [ {oid, meanra, meandec} ]."""

    def fetch(ra: float, dec: float, radius: float) -> list[dict]:
        result = client.query_objects(
            survey="lsst",
            format="pandas",
            ra=float(ra),
            dec=float(dec),
            radius=float(radius),
            page_size=20,
        )
        hits: list[dict] = []
        if result is None or len(result) == 0:
            return hits
        df = result
        if "oid" not in df.columns:
            df = df.reset_index().rename(columns={"index": "oid"})
        for _, hit in df.iterrows():
            hits.append(
                {
                    "oid": str(hit.get("oid")),
                    "meanra": _opt_float(hit.get("meanra")),
                    "meandec": _opt_float(hit.get("meandec")),
                }
            )
        return hits

    return fetch


def _opt_float(value: Any) -> float | None:
    try:
        f = float(value)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def run_route_b(
    candidates: list[dict],
    *,
    max_sep: float,
    cache_dir: Path,
    fetch: Callable[[float, float, float], list[dict]],
    sleep_s: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Cone-search each remaining candidate (Route B). Returns (raw_matches, failures)."""
    raw: list[dict] = []
    failures: list[dict] = []
    for cand in candidates:
        ra, dec = _opt_float(cand.get("ra")), _opt_float(cand.get("dec"))
        if ra is None or dec is None:
            failures.append({"objname": cand.get("objname"), "reason": "missing_coords"})
            continue
        try:
            hits, _from_cache = cone_hits_cached(
                cand["objname"],
                ra,
                dec,
                max_sep,
                cache_dir=cache_dir,
                fetch=fetch,
                sleep_s=sleep_s,
            )
        except Exception as exc:  # tolerate + record; keep going (resumable)
            failures.append(
                {"objname": cand.get("objname"), "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        raw.extend(hits_to_matches(cand, hits, max_sep=max_sep))
    return raw, failures


# ------------------------------------------------------------------ #
# Summary                                                             #
# ------------------------------------------------------------------ #


def summarize(
    *,
    n_candidates: int,
    rows: list[dict],
    matched_objnames_a: set[str],
    failures: list[dict],
) -> dict:
    a_rows = [r for r in rows if r["route"] == ROUTE_LASAIR]
    b_rows = [r for r in rows if r["route"] == ROUTE_ALERCE]
    matched_names = {r["tns_name"] for r in rows}
    even = sum(1 for r in rows if r["hash_route"] == "test")
    odd = sum(1 for r in rows if r["hash_route"] == "train")
    ia = sum(1 for r in rows if r["final_class_ternary"] == "snia")
    return {
        "candidates": n_candidates,
        "matched_rows": len(rows),
        "matched_names": len(matched_names),
        "route_a_rows": len(a_rows),
        "route_b_rows": len(b_rows),
        "matched_objnames_route_a": len(matched_objnames_a),
        "unmatched_names": n_candidates - len(matched_names),
        "hash_even_test": even,
        "hash_odd_train": odd,
        "ia_count": ia,
        "failures": len(failures),
    }


def print_summary(summary: dict) -> None:
    print("\n=== TNS backlog sweep summary ===")
    print(f"  filtered candidates          : {summary['candidates']}")
    print(f"  matched rows (object_ids)    : {summary['matched_rows']}")
    print(f"  distinct TNS names matched   : {summary['matched_names']}")
    print(f"    via Route A (lasair_xm)    : {summary['route_a_rows']}")
    print(f"    via Route B (alerce_cone)  : {summary['route_b_rows']}")
    print(f"  unmatched TNS names          : {summary['unmatched_names']}")
    print(f"  hash split even(test)/odd(train): "
          f"{summary['hash_even_test']}/{summary['hash_odd_train']}")
    print(f"  Ia-like matches (snia)       : {summary['ia_count']}")
    print(f"  Route B failures             : {summary['failures']}")


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tns-bulk", default="data/truth/tns_public.parquet")
    ap.add_argument("--max-dec", type=float, default=15.0)
    ap.add_argument("--since", default="2025-06-01")
    ap.add_argument("--max-sep-arcsec", type=float, default=2.0)
    ap.add_argument("--out", default="data/truth/tns_backlog_matches.parquet")
    ap.add_argument("--limit", type=int, default=None, help="Smoke: cap candidates.")
    ap.add_argument("--cache-dir", default="data/crossmatch/backlog_sweep_cache")
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=100)
    ap.add_argument("--skip-route-b", action="store_true",
                    help="Route A only (no ALeRCE cone calls).")
    args = ap.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    xm_cache = cache_dir / "lasair_xm_pages"
    cone_cache = cache_dir / "alerce_cone"

    df = pd.read_parquet(args.tns_bulk)
    filtered = filter_tns_candidates(df, max_dec=args.max_dec, since=args.since)
    candidates = tns_candidates_from_df(filtered)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    print(f"Filtered TNS candidates: {len(candidates)}")

    # Route A: Lasair crossmatch_tns join.
    print("Route A: harvesting Lasair LSST crossmatch_tns join ...")
    xm_records = harvest_xm_records(
        cache_dir=xm_cache, page_size=args.page_size, max_pages=args.max_pages
    )
    xm_index = build_xm_index(xm_records)
    print(f"  xm records: {len(xm_records)}, distinct names: {len(xm_index)}")
    raw_a, matched_a = match_route_a(candidates, xm_index, max_sep=args.max_sep_arcsec)
    print(f"  Route A raw matches: {len(raw_a)} (objnames matched: {len(matched_a)})")

    # Route B: ALeRCE cone for the unmatched remainder.
    raw_b: list[dict] = []
    failures: list[dict] = []
    remaining = [c for c in candidates if c["objname"] not in matched_a]
    if args.skip_route_b:
        print(f"Route B: skipped ({len(remaining)} unmatched candidates).")
    else:
        print(f"Route B: cone-searching {len(remaining)} unmatched candidates ...")
        from alerce.core import Alerce

        fetch = _make_alerce_fetch(Alerce())
        raw_b, failures = run_route_b(
            remaining, max_sep=args.max_sep_arcsec, cache_dir=cone_cache, fetch=fetch
        )
        print(f"  Route B raw matches: {len(raw_b)}, failures: {len(failures)}")

    rows = finalize_matches(raw_a + raw_b, max_sep=args.max_sep_arcsec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_parquet(out, index=False)
    print(f"OK wrote {len(out_df)} rows -> {out}")

    summary = summarize(
        n_candidates=len(candidates),
        rows=rows,
        matched_objnames_a=matched_a,
        failures=failures,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
