#!/usr/bin/env python3
"""Harvest LSST<->ZTF associations via the INVERTED Lasair direction.

The ZTF->LSST conesearch direction fails (measured 0/25 Lasair, 0/25 Fink,
2/233 name-match; fusion_v11 spec §1). The working mechanism is one paged
Lasair ``objects,crossmatch_tns`` join: parse ZTF ids straight out of the TNS
``disc_int_name`` field (free TNS<->ZTF mapping). The cone endpoint is a
fallback only, for TNS-matched names that carry no ``disc_int_name``.

Outputs:
  * association CSV in the ``load_lsst_ztf_associations`` schema
    (lsst_object_id, ztf_object_id, match_status="matched", sep_arcsec,
     association_kind, association_source="lasair_crossmatch_tns", match_count)
  * (optional) spec-label companion CSV with label_source='ztf_assoc_spec'
    for TNS-typed matches, consumed by build_truth_lsst_live.py.

No network in the pure helpers (parse_ztf_ids / record_to_association_rows /
records_to_spec_rows) — those are unit-tested with synthetic records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

from scripts.crossmatch_lsst_to_ztf import separation_arcsec

ASSOCIATION_SOURCE = "lasair_crossmatch_tns"
ASSOCIATION_SOURCE_CONE = "lasair_cone_fallback"

# Column order matching the load_lsst_ztf_associations consumer schema.
ASSOCIATION_COLUMNS = [
    "lsst_object_id", "ztf_object_id", "match_status", "sep_arcsec",
    "association_kind", "association_source", "match_count",
]

SPEC_COLUMNS = [
    "lsst_object_id", "ztf_object_id", "tns_name", "tns_prefix", "tns_type",
    "final_class_ternary", "label_source", "disc_date", "sep_arcsec",
]


# ------------------------------------------------------------------ #
# Pure helpers (unit-tested, no network)                              #
# ------------------------------------------------------------------ #


def parse_ztf_ids(disc_int_name: Any) -> list[str]:
    """Extract ZTF ids from a TNS ``disc_int_name`` / ``internal_names`` field.

    TNS packs internal names as a ``;`` or ``,`` separated list. Return the
    ZTF-prefixed ones, de-duplicated, order-preserving.
    """
    if disc_int_name is None:
        return []
    text = str(disc_int_name).strip()
    if not text or text.lower() == "nan":
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in text.replace(",", ";").split(";"):
        tok = token.strip()
        if tok.upper().startswith("ZTF") and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _get(record: dict, *keys: str) -> Any:
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _record_sep_arcsec(record: dict) -> float | None:
    """Client-side sep between the LSST object and its TNS crossmatch."""
    obj_ra = _to_float(_get(record, "ra", "objects.ra", "obj_ra"))
    obj_dec = _to_float(_get(record, "decl", "dec", "objects.decl", "obj_decl"))
    tns_ra = _to_float(_get(record, "crossmatch_tns.ra", "tns_ra", "cm_ra"))
    tns_dec = _to_float(_get(record, "crossmatch_tns.decl", "tns_decl", "cm_decl"))
    if None in (obj_ra, obj_dec, tns_ra, tns_dec):
        return None
    return separation_arcsec(obj_ra, obj_dec, tns_ra, tns_dec)


def _record_lsst_id(record: dict) -> str | None:
    val = _get(record, "diaObjectId", "objects.diaObjectId", "objectId")
    if val is None:
        return None
    text = str(val).strip()
    return text or None


def _record_disc_int_name(record: dict) -> Any:
    return _get(record, "disc_int_name", "crossmatch_tns.disc_int_name",
                "internal_names")


def record_to_association_rows(
    record: dict,
    *,
    association_source: str = ASSOCIATION_SOURCE,
    association_kind: str = "disc_int_name",
) -> list[dict]:
    """Map one Lasair ``objects,crossmatch_tns`` record to association rows.

    One row per ZTF id parsed from ``disc_int_name``. ``sep_arcsec`` is the
    LSST<->TNS separation (the 2" cut is applied by the downstream loader).
    """
    lsst_id = _record_lsst_id(record)
    if not lsst_id:
        return []
    ztf_ids = parse_ztf_ids(_record_disc_int_name(record))
    if not ztf_ids:
        return []
    sep = _record_sep_arcsec(record)
    rows: list[dict] = []
    for ztf_id in ztf_ids:
        rows.append({
            "lsst_object_id": lsst_id,
            "ztf_object_id": ztf_id,
            "match_status": "matched",
            "sep_arcsec": sep,
            "association_kind": association_kind,
            "association_source": association_source,
            "match_count": len(ztf_ids),
        })
    return rows


_TNS_IA_PREFIXES = ("SN Ia", "SN Iax")


def _tns_type_to_ternary(tns_type: Any):
    from debass_meta.access.tns import map_tns_type_to_ternary

    return map_tns_type_to_ternary(str(tns_type) if tns_type is not None else None)


def records_to_spec_rows(
    records: list[dict], *, max_sep_arcsec: float = 2.0
) -> list[dict]:
    """Spec-labeled LSST rows (label_source='ztf_assoc_spec') from TNS types.

    Spec LABELS are a precision path: rows separated by more than
    ``max_sep_arcsec`` from the TNS position are dropped (the association CSV
    stays inclusive — leak guards prefer recall — but a >2″ crossmatch must
    never mint a spectroscopic label). ``sep=None`` rows are kept (legacy
    cached pages without aliased coordinates).
    """
    out: list[dict] = []
    for record in records:
        lsst_id = _record_lsst_id(record)
        if not lsst_id:
            continue
        tns_type = _get(record, "type", "crossmatch_tns.type", "tns_type")
        ternary = _tns_type_to_ternary(tns_type)
        if ternary is None:
            continue
        sep = _record_sep_arcsec(record)
        if sep is not None and sep > max_sep_arcsec:
            continue
        ztf_ids = parse_ztf_ids(_record_disc_int_name(record))
        out.append({
            "lsst_object_id": lsst_id,
            "ztf_object_id": ztf_ids[0] if ztf_ids else None,
            "tns_name": _get(record, "tns_name", "crossmatch_tns.tns_name"),
            "tns_prefix": _get(record, "tns_prefix", "crossmatch_tns.tns_prefix"),
            "tns_type": tns_type,
            "final_class_ternary": ternary,
            "label_source": "ztf_assoc_spec",
            "disc_date": _get(record, "disc_date", "crossmatch_tns.disc_date"),
            "sep_arcsec": sep,
        })
    return out


def records_to_associations(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        rows.extend(record_to_association_rows(record))
    return rows


# ------------------------------------------------------------------ #
# Live harvest (network; cached + resumable)                          #
# ------------------------------------------------------------------ #

# objects.ra/crossmatch_tns.ra (and .decl) MUST be aliased: Lasair /query/
# flattens qualified names to bare keys, so without AS both tables collide on
# one "ra"/"decl" key and the client-side sep cut sees NaN (live-verified
# 2026-07-05: aliases are honoured and return distinct coordinates).
_SELECTED = (
    "objects.diaObjectId, objects.ra AS obj_ra, objects.decl AS obj_decl, "
    "objects.firstDiaSourceMjdTai, objects.lastDiaSourceMjdTai, "
    "objects.nDiaSources, crossmatch_tns.ra AS cm_ra, crossmatch_tns.decl AS cm_decl, "
    "crossmatch_tns.tns_name, crossmatch_tns.tns_prefix, crossmatch_tns.type, "
    "crossmatch_tns.z, crossmatch_tns.disc_int_name, crossmatch_tns.disc_date, "
    "crossmatch_tns.lastmodified_date"
)


def harvest_records(
    *,
    cache_dir: Path,
    page_size: int = 1000,
    max_pages: int = 50,
    # ORDER BY makes offset pagination deterministic — without it MySQL may
    # repeat/skip rows across pages (duplicate truth object_ids downstream).
    conditions: str = "1=1 ORDER BY objects.diaObjectId",
) -> list[dict]:
    """Paged Lasair ``objects,crossmatch_tns`` harvest, one JSON page per file.

    Resumable: a cached page is loaded from disk; a short page (< page_size)
    or an empty page ends the harvest.
    """
    from debass_meta.access.lasair import LasairAdapter

    cache_dir.mkdir(parents=True, exist_ok=True)
    adapter = LasairAdapter()
    records: list[dict] = []
    # Cache key includes a hash of the full query so pages cached under an
    # older column list (the pre-alias collided schema) or an older/unordered
    # conditions string are never reused.
    sel_tag = hashlib.sha1(f"{_SELECTED}|{conditions}".encode()).hexdigest()[:8]
    for page in range(max_pages):
        page_path = cache_dir / f"page_{sel_tag}_{page:04d}.json"
        if page_path.exists():
            rows = json.loads(page_path.read_text())
        else:
            rows = adapter._query_api(
                selected=_SELECTED,
                tables="objects,crossmatch_tns",
                conditions=conditions,
                limit=page_size,
                offset=page * page_size,
                survey="lsst",
            )
            page_path.write_text(json.dumps(rows))
        records.extend(rows)
        print(f"  page {page}: {len(rows)} rows (total {len(records)})")
        if len(rows) < page_size:
            break
    return records


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/crossmatch/lsst_to_ztf.csv",
                    help="Association CSV (load_lsst_ztf_associations schema). This is "
                         "the canonical path every v11 consumer reads (builder "
                         "--association-csv default, run_fusion_v11_*.sh, "
                         "train_seq_encoder.py) — the harvest is the WORKING "
                         "association mechanism (spec §8 dev #3); the legacy "
                         "conesearch crossmatch_lsst_to_ztf.py yields ~0.")
    ap.add_argument("--spec-out", default="data/truth/ztf_assoc_spec.csv",
                    help="Companion spec-label CSV (label_source=ztf_assoc_spec)")
    ap.add_argument("--cache-dir", default="data/crossmatch/lasair_harvest_cache")
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--conditions", default="1=1")
    args = ap.parse_args()

    records = harvest_records(
        cache_dir=Path(args.cache_dir),
        page_size=args.page_size,
        max_pages=args.max_pages,
        conditions=args.conditions,
    )
    print(f"Harvested {len(records)} objects,crossmatch_tns records")

    assoc_rows = records_to_associations(records)
    assoc_df = pd.DataFrame(assoc_rows, columns=ASSOCIATION_COLUMNS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    assoc_df.to_csv(out, index=False)
    print(f"OK wrote {len(assoc_df)} association rows -> {out}")

    spec_rows = records_to_spec_rows(records)
    spec_df = pd.DataFrame(spec_rows, columns=SPEC_COLUMNS)
    spec_out = Path(args.spec_out)
    spec_out.parent.mkdir(parents=True, exist_ok=True)
    spec_df.to_csv(spec_out, index=False)
    print(f"OK wrote {len(spec_df)} ztf_assoc_spec rows -> {spec_out}")


if __name__ == "__main__":
    main()
