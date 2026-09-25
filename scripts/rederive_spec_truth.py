#!/usr/bin/env python3
"""Re-derive the ZTF spectroscopic truth under the B0 fix (fusion_v11 P1).

The current ``data/truth/object_truth.parquet`` force-maps BTS-untyped rows
(``bts_type='-'``) into ``label_quality='spectroscopic'`` + a subtype ternary
(measured: 3,149/5,172 = 61% of the spec nonIa class is untyped filler,
spec §0 B0). Head-2 trained on that reproduces the v11 bug class *inside* the
ZTF spec corpus. This script rebuilds a corrected truth table:

  * typed BTS rows (``bts_type`` not in {'-', ''}) → unchanged.
  * rows already carrying a concrete ``tns_type`` → unchanged (typed via TNS).
  * BTS-untyped filler rows (untyped ``bts_type`` AND empty ``tns_type``):
      - TNS name-join (via ``internal_names`` / BTS ZTF id) resolves a concrete,
        non-ambiguous type → corrected subtype, stays ``spectroscopic`` and the
        ``tns_type`` column is stamped (so G7 in P3 admits the row).
      - otherwise (no TNS type, or generic/ambiguous "SN") → NEW weak tier
        ``label_quality='bts_untyped'`` (ternary None; Head-1 ``is_sn=1`` only),
        mirroring ``tns_untyped`` rows. NEVER ``spectroscopic``+subtype.

Outputs (originals are never overwritten):
  * ``data/truth/object_truth_v11.parquet`` — same 20-column schema.
  * ``data/truth/label_delta_v11.csv``     — every touched row, incl. the
    ``in_locked_test`` flag for the locked ZTF-spec test (delta recorded, never
    silent — spec §0 B0 / §6).

Pure helpers (``is_bts_untyped`` / ``rederive_label`` / ``build_tns_type_lookups``)
carry no I/O and are unit-tested in tests/test_truth_lsst_live.py.
"""
from __future__ import annotations

import argparse
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

from debass_meta.access.tns import is_ambiguous_type, map_tns_type_to_ternary

BTS_UNTYPED_TOKENS = {"", "-", "nan", "none", "null"}
BTS_UNTYPED_QUALITY = "bts_untyped"

DELTA_COLUMNS = [
    "object_id", "in_locked_test", "old_label_quality", "new_label_quality",
    "old_final_class_ternary", "new_final_class_ternary", "old_bts_type",
    "resolved_tns_type", "reason",
]


# ------------------------------------------------------------------ #
# Pure helpers (no I/O; unit-tested)                                  #
# ------------------------------------------------------------------ #


def _clean(value: Any) -> str | None:
    """Trim to a non-empty string, mapping NaN/None/sentinels to None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def is_bts_untyped(bts_type: Any) -> bool:
    """True when a BTS type carries no concrete subtype ('-', '', None, NaN)."""
    if bts_type is None:
        return True
    return str(bts_type).strip().lower() in BTS_UNTYPED_TOKENS


def rederive_label(
    *,
    bts_type: Any,
    existing_tns_type: Any,
    resolved_tns_type: Any,
    orig_quality: Any,
    orig_ternary: Any,
) -> dict:
    """Corrected label for one truth row (B0 rules).

    Returns a dict with ``label_quality``, ``final_class_ternary``,
    ``final_class_raw``, ``resolved_tns_type``, ``changed``, ``reason``.

    Only ``spectroscopic`` rows whose provenance is BTS-untyped *and* lacking a
    concrete ``tns_type`` are ever touched — exactly the rows G7 (P3) would
    otherwise reject. Everything else is returned unchanged.
    """
    unchanged = {
        "label_quality": orig_quality,
        "final_class_ternary": orig_ternary,
        "final_class_raw": None,
        "resolved_tns_type": None,
        "changed": False,
        "reason": "unchanged",
    }
    if _clean(orig_quality) != "spectroscopic":
        return unchanged
    if not is_bts_untyped(bts_type):
        return unchanged  # concrete BTS subtype — legitimately spectroscopic
    if _clean(existing_tns_type) is not None:
        return unchanged  # already typed via TNS — legitimately spectroscopic

    # BTS-untyped filler with no TNS type on the row: try the TNS name join.
    t = _clean(resolved_tns_type)
    if t is not None and not is_ambiguous_type(t):
        tern = map_tns_type_to_ternary(t)
        if tern is not None:
            return {
                "label_quality": "spectroscopic",
                "final_class_ternary": tern,
                "final_class_raw": t,
                "resolved_tns_type": t,
                "changed": True,  # tns_type is now stamped -> provenance added
                "reason": "tns_corrected",
            }
    # Unresolved, or an ambiguous generic "SN": demote to the weak tier.
    return {
        "label_quality": BTS_UNTYPED_QUALITY,
        "final_class_ternary": None,
        "final_class_raw": None,
        "resolved_tns_type": t,
        "changed": True,
        "reason": "bts_untyped_demote",
    }


def _norm_name(name: Any) -> str:
    text = str(name or "").strip()
    for pref in ("SN ", "AT ", "SN", "AT"):
        if text.startswith(pref):
            text = text[len(pref):]
            break
    return text.strip().lower()


def build_tns_type_lookups(tns_df) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Build (ZTF-id → type, normalised-name → type) maps from a TNS bulk frame.

    The TNS ``internal_names`` field carries ZTF ids (';'/',' separated) — the
    free TNS↔ZTF mapping (spec §1). ``type`` may be empty for AT-only objects
    (stored as None so the caller can fall back to a name match).
    """
    ztf_map: dict[str, str | None] = {}
    name_map: dict[str, str | None] = {}
    name_col = "objname" if "objname" in tns_df.columns else "name"
    has_internal = "internal_names" in tns_df.columns
    for _, r in tns_df.iterrows():
        typ = _clean(r.get("type"))
        nm = _clean(r.get(name_col))
        if nm is not None:
            name_map[_norm_name(nm)] = typ
        if has_internal:
            internal = str(r.get("internal_names") or "")
            for tok in internal.replace(",", ";").split(";"):
                z = tok.strip()
                if z.upper().startswith("ZTF"):
                    ztf_map[z] = typ
    return ztf_map, name_map


def resolve_tns_type(
    object_id: Any,
    tns_name: Any,
    ztf_map: dict[str, str | None],
    name_map: dict[str, str | None],
) -> str | None:
    """Resolve a TNS type for a row via ZTF-id join first, then TNS name."""
    oid = _clean(object_id)
    if oid is not None and oid.upper().startswith("ZTF") and oid in ztf_map:
        typ = ztf_map[oid]
        if typ is not None:
            return typ
    nm = _clean(tns_name)
    if nm is not None:
        typ = name_map.get(_norm_name(nm))
        if typ is not None:
            return typ
    # ZTF id present but only an untyped TNS record — still return None.
    return None


# ------------------------------------------------------------------ #
# I/O driver                                                          #
# ------------------------------------------------------------------ #


def load_tns_bulk_any(path: Path):
    """Load a TNS bulk table from parquet (v11 master) or raw CSV (fallback)."""
    import pandas as pd

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    from scripts.crossmatch_tns import _load_tns_bulk_csv

    return _load_tns_bulk_csv(p)


def _bts_type_map(bts_path: Path | None) -> dict[str, Any]:
    """object_id → bts_type from ztf_bts.parquet (fills gaps in object_truth)."""
    if bts_path is None or not Path(bts_path).exists():
        return {}
    import pandas as pd

    df = pd.read_parquet(bts_path)
    if "object_id" not in df.columns or "bts_type" not in df.columns:
        return {}
    out: dict[str, Any] = {}
    for _, r in df.iterrows():
        oid = _clean(r.get("object_id"))
        if oid is not None:
            out[oid] = r.get("bts_type")
    return out


def _load_locked_test_ids(split_path: Path | None) -> set[str]:
    if split_path is None or not Path(split_path).exists():
        return set()
    import json

    data = json.loads(Path(split_path).read_text())
    ids = data.get("test_ids") or []
    return {str(x).strip() for x in ids if str(x).strip()}


def rederive(
    *,
    truth_df,
    tns_df=None,
    bts_map: dict[str, Any] | None = None,
    locked_test_ids: set[str] | None = None,
):
    """Re-derive truth rows; return (corrected_df, delta_df).

    Pure over its inputs (no filesystem access) so it can be unit-tested with
    synthetic frames.
    """
    import pandas as pd

    bts_map = bts_map or {}
    locked_test_ids = locked_test_ids or set()
    if tns_df is not None and len(tns_df) > 0:
        ztf_map, name_map = build_tns_type_lookups(tns_df)
    else:
        ztf_map, name_map = {}, {}

    out = truth_df.copy()
    delta_rows: list[dict] = []

    for idx, row in out.iterrows():
        oid = _clean(row.get("object_id"))
        orig_quality = row.get("label_quality")
        orig_ternary = row.get("final_class_ternary")
        bts_type = row.get("bts_type")
        if is_bts_untyped(bts_type) and oid is not None and oid in bts_map:
            # object_truth bts_type is null for some rows — fall back to BTS.
            bts_type = bts_map[oid]
        existing_tns_type = row.get("tns_type")
        resolved = resolve_tns_type(oid, row.get("tns_name"), ztf_map, name_map)

        res = rederive_label(
            bts_type=bts_type,
            existing_tns_type=existing_tns_type,
            resolved_tns_type=resolved,
            orig_quality=orig_quality,
            orig_ternary=orig_ternary,
        )
        if not res["changed"]:
            continue

        new_quality = res["label_quality"]
        new_ternary = res["final_class_ternary"]
        out.at[idx, "label_quality"] = new_quality
        out.at[idx, "final_class_ternary"] = new_ternary
        out.at[idx, "follow_proxy"] = int(new_ternary == "snia")
        if res["reason"] == "tns_corrected":
            out.at[idx, "tns_type"] = res["resolved_tns_type"]
            out.at[idx, "final_class_raw"] = res["final_class_raw"]
            out.at[idx, "tns_has_spectra"] = True
        else:  # bts_untyped_demote
            out.at[idx, "final_class_raw"] = None
            out.at[idx, "tns_has_spectra"] = False

        delta_rows.append({
            "object_id": oid,
            "in_locked_test": bool(oid in locked_test_ids),
            "old_label_quality": orig_quality,
            "new_label_quality": new_quality,
            "old_final_class_ternary": orig_ternary,
            "new_final_class_ternary": new_ternary,
            "old_bts_type": row.get("bts_type"),
            "resolved_tns_type": res["resolved_tns_type"],
            "reason": res["reason"],
        })

    # Preserve the integer dtype the schema pins on follow_proxy.
    out["follow_proxy"] = pd.to_numeric(
        out["follow_proxy"], errors="coerce"
    ).fillna(0).astype("int64")
    delta_df = pd.DataFrame(delta_rows, columns=DELTA_COLUMNS)
    return out, delta_df


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default="data/truth/object_truth.parquet",
                    help="Input object_truth parquet (NEVER overwritten)")
    ap.add_argument("--bts", default="data/truth/ztf_bts.parquet",
                    help="ztf_bts parquet (fills object_id -> bts_type gaps)")
    ap.add_argument("--tns-bulk", default="data/truth/tns_public.parquet",
                    help="TNS bulk parquet (or CSV) for the name join")
    ap.add_argument("--locked-split", default="data/gold/split_fusion_v10.json",
                    help="Locked v10 split (test_ids flagged in the delta)")
    ap.add_argument("--out", default="data/truth/object_truth_v11.parquet")
    ap.add_argument("--delta-out", default="data/truth/label_delta_v11.csv")
    args = ap.parse_args()

    truth_path = Path(args.truth)
    if not truth_path.exists():
        raise SystemExit(f"missing input truth {truth_path}")
    truth_df = pd.read_parquet(truth_path)
    print(f"loaded {len(truth_df)} truth rows from {truth_path}")

    tns_df = None
    if args.tns_bulk and Path(args.tns_bulk).exists():
        tns_df = load_tns_bulk_any(Path(args.tns_bulk))
        print(f"loaded {len(tns_df)} TNS bulk rows from {args.tns_bulk}")
    else:
        print(f"WARNING: TNS bulk absent ({args.tns_bulk}); "
              f"untyped fillers all demote to bts_untyped (no TNS corrections)")

    bts_map = _bts_type_map(Path(args.bts) if args.bts else None)
    print(f"bts_type fallback map: {len(bts_map)} objects")
    locked_ids = _load_locked_test_ids(Path(args.locked_split) if args.locked_split else None)
    print(f"locked-test ids: {len(locked_ids)}")

    corrected, delta = rederive(
        truth_df=truth_df,
        tns_df=tns_df,
        bts_map=bts_map,
        locked_test_ids=locked_ids,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.resolve() == truth_path.resolve():
        raise SystemExit("refusing to overwrite the input truth file")
    corrected.to_parquet(out_path, index=False)
    delta_path = Path(args.delta_out)
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta.to_csv(delta_path, index=False)

    n_corrected = int((delta["reason"] == "tns_corrected").sum())
    n_demoted = int((delta["reason"] == "bts_untyped_demote").sum())
    n_locked = int(delta["in_locked_test"].sum()) if len(delta) else 0
    print(f"OK wrote {len(corrected)} rows -> {out_path}")
    print(f"   delta rows: {len(delta)} "
          f"(tns_corrected={n_corrected}, bts_untyped_demote={n_demoted}, "
          f"in_locked_test={n_locked}) -> {delta_path}")
    print("   new label_quality counts:",
          corrected["label_quality"].value_counts(dropna=False).to_dict())
    if n_corrected and "new_final_class_ternary" in delta.columns:
        corr = delta[delta["reason"] == "tns_corrected"]
        print("   tns_corrected ternary:",
              corr["new_final_class_ternary"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
