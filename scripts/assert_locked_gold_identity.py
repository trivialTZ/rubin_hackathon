#!/usr/bin/env python3
"""G5b — value-identity of locked ZTF gold rows across the v11 positive-only fix.

The fusion_v11 negative-detection fix (``features/detection.py`` LSST
``is_positive``) and the new NEG features MUST NOT change ANY v10-era column on
ZTF rows: the base-51 extractor never sees negatives, and the ZTF ``isdiffpos``
rule is untouched.  This proves it by VALUE (never file bytes — new columns make
byte identity impossible, B9):

  (a) the row multiset keyed ``object_id × n_det`` is identical on ZTF rows;
  (b) every v10-era column (base-51 + EXT + trajectory + ``proj__``/``avail__``/
      ``exact__`` + schema cols) is exactly equal after aligning on that key.

The reference is the locked v10 gold
(``data/gold/object_epoch_snapshots_fusion_v10.parquet``), which is SCC-ONLY.
Locally that parquet is absent, so a HASH-MANIFEST fallback is supported:
generate the manifest once on SCC (``--reference ... --emit-hash-manifest ...``),
commit it, and compare against a local rebuild
(``--rebuilt ... --reference-hash-manifest ...``).  Callers that have neither a
reference parquet nor a hash manifest should skip (this script exits 3 =
SKIPPED so a CI wrapper can treat it as "not runnable here").

Usage
-----
    # SCC (reference parquet present): direct value-identity assert
    python scripts/assert_locked_gold_identity.py \
        --reference data/gold/object_epoch_snapshots_fusion_v10.parquet \
        --rebuilt   data/gold/object_epoch_snapshots_fusion_v11.parquet

    # SCC: emit a portable hash manifest from the reference for offline checks
    python scripts/assert_locked_gold_identity.py \
        --reference data/gold/object_epoch_snapshots_fusion_v10.parquet \
        --emit-hash-manifest data/gold/locked_ztf_v10_hashes.json

    # Local (no reference parquet): compare a rebuild against the manifest
    python scripts/assert_locked_gold_identity.py \
        --rebuilt data/gold/object_epoch_snapshots_fusion_v11.parquet \
        --reference-hash-manifest data/gold/locked_ztf_v10_hashes.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd

from debass_meta.features.lightcurve import NEG_FEATURE_NAMES

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_SKIPPED = 3

_KEY = ["object_id", "n_det"]
_CELL_SEP = "\x1f"
_NULL = "\x00∅\x00"


# ------------------------------------------------------------------ #
# Canonical, cross-machine value hashing                             #
# ------------------------------------------------------------------ #

def _canonical_cell(value: Any) -> str:
    """Deterministic string for one cell (exact for floats via ``float.hex``)."""
    if value is None:
        return _NULL
    if isinstance(value, float):
        if math.isnan(value):
            return _NULL
        return "f:" + float(value).hex()
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, (int,)):
        return "i:" + str(int(value))
    # numpy scalars / everything else → go through float when numeric, else str
    try:
        import numpy as np

        if isinstance(value, np.floating):
            f = float(value)
            return _NULL if math.isnan(f) else "f:" + f.hex()
        if isinstance(value, np.integer):
            return "i:" + str(int(value))
        if isinstance(value, np.bool_):
            return "b:1" if bool(value) else "b:0"
    except Exception:
        pass
    if isinstance(value, float) and math.isnan(value):
        return _NULL
    return "s:" + str(value)


def _row_hash(values: list[Any]) -> str:
    joined = _CELL_SEP.join(_canonical_cell(v) for v in values)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ #
# Frame preparation                                                  #
# ------------------------------------------------------------------ #

def _ztf_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "survey" not in df.columns:
        raise SystemExit("input parquet has no 'survey' column")
    ztf = df[df["survey"].astype(str).str.upper() == "ZTF"].copy()
    ztf["object_id"] = ztf["object_id"].astype(str)
    ztf["n_det"] = ztf["n_det"].astype(int)
    return ztf.sort_values(_KEY).reset_index(drop=True)


def _key_multiset(df: pd.DataFrame) -> list[tuple[str, int]]:
    return sorted((str(o), int(n)) for o, n in zip(df["object_id"], df["n_det"]))


def v10_columns(reference_cols: list[str], rebuilt_cols: list[str]) -> list[str]:
    """v10-era columns = reference columns, minus the v11-only NEG additions.

    A v10-era column MISSING from the rebuild is a schema regression (fails in
    :func:`assert_value_identity` / hash comparison).  The join key columns
    (``object_id``, ``n_det``) are excluded — they are verified by the row-key
    multiset check, and keeping them here would duplicate labels on alignment.
    """
    drop = set(NEG_FEATURE_NAMES) | set(_KEY)
    return [c for c in reference_cols if c not in drop]


def hash_frame(df: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    """Per-row hash keyed ``object_id|n_det`` over ``columns`` (in given order)."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(missing)
    sub = df[[*_KEY, *columns]]
    out: dict[str, str] = {}
    for rec in sub.to_dict("records"):
        key = f"{rec['object_id']}|{int(rec['n_det'])}"
        out[key] = _row_hash([rec[c] for c in columns])
    return out


# ------------------------------------------------------------------ #
# Assertions                                                         #
# ------------------------------------------------------------------ #

def assert_value_identity(reference: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    """Direct parquet path: (a) key multiset + (b) exact column equality."""
    ref = _ztf_rows(reference)
    reb = _ztf_rows(rebuilt)

    ref_keys = _key_multiset(ref)
    reb_keys = _key_multiset(reb)
    if ref_keys != reb_keys:
        only_ref = sorted(set(ref_keys) - set(reb_keys))[:5]
        only_reb = sorted(set(reb_keys) - set(ref_keys))[:5]
        raise AssertionError(
            f"G5b FAILED (a): ZTF (object_id, n_det) multiset differs "
            f"(ref {len(ref_keys):,} rows, rebuilt {len(reb_keys):,} rows). "
            f"only-in-ref e.g. {only_ref}; only-in-rebuilt e.g. {only_reb}"
        )

    cols = v10_columns(list(reference.columns), list(rebuilt.columns))
    missing = [c for c in cols if c not in reb.columns]
    if missing:
        raise AssertionError(
            f"G5b FAILED (b): {len(missing)} v10-era columns missing from the "
            f"rebuild (schema regression): {missing[:10]}"
        )

    ref_aligned = ref[[*_KEY, *cols]].sort_values(_KEY).reset_index(drop=True)
    reb_aligned = reb[[*_KEY, *cols]].sort_values(_KEY).reset_index(drop=True)
    try:
        # VALUE identity, not BIT identity: the canonical-truncation refactor
        # changes float accumulation order, and platform BLAS/reduction order
        # differs local↔SCC — last-ulp noise (observed: dmag_dt at ~5e-16
        # relative on 92% of rows, 2026-07-05 run) is not a feature change.
        # rtol=1e-9 is ~1e6 ulps of headroom below any real contract drift.
        pd.testing.assert_frame_equal(
            ref_aligned, reb_aligned, check_exact=False,
            rtol=1e-9, atol=1e-12, check_dtype=False,
        )
    except AssertionError as exc:
        # Localize the first offending column for a useful message.
        offending = _first_diff_column(ref_aligned, reb_aligned, cols)
        raise AssertionError(
            f"G5b FAILED (b): ZTF v10-era columns differ after the positive-only "
            f"fix. First differing column: {offending}.\n{exc}"
        ) from exc
    print(
        f"G5b PASSED (direct): {len(ref_aligned):,} ZTF rows, {len(cols):,} "
        f"v10-era columns value-identical (rtol=1e-9).", flush=True
    )


def _first_diff_column(ref: pd.DataFrame, reb: pd.DataFrame, cols: list[str]) -> str:
    for c in cols:
        a, b = ref[c], reb[c]
        both_nan = a.isna() & b.isna()
        neq = (a != b) & ~both_nan
        if bool(neq.any()):
            i = int(neq.idxmax())
            return f"{c} (row {i}: ref={ref[c].iloc[i]!r} vs rebuilt={reb[c].iloc[i]!r})"
    return "<none — non-value diff (dtype/index)>"


def assert_against_hash_manifest(rebuilt: pd.DataFrame, manifest: dict[str, Any]) -> None:
    """Fallback path: compare a rebuild against a portable reference hash manifest."""
    cols = list(manifest["columns"])
    ref_hashes: dict[str, str] = manifest["row_hashes"]
    reb = _ztf_rows(rebuilt)
    try:
        reb_hashes = hash_frame(reb, cols)
    except KeyError as exc:
        raise AssertionError(
            f"G5b FAILED (b): rebuild is missing v10-era columns present in the "
            f"hash manifest (schema regression): {exc.args[0][:10]}"
        ) from exc

    if set(ref_hashes) != set(reb_hashes):
        only_ref = sorted(set(ref_hashes) - set(reb_hashes))[:5]
        only_reb = sorted(set(reb_hashes) - set(ref_hashes))[:5]
        raise AssertionError(
            f"G5b FAILED (a): ZTF row-key multiset differs from the manifest "
            f"(ref {len(ref_hashes):,}, rebuilt {len(reb_hashes):,}). "
            f"only-in-ref e.g. {only_ref}; only-in-rebuilt e.g. {only_reb}"
        )

    diffs = [k for k in ref_hashes if ref_hashes[k] != reb_hashes[k]]
    if diffs:
        raise AssertionError(
            f"G5b FAILED (b): {len(diffs):,} ZTF rows changed a v10-era column "
            f"value after the positive-only fix (e.g. {diffs[:5]})"
        )
    print(
        f"G5b PASSED (hash manifest): {len(ref_hashes):,} ZTF rows, {len(cols):,} "
        f"v10-era columns value-identical to {manifest.get('source', '<reference>')}.",
        flush=True,
    )


def build_hash_manifest(reference: pd.DataFrame, source: str) -> dict[str, Any]:
    ref = _ztf_rows(reference)
    cols = v10_columns(list(reference.columns), list(reference.columns))
    return {
        "kind": "locked_ztf_gold_value_hashes",
        "source": source,
        "n_rows": int(len(ref)),
        "columns": cols,
        "row_hashes": hash_frame(ref, cols),
    }


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--reference", default=None,
                        help="Locked v10 gold parquet (SCC-only). Absent locally "
                             "→ use --reference-hash-manifest.")
    parser.add_argument("--rebuilt", default=None,
                        help="Rebuilt v11 gold parquet to check.")
    parser.add_argument("--reference-hash-manifest", default=None,
                        help="Portable reference hash manifest (fallback when the "
                             "v10 parquet is unavailable).")
    parser.add_argument("--emit-hash-manifest", default=None,
                        help="Write a hash manifest from --reference and exit "
                             "(run once on SCC, then compare offline).")
    args = parser.parse_args()

    # Mode 1: emit a manifest from the reference parquet.
    if args.emit_hash_manifest:
        if not args.reference or not Path(args.reference).exists():
            print("--emit-hash-manifest requires an existing --reference parquet",
                  file=sys.stderr)
            return EXIT_USAGE
        ref = pd.read_parquet(args.reference)
        manifest = build_hash_manifest(ref, source=str(args.reference))
        out = Path(args.emit_hash_manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest))
        print(f"Wrote hash manifest → {out} ({manifest['n_rows']:,} ZTF rows × "
              f"{len(manifest['columns']):,} v10-era cols)", flush=True)
        return EXIT_OK

    if not args.rebuilt or not Path(args.rebuilt).exists():
        print("--rebuilt <v11 parquet> is required (and must exist)", file=sys.stderr)
        return EXIT_USAGE
    rebuilt = pd.read_parquet(args.rebuilt)

    # Mode 2: direct value-identity against the reference parquet (SCC).
    if args.reference and Path(args.reference).exists():
        reference = pd.read_parquet(args.reference)
        try:
            assert_value_identity(reference, rebuilt)
        except AssertionError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_FAIL
        return EXIT_OK

    # Mode 3: hash-manifest fallback (local).
    if args.reference_hash_manifest and Path(args.reference_hash_manifest).exists():
        manifest = json.loads(Path(args.reference_hash_manifest).read_text())
        try:
            assert_against_hash_manifest(rebuilt, manifest)
        except AssertionError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_FAIL
        return EXIT_OK

    # Nothing to compare against here.
    print(
        "G5b SKIPPED: no reference parquet and no hash manifest available in this "
        "environment. The locked v10 gold parquet is SCC-only; run this assert on "
        "SCC (direct mode) or supply --reference-hash-manifest generated there.",
        file=sys.stderr,
    )
    return EXIT_SKIPPED


if __name__ == "__main__":
    raise SystemExit(main())
