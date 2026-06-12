#!/usr/bin/env python3
"""Build the fusion_v8 long-format Stage-A helpfulness table.

One row per (gold snapshot row × available expert), exactly like
data/gold/expert_helpfulness.parquet — the row construction (targets
``is_topclass_correct`` / ``is_sn`` / ``is_helpful_for_follow_proxy``,
per-expert ``proj__``/``avail__``/``exact__`` blocks, base-51 LC context)
is REUSED by importing scripts/build_expert_helpfulness.py and running its
builder unchanged over object-chunks of the fusion snapshot.  This script
then augments each chunk with the extra Stage-A context the pooled trust
model (src/debass_meta/models/pooled_trust.py) needs:

  - ``survey`` + the 50 EXT lightcurve features (generic LC context),
  - the 6 cross-expert trajectory aggregates ``traj_x__*``,
  - 10 generic own-trajectory slots ``own_traj__{n,last,mean,std,min,max,
    slope,delta,volatility,ewm}``: the row's OWN expert's
    ``traj__<san>__<stat>`` value (NaN for non-trajectory experts) — the
    shared column space lets pooled-LGBM split structure transfer across
    experts.

No-leakage argument: every added column is copied from the SAME
(object_id, n_det) gold row the helpfulness row was derived from; the gold
row itself is as-of alert_jd by construction, so no future information is
introduced.  Targets are label-derived and live in y-space only (Stage A
asserts no ``q__``/``trust_source__`` columns enter X).

Output: data/gold/expert_helpfulness_fusion_v8.parquet

Usage
-----
    python scripts/build_helpfulness_fusion.py
    python scripts/build_helpfulness_fusion.py --smoke
    python scripts/build_helpfulness_fusion.py --parity-check
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd

from debass_meta.projectors import ALL_EXPERT_KEYS, sanitize_expert_key

try:
    from debass_meta.features.lightcurve_ext import EXT_FEATURE_NAMES  # type: ignore
except ImportError:  # pragma: no cover - exercised only pre-integration
    EXT_FEATURE_NAMES: list[str] = []

_DEFAULT_SNAPSHOTS = "data/gold/object_epoch_snapshots_fusion_v8.parquet"
_DEFAULT_OUTPUT = "data/gold/expert_helpfulness_fusion_v8.parquet"
_SMOKE_N_OBJECTS = 200

# Per-expert trajectory statistic suffixes (pinned TRAJ_FEATURE_NAMES contract:
# traj__<san>__{n,last,mean,std,min,max,slope,delta,volatility,ewm}).
TRAJ_STATS = ["n", "last", "mean", "std", "min", "max", "slope",
              "delta", "volatility", "ewm"]

# v6e2 helpfulness base rates for the trust-target parity check
# (spec correction #4e; tolerance ±0.005 on matched rows).
_PARITY_EXPERTS = ("fink/snn", "fink/rf_ia", "supernnova")
_PARITY_TOLERANCE = 0.005


def _load_legacy_builder():
    """Import scripts/build_expert_helpfulness.py as a module (scripts/ is not
    a package on every deployment path, so load it by file location)."""
    path = Path(__file__).resolve().parent / "build_expert_helpfulness.py"
    spec = importlib.util.spec_from_file_location("_legacy_build_expert_helpfulness", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy helpfulness builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_traj_columns(columns: list[str]) -> dict[str, dict[str, str]]:
    """Map sanitized expert key → {stat: column} for traj__<san>__<stat> cols.

    The sanitized key itself contains '__' (e.g. fink__snn), so the stat is
    the segment after the FINAL '__' and must be a known TRAJ_STATS suffix.
    """
    by_san: dict[str, dict[str, str]] = {}
    for col in columns:
        if not col.startswith("traj__"):
            continue
        body = col[len("traj__"):]
        san, sep, stat = body.rpartition("__")
        if not sep or stat not in TRAJ_STATS:
            continue
        by_san.setdefault(san, {})[stat] = col
    return by_san


def build_helpfulness_fusion(
    snapshot_path: Path,
    output_path: Path,
    *,
    chunk_objects: int = 2000,
    limit_objects: int | None = None,
) -> Path:
    legacy = _load_legacy_builder()
    t0 = time.time()

    snap = pd.read_parquet(snapshot_path)
    snap["object_id"] = snap["object_id"].astype(str)
    object_ids = list(dict.fromkeys(snap["object_id"]))
    if limit_objects is not None:
        object_ids = object_ids[:limit_objects]
        snap = snap[snap["object_id"].isin(set(object_ids))]
    print(f"fusion_v8 helpfulness builder: {len(object_ids):,} objects, "
          f"{len(snap):,} gold rows", flush=True)

    # Generic Stage-A context columns added on top of the legacy builder's
    # alert_jd + base-51 block.
    ext_cols = [c for c in EXT_FEATURE_NAMES if c in snap.columns]
    traj_x_cols = [c for c in snap.columns if c.startswith("traj_x__")]
    extra_generic = [c for c in ["survey", *ext_cols, *traj_x_cols] if c in snap.columns]
    traj_by_san = _parse_traj_columns(list(snap.columns))
    all_traj_cols = sorted({c for cols in traj_by_san.values() for c in cols.values()})
    known_sans = {sanitize_expert_key(k) for k in ALL_EXPERT_KEYS}
    unknown = sorted(set(traj_by_san) - known_sans)
    if unknown:
        print(f"  WARNING: trajectory columns for unregistered experts ignored: {unknown}",
              flush=True)
    if not ext_cols:
        print("  WARNING: no EXT feature columns in snapshot "
              "(lightcurve_ext stubbed at snapshot build time?)", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    chunks = [object_ids[i:i + chunk_objects] for i in range(0, len(object_ids), chunk_objects)]
    for ci, chunk in enumerate(chunks):
        chunk_snap = snap[snap["object_id"].isin(set(chunk))]
        tmp_in = output_path.parent / f"{output_path.stem}.__chunk{ci}_in.parquet"
        tmp_out = output_path.parent / f"{output_path.stem}.__chunk{ci}_base.parquet"
        try:
            chunk_snap.to_parquet(tmp_in, index=False)
            # REUSE: the v6e2 row construction (targets + per-expert blocks)
            # runs unchanged — only the context join below is new.
            legacy.build_expert_helpfulness(tmp_in, tmp_out)
            base = pd.read_parquet(tmp_out)
        finally:
            tmp_in.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)
        if len(base) == 0:
            continue
        base["object_id"] = base["object_id"].astype(str)
        base["n_det"] = base["n_det"].astype(int)

        ctx_cols = ["object_id", "n_det", *extra_generic, *all_traj_cols]
        ctx = chunk_snap[[c for c in ctx_cols if c in chunk_snap.columns]].copy()
        ctx["n_det"] = ctx["n_det"].astype(int)
        merged = base.merge(ctx, on=["object_id", "n_det"], how="left",
                            validate="many_to_one")

        # Generic own-trajectory slots: the row's own expert's traj stats.
        for stat in TRAJ_STATS:
            merged[f"own_traj__{stat}"] = np.nan
        for san, stat_cols in traj_by_san.items():
            if san not in known_sans:
                continue
            expert_key = next(k for k in ALL_EXPERT_KEYS if sanitize_expert_key(k) == san)
            mask = merged["expert_key"] == expert_key
            if not bool(mask.any()):
                continue
            for stat, col in stat_cols.items():
                if col in merged.columns:
                    merged.loc[mask, f"own_traj__{stat}"] = merged.loc[mask, col]
        # Per-expert traj columns are dropped from the long table: the pooled
        # model must only see the OWN-expert slots (shared column space).
        merged = merged.drop(columns=[c for c in all_traj_cols if c in merged.columns])
        parts.append(merged)
        print(f"  chunk {ci + 1}/{len(chunks)}: {len(base):,} helpfulness rows "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["object_id", "n_det", "alert_jd", "expert_key"]
    )

    # Loud failures (spec coding standards).
    forbidden = [c for c in out.columns if c.startswith("q__") or c.startswith("trust_source__")]
    assert not forbidden, f"Stage-A table contains trust-output columns: {forbidden}"
    if len(out) > 0:
        dup = out.duplicated(subset=["object_id", "n_det", "expert_key"])
        assert not bool(dup.any()), (
            f"{int(dup.sum())} duplicate (object_id, n_det, expert_key) rows"
        )

    out.to_parquet(output_path, index=False)
    print(f"Wrote fusion_v8 helpfulness → {output_path} "
          f"({len(out):,} rows × {len(out.columns):,} cols)", flush=True)
    if len(out) > 0:
        summary = (
            out.groupby("expert_key")["is_topclass_correct"]
            .agg(n="size", base_rate="mean")
        )
        print(summary.to_string(), flush=True)
    return output_path


def run_parity_check(
    fusion_path: Path,
    v6e2_path: Path,
    *,
    experts: tuple[str, ...] = _PARITY_EXPERTS,
    tolerance: float = _PARITY_TOLERANCE,
) -> None:
    """Trust-target parity: on rows shared with the v6e2 helpfulness table,
    the fusion table's is_topclass_correct base rates must agree within
    *tolerance* (spec correction #4e).  Raises AssertionError on failure.
    """
    fusion = pd.read_parquet(fusion_path)
    old = pd.read_parquet(v6e2_path)
    keys = ["object_id", "n_det", "expert_key"]
    for df in (fusion, old):
        df["object_id"] = df["object_id"].astype(str)
        df["n_det"] = df["n_det"].astype(int)
    merged = fusion.merge(
        old[keys + ["is_topclass_correct"]].rename(
            columns={"is_topclass_correct": "is_topclass_correct_v6e2"}
        ),
        on=keys, how="inner",
    )
    failures: list[str] = []
    for expert in experts:
        rows = merged[merged["expert_key"] == expert]
        rows = rows[rows["is_topclass_correct"].notna()
                    & rows["is_topclass_correct_v6e2"].notna()]
        if len(rows) == 0:
            print(f"  parity [{expert}]: no shared labelled rows — skipped", flush=True)
            continue
        new_rate = float(rows["is_topclass_correct"].astype(float).mean())
        old_rate = float(rows["is_topclass_correct_v6e2"].astype(float).mean())
        status = "OK" if abs(new_rate - old_rate) <= tolerance else "FAIL"
        print(f"  parity [{expert}]: fusion={new_rate:.3f} v6e2={old_rate:.3f} "
              f"n={len(rows):,} → {status}", flush=True)
        if status == "FAIL":
            failures.append(f"{expert}: |{new_rate:.4f} - {old_rate:.4f}| > {tolerance}")
    assert not failures, "trust-target parity check FAILED: " + "; ".join(failures)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--snapshots", default=_DEFAULT_SNAPSHOTS)
    parser.add_argument("--output", default=None,
                        help=f"Output parquet (default {_DEFAULT_OUTPUT})")
    parser.add_argument("--chunk-objects", type=int, default=2000,
                        help="Objects per legacy-builder chunk (bounds memory)")
    parser.add_argument("--smoke", action="store_true",
                        help=f"Process only the first {_SMOKE_N_OBJECTS} objects "
                             f"(output gets a .smoke suffix unless overridden)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N objects (dev shortcut)")
    parser.add_argument("--parity-check", action="store_true",
                        help="After building, check is_topclass_correct base-rate "
                             "parity vs the v6e2 helpfulness table on shared rows")
    parser.add_argument("--v6e2-helpfulness", default="data/gold/expert_helpfulness.parquet")
    args = parser.parse_args()

    dev_run = args.smoke or (args.limit is not None)
    output = Path(args.output) if args.output else Path(
        _DEFAULT_OUTPUT.replace(".parquet", ".smoke.parquet") if dev_run else _DEFAULT_OUTPUT
    )
    limit = args.limit if args.limit is not None else (_SMOKE_N_OBJECTS if args.smoke else None)

    path = build_helpfulness_fusion(
        Path(args.snapshots),
        output,
        chunk_objects=args.chunk_objects,
        limit_objects=limit,
    )
    if args.parity_check:
        v6e2 = Path(args.v6e2_helpfulness)
        if v6e2.exists():
            run_parity_check(path, v6e2)
        else:
            print(f"  parity check skipped: {v6e2} not found", flush=True)


if __name__ == "__main__":
    main()
