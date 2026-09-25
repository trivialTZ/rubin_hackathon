#!/usr/bin/env python3
"""Training truth for fusion_v12: the v11 truth with the Rubin/LSST labels rebuilt from non-broker sources.

v11 trained its LSST rows on ALeRCE stamp classes (weak, --lsst-candidates) and Lasair Sherlock classes (context),
and both are also model inputs. For v12 the LSST labels are:

  spectroscopic  TNS types: the v11/live truth rows plus the backlog sweep (--lsst-spec)          always win
  context        catalogue labels from scripts/build_lsst_catalog_context.py (--lsst-catalog)      next
  (other tiers)  whatever the base truth had (tns_untyped, stale_xmatch, ...), unchanged

and Sherlock-derived LSST context rows (label_source 'sherlock:*') are dropped unless a catalogue label replaces them.
Ids in --exclude (hold-out objects) get no truth row, so they never reach the gold. ZTF rows are untouched.

The gold build must use --no-lsst-weak, so LSST candidates without a label here drop out of the gold.

Usage (SCC):
  python scripts/build_truth_fusion_v12.py --base data/truth/object_truth_v11_merged.parquet \\
      --lsst-catalog data/truth/lsst_candidates_catalog.parquet \\
      --lsst-spec data/truth/backlog_train_truth_20260924.parquet \\
      --exclude data/truth/backlog_holdout_truth_20260924.parquet \\
      --out data/truth/object_truth_v12_merged.parquet
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.build_truth_lsst_live import dedupe_truth_frame  # noqa: E402

_LSST_ID = re.compile(r"^\d{15,20}$")


def is_lsst_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.match(_LSST_ID)


def _rows_like(base: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Align new rows to the base schema and dtypes (missing columns become NA)."""
    out = rows.reindex(columns=base.columns)
    for c in base.columns:
        try:
            out[c] = out[c].astype(base[c].dtype)
        except (TypeError, ValueError):
            pass
    return out


def build_v12_truth(base: pd.DataFrame, catalog: pd.DataFrame | None, spec: pd.DataFrame | None,
                    exclude: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    base = base.copy()
    base["object_id"] = base["object_id"].astype(str)
    lsst = is_lsst_id(base["object_id"])
    src = base["label_source"].astype(str) if "label_source" in base else pd.Series("", index=base.index)
    sherlock = lsst & (base["label_quality"] == "context") & src.str.startswith("sherlock")
    stats = {"base_rows": len(base), "dropped_sherlock_context": int(sherlock.sum())}
    base = base[~sherlock]

    parts = [base]
    if catalog is not None and len(catalog):
        c = catalog[catalog["final_class_ternary"] == "other"].copy()
        c["object_id"] = c["object_id"].astype(str)
        c = pd.DataFrame({
            "object_id": c["object_id"], "final_class_ternary": "other", "follow_proxy": 0,
            "label_source": c["label_source"], "label_quality": "context",
            "tns_ra": c.get("ra"), "tns_dec": c.get("dec"),
        })
        parts.append(_rows_like(base, c))
        stats["catalog_context_rows"] = len(c)
    if spec is not None and len(spec):
        s = spec.copy()
        s["object_id"] = s["object_id"].astype(str)
        s = pd.DataFrame({
            "object_id": s["object_id"], "final_class_ternary": s["final_class_ternary"],
            "follow_proxy": (s["final_class_ternary"] == "snia").astype(int),
            "label_source": s.get("label_source", "tns_backlog_sweep"), "label_quality": "spectroscopic",
            "tns_name": s.get("tns_name"), "tns_type": s.get("tns_type"), "final_class_raw": s.get("tns_type"),
            "tns_redshift": s.get("tns_redshift"), "tns_has_spectra": True,
        })
        parts.append(_rows_like(base, s))
        stats["spec_rows"] = len(s)
    allrows = pd.concat(parts, ignore_index=True)

    # Priority: spectroscopic > catalogue context > anything else (then the repo's quality order).
    is_cat = allrows["label_source"].astype(str).str.startswith("catalog:")
    rank = np.where(allrows["label_quality"] == "spectroscopic", 0, np.where(is_cat, 1, 2))
    allrows = allrows.assign(_r=rank).sort_values("_r", kind="stable")
    top = allrows.drop_duplicates("object_id", keep="first").drop(columns="_r")
    top = dedupe_truth_frame(top)
    if exclude:
        n0 = len(top)
        top = top[~top["object_id"].isin(exclude)]
        stats["excluded_rows"] = n0 - len(top)
    assert top["object_id"].is_unique
    lt = top[is_lsst_id(top["object_id"])]
    stats["lsst_rows"] = len(lt)
    stats["lsst_by_quality_class"] = (lt.groupby(["label_quality", lt["final_class_ternary"].fillna("none")])
                                      .size().to_dict())
    return top.reset_index(drop=True), stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="v11 merged truth (object_truth_v11_merged.parquet)")
    ap.add_argument("--lsst-catalog", default=None)
    ap.add_argument("--lsst-spec", default=None)
    ap.add_argument("--exclude", default=None, help="parquet/csv with object_id to leave out (hold-out objects)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    read = lambda p: pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p, dtype={"object_id": str})  # noqa: E731
    base = read(a.base)
    ex = set(read(a.exclude)["object_id"].astype(str)) if a.exclude else None
    out, stats = build_v12_truth(base, read(a.lsst_catalog) if a.lsst_catalog else None,
                                 read(a.lsst_spec) if a.lsst_spec else None, ex)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out):,} truth rows -> {a.out}")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
