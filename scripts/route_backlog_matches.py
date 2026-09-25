#!/usr/bin/env python3
"""Route TNS-backlog sweep matches into the benchmark manifest + train pools.

Consumes the output of scripts/sweep_tns_backlog.py and applies the leak
guards, in order:

  G-A  already in the frozen benchmark manifest (test_ids or excluded_ids)
       -> skip (label may still upgrade truth elsewhere; never re-routed).
  G-B  object_id already in the v11 TRAINING split train ∪ cal
       -> 'label_upgrade' pool ONLY (it has been trained on — with a weak
       label — so it must NEVER enter the benchmark; its spec label is still
       valuable for future retrains).  v11 *test* membership does NOT
       disqualify (held-out on both sides is fine — 2026ezw precedent).
  G-C  a ZTF counterpart (TNS internal_names) sits in the LOCKED ZTF v6e2
       train ∪ cal -> excluded_ids with reason (counterpart integrity, same
       mechanism as 2026ekf/2026gzf).  Counterpart in locked ZTF *test* is
       fine.
  G-D  survivors hash-route (sha1 last hex, B1 policy):
         even -> benchmark manifest test_ids APPEND;
         odd  -> train side, sub-split deterministically by the SECOND-to-last
                 hash hex: {0..b} -> 'extra_train' (GRU --extra-truth food),
                 {c..f} -> 'holdout_val' (internal LSST spec validation set —
                 never trained on, not part of the frozen benchmark).

Outputs (all overwritten):
  data/truth/backlog_train_truth.parquet    extra_train + label_upgrade rows
                                            (trainer --extra-truth schema)
  data/truth/backlog_holdout_truth.parquet  holdout_val rows
  data/gold/lsst_live_locked_test.json      updated IN PLACE (append-only)
                                            unless --dry-run

Run with --dry-run first; the printout is the audit trail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def ztf_names_from_internal(internal: str | None) -> list[str]:
    if not internal or not isinstance(internal, str):
        return []
    return re.findall(r"ZTF\w+", internal)


def second_hash_hex(object_id: str) -> str:
    return hashlib.sha1(str(object_id).encode()).hexdigest()[-2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", default="data/truth/tns_backlog_matches.parquet")
    ap.add_argument("--manifest", default="data/gold/lsst_live_locked_test.json")
    ap.add_argument("--v11-split", default="data/gold/split_fusion_v11_scc.json")
    ap.add_argument("--locked-ztf", action="append", default=None,
                    help="locked ZTF split file(s); default: every one of models/trust/metadata.json and "
                         "models/trust/metadata_scc_locked.json (the SCC copy, which is the split the SCC builds "
                         "use — the local metadata.json is an older v6e2 split) that exists")
    ap.add_argument("--tns-bulk", default="data/truth/tns_public.parquet",
                    help="for internal_names (ZTF counterpart lookup)")
    ap.add_argument("--train-out", default="data/truth/backlog_train_truth.parquet")
    ap.add_argument("--holdout-out", default="data/truth/backlog_holdout_truth.parquet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    m = pd.read_parquet(args.matches)
    m["object_id"] = m["object_id"].astype(str)
    man = json.loads(Path(args.manifest).read_text())
    man_test = set(map(str, man["test_ids"]))
    man_excl = dict(man.get("excluded_ids", {}))
    split = json.loads(Path(args.v11_split).read_text())
    v11_traincal = set(map(str, split["train_ids"])) | set(map(str, split["cal_ids"]))
    locked_paths = args.locked_ztf or [p for p in ("models/trust/metadata.json",
                                                   "models/trust/metadata_scc_locked.json") if Path(p).exists()]
    ztf_locked_traincal: set[str] = set()
    for lp in locked_paths:
        trust = json.loads(Path(lp).read_text())
        ztf_locked_traincal |= set(trust["train_ids"]) | set(trust["cal_ids"])
    print(f"locked ZTF train/cal from {locked_paths}: {len(ztf_locked_traincal):,} ids")

    tns = pd.read_parquet(args.tns_bulk, columns=["objname", "internal_names"])
    tns["norm"] = tns["objname"].astype(str).str.strip().str.lower()
    internal_by_name = dict(zip(tns["norm"], tns["internal_names"]))

    def norm_tns(name: str) -> str:
        return re.sub(r"^(AT|SN)\s*", "", str(name), flags=re.I).strip().lower()

    routes: dict[str, str] = {}
    new_excluded: dict[str, str] = {}
    for _, row in m.iterrows():
        oid = row["object_id"]
        if oid in man_test or oid in man_excl:
            routes[oid] = "already_in_manifest"
            continue
        if oid in v11_traincal:
            routes[oid] = "label_upgrade"          # G-B: trained on; train pool only
            continue
        twins = ztf_names_from_internal(internal_by_name.get(norm_tns(row["tns_name"])))
        locked_twins = [z for z in twins if z in ztf_locked_traincal]
        if locked_twins:
            routes[oid] = "excluded_counterpart"    # G-C
            new_excluded[oid] = (
                f"ztf_counterpart_{locked_twins[0]}_in_locked_v6e2/v10_train_cal "
                f"({row['tns_name']}, {row['tns_type']}; backlog sweep 2026-07-07)")
            continue
        if row["hash_route"] == "test":             # G-D even
            routes[oid] = "benchmark_test"
        else:                                       # G-D odd sub-split
            routes[oid] = ("extra_train" if second_hash_hex(oid) in "0123456789ab"
                           else "holdout_val")

    m["route"] = m["object_id"].map(routes)
    print("=== routing summary ===")
    print(m["route"].value_counts().to_string())
    print("\nIa (snia) per route:")
    print(m[m["final_class_ternary"] == "snia"]["route"].value_counts().to_string())

    truth_cols = {
        "object_id": m["object_id"], "tns_name": m["tns_name"],
        "tns_type": m["tns_type"], "final_class_ternary": m["final_class_ternary"],
        "label_quality": "spectroscopic", "label_source": "tns_backlog_sweep",
        "tns_redshift": m.get("tns_redshift"), "sep_arcsec": m["sep_arcsec"],
    }
    truth = pd.DataFrame(truth_cols)
    train_truth = truth[m["route"].isin(["extra_train", "label_upgrade"])]
    holdout_truth = truth[m["route"] == "holdout_val"]
    bench_new = m[m["route"] == "benchmark_test"]["object_id"].tolist()

    print(f"\nbenchmark append: {len(bench_new)} | excluded: {len(new_excluded)} | "
          f"extra_train: {(m['route'] == 'extra_train').sum()} | "
          f"label_upgrade: {(m['route'] == 'label_upgrade').sum()} | "
          f"holdout_val: {len(holdout_truth)}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return

    train_truth.to_parquet(args.train_out, index=False)
    holdout_truth.to_parquet(args.holdout_out, index=False)
    print(f"wrote {len(train_truth)} rows -> {args.train_out}")
    print(f"wrote {len(holdout_truth)} rows -> {args.holdout_out}")

    for oid in bench_new:
        if oid not in man_test:
            man["test_ids"].append(oid)
            man_test.add(oid)
    man_excl.update(new_excluded)
    man["excluded_ids"] = man_excl
    Path(args.manifest).write_text(json.dumps(man, indent=2))
    print(f"manifest -> {len(man['test_ids'])} test_ids "
          f"(+{len(bench_new)}), {len(man_excl)} excluded")


if __name__ == "__main__":
    main()
