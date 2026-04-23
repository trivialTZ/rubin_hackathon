#!/usr/bin/env python3
"""Export the train-fold object list used by train_expert_trust / train_followup.

Uses `group_train_cal_test_split` with seed=42 to reproduce the exact same
split the downstream trust + followup heads use.  Writes the train-fold
object ids to a CSV for retraining local experts without leaking cal/test
objects into their training data.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from debass_meta.models.splitters import group_train_cal_test_split  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_safe_v2.parquet",
                   help="source for object→label mapping (target_class)")
    p.add_argument("--out-train", default="data/labels_train_fold.csv")
    p.add_argument("--out-cal",   default=None)
    p.add_argument("--out-test",  default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_parquet(args.snapshots, columns=["object_id", "target_class"])
    obj_to_label = (
        df.dropna(subset=["target_class"])
        .drop_duplicates("object_id")
        .set_index("object_id")["target_class"]
        .astype(str)
        .to_dict()
    )
    print(f"object→label entries: {len(obj_to_label)}")

    split = group_train_cal_test_split(obj_to_label, seed=args.seed)
    print(f"train={len(split.train_ids)}  cal={len(split.cal_ids)}  test={len(split.test_ids)}")

    def _write(path: str, ids: set[str]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["object_id"])
            for oid in sorted(ids):
                w.writerow([oid])
        print(f"  wrote {path} ({len(ids)})")

    _write(args.out_train, split.train_ids)
    if args.out_cal:
        _write(args.out_cal, split.cal_ids)
    if args.out_test:
        _write(args.out_test, split.test_ids)


if __name__ == "__main__":
    main()
