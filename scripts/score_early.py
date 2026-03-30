"""scripts/score_early.py — Score a new object at any n_det and print follow-up priority.

Usage:
    python scripts/score_early.py --object ZTF21abywdxt --n-det 4
    python scripts/score_early.py --object ZTF21abywdxt   # all epochs
    python scripts/score_early.py --from-labels data/labels.csv --n-det 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass.models.early_meta import EarlyMetaClassifier

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


def _print_results(results: list[dict]) -> None:
    if console and results:
        t = Table(title="Early-Epoch Classification")
        for col in ["object_id", "n_det", "alert_mjd", "p_snia", "p_nonIa_snlike", "p_other", "predicted_class", "follow_up_priority"]:
            t.add_column(col, overflow="fold")
        for r in results:
            if "error" in r:
                t.add_row(r.get("object_id", ""), str(r.get("n_det", "")), "", "", "", "", "", r["error"])
            else:
                priority = r["follow_up_priority"]
                color = "green" if priority > 0.6 else "yellow" if priority > 0.3 else "red"
                t.add_row(
                    r["object_id"],
                    str(r["n_det"]),
                    str(r.get("alert_mjd", ""  )),
                    str(r["p_snia"]),
                    str(r["p_nonIa_snlike"]),
                    str(r["p_other"]),
                    r["predicted_class"],
                    f"[{color}]{priority}[/{color}]",
                )
        console.print(t)
    else:
        for r in results:
            print(json.dumps(r))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score objects with early-epoch meta-classifier")
    parser.add_argument("--object", default=None, help="Single ZTF object ID")
    parser.add_argument("--n-det", type=int, default=None, help="Score at this detection epoch only")
    parser.add_argument("--from-labels", default=None, help="CSV with object_id column")
    parser.add_argument("--epochs-dir", default="data/epochs")
    parser.add_argument("--model-dir",  default="models/early_meta")
    parser.add_argument("--max-n-det",  type=int, default=20)
    args = parser.parse_args()

    import pandas as pd

    # Load epoch table
    epoch_path = Path(args.epochs_dir) / "epoch_features.parquet"
    if not epoch_path.exists():
        print(f"Epoch table not found: {epoch_path}")
        print("Run: python scripts/fetch_lightcurves.py && python scripts/build_epoch_table.py")
        sys.exit(1)
    df = pd.read_parquet(epoch_path)

    # Load model
    model_dir = Path(args.model_dir)
    if not (model_dir / "early_meta.pkl").exists():
        print(f"Model not found in {model_dir}")
        print("Run: python scripts/train_early.py")
        sys.exit(1)
    clf = EarlyMetaClassifier.load(model_dir)

    # Determine which objects to score
    if args.object:
        oids = [args.object]
    elif args.from_labels:
        oids = []
        with open(args.from_labels) as fh:
            for row in csv.DictReader(fh):
                oids.append(row["object_id"])
    else:
        oids = list(df["object_id"].unique())

    results = []
    for oid in oids:
        obj_df = df[df["object_id"] == oid]
        if len(obj_df) == 0:
            results.append({"object_id": oid, "n_det": args.n_det or 0, "error": "not in epoch table"})
            continue

        if args.n_det is not None:
            r = clf.score_object_at_n_det(oid, args.n_det, df)
            results.append(r)
        else:
            # Score at all available epochs up to max_n_det
            n_dets = sorted(obj_df["n_det"].unique())
            for nd in n_dets[:args.max_n_det]:
                r = clf.score_object_at_n_det(oid, nd, df)
                results.append(r)

    _print_results(results)


if __name__ == "__main__":
    main()
