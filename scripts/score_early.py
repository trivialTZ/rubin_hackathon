"""scripts/score_early.py — Score objects with the baseline early-epoch model.

This script is kept for benchmark comparisons only.
It is not the primary trust-aware scoring path.

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

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_ROOT))

try:
    from debass_meta_meta.models.early_meta import EarlyMetaClassifier
except ModuleNotFoundError as exc:
    print(f"ERROR: failed to import DEBASS package modules from {_SRC_ROOT}: {exc}")
    print("This usually means the SCC checkout is incomplete or stale. Re-sync the repository and retry.")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


def _print_results(results: list[dict], *, summary: dict | None = None) -> None:
    if console and results:
        t = Table(title="Baseline Early-Epoch Classification")
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
                    str(r.get("alert_mjd", "")),
                    str(r["p_snia"]),
                    str(r["p_nonIa_snlike"]),
                    str(r["p_other"]),
                    r["predicted_class"],
                    f"[{color}]{priority}[/{color}]",
                )
        console.print(t)
        if summary:
            console.print(
                f"Scored {summary.get('scored', 0)} objects"
                + (
                    f"; skipped {summary.get('skipped_missing_epoch', 0)} objects with fewer than n_det={summary.get('requested_n_det')}"
                    if summary.get("requested_n_det") is not None
                    else ""
                )
            )
    else:
        for r in results:
            print(json.dumps(r))
        if summary:
            print(json.dumps({"summary": summary}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score objects with the baseline early-epoch meta-classifier")
    parser.add_argument("--object", default=None, help="Single ZTF object ID")
    parser.add_argument("--n-det", type=int, default=None, help="Score at this detection epoch only")
    parser.add_argument("--from-labels", default=None, help="CSV with object_id column")
    parser.add_argument("--epochs-dir", default="data/epochs")
    parser.add_argument("--model-dir",  default="models/early_meta")
    parser.add_argument("--max-n-det",  type=int, default=20)
    args = parser.parse_args()

    import pandas as pd

    print("NOTE: score_early.py is the baseline benchmark scorer, not the primary trust-aware payload path.")

    epoch_path = Path(args.epochs_dir) / "epoch_features.parquet"
    if not epoch_path.exists():
        print(f"Epoch table not found: {epoch_path}")
        print("Run: python scripts/build_epoch_table_from_lc.py")
        sys.exit(1)
    df = pd.read_parquet(epoch_path)

    model_dir = Path(args.model_dir)
    if not (model_dir / "early_meta.pkl").exists():
        print(f"Model not found in {model_dir}")
        print("Run: python scripts/train_early.py")
        sys.exit(1)
    clf = EarlyMetaClassifier.load(model_dir)

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
    skipped_missing_epoch = 0
    for oid in oids:
        obj_df = df[df["object_id"] == oid]
        if len(obj_df) == 0:
            results.append({"object_id": oid, "n_det": args.n_det or 0, "error": "not in epoch table"})
            continue

        if args.n_det is not None:
            if args.n_det not in set(obj_df["n_det"].astype(int).tolist()):
                if args.object:
                    results.append({"object_id": oid, "n_det": args.n_det, "error": f"No epoch row for {oid} at n_det={args.n_det}"})
                else:
                    skipped_missing_epoch += 1
                continue
            r = clf.score_object_at_n_det(oid, args.n_det, df)
            results.append(r)
        else:
            n_dets = sorted(obj_df["n_det"].unique())
            for nd in n_dets[:args.max_n_det]:
                r = clf.score_object_at_n_det(oid, nd, df)
                results.append(r)

    summary = {
        "scored": len([r for r in results if "error" not in r]),
        "errors": len([r for r in results if "error" in r]),
        "requested_n_det": args.n_det,
        "skipped_missing_epoch": skipped_missing_epoch,
    }
    _print_results(results, summary=summary)


if __name__ == "__main__":
    main()
