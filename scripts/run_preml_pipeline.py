"""Run the full pre-ML DEBASS pipeline in a fixed, benchmarked order."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_preml_readiness import evaluate_preml_readiness


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"$ {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DEBASS pre-ML data pipeline")
    parser.add_argument("--root", default="tmp/preml_run")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--truth-input", default=None, help="Optional curated truth CSV/parquet")
    parser.add_argument("--lightcurves-dir", default="data/lightcurves")
    parser.add_argument("--broker", default="all")
    parser.add_argument(
        "--association-csv",
        default=None,
        help="Optional LSST→ZTF association CSV for broker routing and lightcurve sourcing",
    )
    parser.add_argument("--max-n-det", type=int, default=20)
    parser.add_argument("--skip-local-infer", action="store_true")
    parser.add_argument("--local-expert", default="all")
    parser.add_argument("--emit-scoring-snapshots", action="store_true",
                        help="Deprecated; scoring snapshots are emitted by default.")
    parser.add_argument("--skip-scoring-snapshots", action="store_true",
                        help="Skip building scoring snapshots that include latest_object_unsafe broker rows.")
    parser.add_argument("--benchmarks", default="benchmarks/preml_data_engineering.toml")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--require-trust-training-ready", action="store_true")
    parser.add_argument("--require-full-phase1-ready", action="store_true")
    args = parser.parse_args()

    emit_scoring_snapshots = args.emit_scoring_snapshots or not args.skip_scoring_snapshots

    repo_root = Path(__file__).resolve().parent.parent
    root = Path(args.root)
    bronze_dir = root / "bronze"
    silver_dir = root / "silver"
    gold_dir = root / "gold"
    truth_path = root / "truth/object_truth.parquet"
    labels_path = Path(args.labels)
    lc_dir = Path(args.lightcurves_dir)

    for path in [bronze_dir, silver_dir, gold_dir, truth_path.parent, root / "reports"]:
        path.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    _run(
        [
            py,
            "scripts/backfill.py",
            "--broker",
            args.broker,
            "--from-labels",
            str(labels_path),
            "--bronze-dir",
            str(bronze_dir),
            *(
                ["--association-csv", str(args.association_csv)]
                if args.association_csv
                else []
            ),
        ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "scripts/normalize.py",
            "--bronze-dir",
            str(bronze_dir),
            "--silver-dir",
            str(silver_dir),
        ],
        cwd=repo_root,
    )

    truth_cmd = [py, "scripts/build_truth_table.py", "--output", str(truth_path)]
    if args.truth_input:
        truth_cmd.extend(["--input", args.truth_input])
    else:
        truth_cmd.extend(["--labels", str(labels_path)])
    _run(truth_cmd, cwd=repo_root)

    if not args.skip_local_infer:
        _run(
            [
                py,
                "scripts/local_infer.py",
                "--expert",
                args.local_expert,
                "--from-labels",
                str(labels_path),
                "--lc-dir",
                str(lc_dir),
                "--silver-dir",
                str(silver_dir),
                "--max-n-det",
                str(args.max_n_det),
            ],
            cwd=repo_root,
        )

    _run(
        [
            py,
            "scripts/build_object_epoch_snapshots.py",
            "--lc-dir",
            str(lc_dir),
            "--silver-dir",
            str(silver_dir),
            "--gold-dir",
            str(gold_dir),
            "--truth",
            str(truth_path),
                "--objects-csv",
                str(labels_path),
                "--max-n-det",
                str(args.max_n_det),
                *(
                    ["--association-csv", str(args.association_csv)]
                    if args.association_csv
                    else []
                ),
            ],
        cwd=repo_root,
    )
    _run(
        [
            py,
            "scripts/build_expert_helpfulness.py",
            "--snapshots",
            str(gold_dir / "object_epoch_snapshots.parquet"),
            "--output",
            str(gold_dir / "expert_helpfulness.parquet"),
        ],
        cwd=repo_root,
    )

    if emit_scoring_snapshots:
        _run(
            [
                py,
                "scripts/build_object_epoch_snapshots.py",
                "--lc-dir",
                str(lc_dir),
                "--silver-dir",
                str(silver_dir),
                "--gold-dir",
                str(gold_dir),
                "--truth",
                str(truth_path),
                "--objects-csv",
                str(labels_path),
                "--max-n-det",
                str(args.max_n_det),
                "--allow-unsafe-latest-snapshot",
                "--output",
                str(gold_dir / "object_epoch_snapshots_scoring.parquet"),
                *(
                    ["--association-csv", str(args.association_csv)]
                    if args.association_csv
                    else []
                ),
            ],
            cwd=repo_root,
        )

    _run(
        [
            py,
            "scripts/audit_data_readiness.py",
            "--root",
            str(root),
            "--labels",
            str(labels_path),
            "--lightcurves-dir",
            str(lc_dir),
        ],
        cwd=repo_root,
    )

    if not args.skip_benchmarks:
        _run(
            [
                py,
                "scripts/check_data_benchmarks.py",
                "--root",
                str(root),
                "--benchmarks",
                str(args.benchmarks),
                "--labels",
                str(labels_path),
                "--lightcurves-dir",
                str(lc_dir),
            ],
            cwd=repo_root,
        )

    readiness = evaluate_preml_readiness(
        root=root,
        labels_path=labels_path,
        lightcurve_dir=lc_dir,
    )
    readiness_path = root / "reports/summary/preml_readiness.json"
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    with open(readiness_path, "w") as fh:
        json.dump(readiness, fh, indent=2)
    print(f"Wrote pre-ML readiness report → {readiness_path}")

    if args.require_trust_training_ready and not readiness["ready_for_trust_training_now"]:
        raise SystemExit("pre-ML pipeline completed, but trust-training readiness is still blocked")
    if args.require_full_phase1_ready and not readiness["ready_for_full_phase1_goal"]:
        raise SystemExit("pre-ML pipeline completed, but the full phase-1 goal is still blocked")


if __name__ == "__main__":
    main()
