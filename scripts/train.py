"""scripts/train.py — Legacy training entry point.

This script is retained for baseline compatibility only.
It targets the old `snapshots.parquet` + `MetaClassifier` stack and is not part
of the production trust-aware pipeline.
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy training entry point")
    parser.add_argument("--method", default="lgbm", choices=["lgbm", "logistic"])
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--model-dir", default="models")
    _ = parser.parse_args()

    print("train.py is a legacy baseline entry point and is not part of the trust-aware pipeline.")
    print("Use one of the following instead:")
    print("  Trust-aware: scripts/train_expert_trust.py && scripts/train_followup.py")
    print("  Baseline benchmark: scripts/train_early.py")
    sys.exit(1)


if __name__ == "__main__":
    main()
