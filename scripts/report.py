"""scripts/report.py — Legacy evaluation entry point.

This script is retained for baseline compatibility only.
It targets the old `snapshots.parquet` + `MetaClassifier` stack and is not part
of the production trust-aware pipeline.
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy evaluation entry point")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--gold-dir", default="data/gold")
    _ = parser.parse_args()

    print("report.py is a legacy baseline evaluation entry point and is not part of the trust-aware pipeline.")
    print("Use one of the following instead:")
    print("  Trust-aware scoring: scripts/score_nightly.py")
    print("  Baseline benchmark scoring: scripts/score_early.py")
    sys.exit(1)


if __name__ == "__main__":
    main()
