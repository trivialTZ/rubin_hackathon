"""Validate canonical truth tables before they enter the trust pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

ALLOWED_TERNARY = {"snia", "nonIa_snlike", "other"}
ALLOWED_QUALITY = {"spectroscopic", "consensus", "weak", "strong"}


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def validate_truth_frame(
    df: pd.DataFrame,
    *,
    require_strong: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    required_columns = {
        "object_id",
        "final_class_raw",
        "final_class_ternary",
        "follow_proxy",
        "label_source",
        "label_quality",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")
        return {"passed": False, "errors": errors, "warnings": warnings}

    if df["object_id"].isna().any():
        errors.append("object_id contains null values")
    duplicate_ids = df["object_id"][df["object_id"].duplicated()].astype(str).unique().tolist()
    if duplicate_ids:
        errors.append(f"duplicate object_id rows detected: {duplicate_ids[:10]}")

    invalid_ternary = sorted(
        set(
            str(value)
            for value in df["final_class_ternary"].dropna().astype(str).unique()
            if str(value) not in ALLOWED_TERNARY
        )
    )
    if invalid_ternary:
        errors.append(f"invalid final_class_ternary values: {invalid_ternary}")

    invalid_quality = sorted(
        set(
            str(value)
            for value in df["label_quality"].dropna().astype(str).unique()
            if str(value) not in ALLOWED_QUALITY
        )
    )
    if invalid_quality:
        errors.append(f"invalid label_quality values: {invalid_quality}")

    for idx, row in df.iterrows():
        follow_proxy = row.get("follow_proxy")
        ternary = row.get("final_class_ternary")
        if pd.isna(follow_proxy):
            errors.append(f"row {idx} has null follow_proxy")
            continue
        try:
            follow_proxy_int = int(follow_proxy)
        except Exception:
            errors.append(f"row {idx} has non-integer follow_proxy={follow_proxy!r}")
            continue
        if follow_proxy_int not in {0, 1}:
            errors.append(f"row {idx} has invalid follow_proxy={follow_proxy_int!r}")
        expected = int(str(ternary) == "snia") if pd.notna(ternary) else None
        if expected is not None and follow_proxy_int != expected:
            errors.append(
                f"row {idx} has follow_proxy={follow_proxy_int} but final_class_ternary={ternary!r}"
            )

    quality_counts = df["label_quality"].fillna("").astype(str).value_counts().to_dict()
    if quality_counts.get("weak", 0) > 0:
        warnings.append("weak truth rows are present")
    strong_count = quality_counts.get("strong", 0) + quality_counts.get("spectroscopic", 0)
    if strong_count == 0 and quality_counts.get("consensus", 0) == 0:
        warnings.append("no strong or consensus truth rows present")
    elif strong_count == 0:
        warnings.append("no spectroscopic/strong truth rows present (consensus available)")
    if require_strong and strong_count == 0:
        errors.append("require_strong=True but no strong/spectroscopic truth rows are present")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_rows": int(len(df)),
        "label_quality_counts": quality_counts,
        "label_source_counts": df["label_source"].fillna("").astype(str).value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical truth table")
    parser.add_argument("path", help="CSV or parquet truth table")
    parser.add_argument("--require-strong", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    path = Path(args.path)
    result = validate_truth_frame(_load_table(path), require_strong=args.require_strong)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote truth validation report → {out_path}")

    if result["passed"]:
        print("Truth validation passed")
        for warning in result["warnings"]:
            print(f" - WARNING: {warning}")
        return

    print("Truth validation failed")
    for error in result["errors"]:
        print(f" - ERROR: {error}")
    for warning in result["warnings"]:
        print(f" - WARNING: {warning}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
