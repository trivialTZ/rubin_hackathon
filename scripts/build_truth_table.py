"""Build the canonical truth table.

Supports three input modes (in priority order):
  1. ``--tns-crossmatch`` — TNS crossmatch parquet (from crossmatch_tns.py)
     with three-tier quality (spectroscopic / consensus / weak)
  2. ``--input`` — external curated CSV/parquet truth table
  3. ``--labels`` — weak self-label CSV (broker labels only)

Modes can be combined: TNS crossmatch is merged with --labels fallback,
and --input overrides everything for objects it covers.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from validate_truth_table import validate_truth_frame


def _read_table(path: Path):
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _print_weak_label_warning(labels_path: Path) -> None:
    print(
        "WARNING: building truth from weak labels in "
        f"{labels_path}. label_source will be 'alerce_self_label' and "
        "label_quality will be 'weak'."
    )


def build_truth_table(
    *,
    input_path: Path | None,
    labels_path: Path | None,
    tns_crossmatch_path: Path | None = None,
    output_path: Path,
    label_source: str | None = None,
    label_quality: str | None = None,
) -> Path:
    import pandas as pd

    frames: list[pd.DataFrame] = []

    # Layer 1: weak labels from labels.csv (lowest priority)
    if labels_path is not None and labels_path.exists():
        rows = []
        with open(labels_path) as fh:
            for record in csv.DictReader(fh):
                ternary = record.get("label")
                rows.append({
                    "object_id": record["object_id"],
                    "final_class_raw": ternary,
                    "final_class_ternary": ternary,
                    "follow_proxy": int(ternary == "snia") if ternary is not None else None,
                    "label_source": label_source or "alerce_self_label",
                    "label_quality": label_quality or "weak",
                    "truth_timestamp": None,
                })
        if rows:
            frames.append(pd.DataFrame(rows))

    # Layer 2: TNS crossmatch (overrides weak labels)
    if tns_crossmatch_path is not None and tns_crossmatch_path.exists():
        tns_df = _read_table(tns_crossmatch_path).copy()
        if "object_id" in tns_df.columns and "final_class_ternary" in tns_df.columns:
            if "follow_proxy" not in tns_df.columns:
                tns_df["follow_proxy"] = (tns_df["final_class_ternary"] == "snia").astype(int)
            if "final_class_raw" not in tns_df.columns:
                tns_df["final_class_raw"] = tns_df.get("tns_type", tns_df["final_class_ternary"])
            frames.append(tns_df)

    # Layer 3: external curated input (highest priority, overrides everything)
    if input_path is not None and input_path.exists():
        df_ext = _read_table(input_path).copy()
        if "object_id" not in df_ext.columns:
            raise ValueError("truth input must contain object_id")
        if "final_class_raw" not in df_ext.columns:
            if "label" in df_ext.columns:
                df_ext["final_class_raw"] = df_ext["label"]
            elif "final_class_ternary" in df_ext.columns:
                df_ext["final_class_raw"] = df_ext["final_class_ternary"]
        if "final_class_ternary" not in df_ext.columns:
            if "label" in df_ext.columns:
                df_ext["final_class_ternary"] = df_ext["label"]
            else:
                raise ValueError("truth input must contain final_class_ternary or label")
        if "follow_proxy" not in df_ext.columns:
            df_ext["follow_proxy"] = (df_ext["final_class_ternary"] == "snia").astype(int)
        if "label_source" not in df_ext.columns:
            df_ext["label_source"] = "external_curated"
        if "label_quality" not in df_ext.columns:
            df_ext["label_quality"] = "strong"
        frames.append(df_ext)

    if not frames:
        raise FileNotFoundError("provide --input, --tns-crossmatch, or --labels")

    # Merge: later frames override earlier ones (per object_id)
    df = frames[0]
    for next_df in frames[1:]:
        # Keep only the columns that exist in both + new columns from next_df
        override_ids = set(next_df["object_id"].dropna().astype(str))
        df = pd.concat([
            df[~df["object_id"].astype(str).isin(override_ids)],
            next_df,
        ], ignore_index=True, sort=False)

    required_cols = [
        "object_id",
        "final_class_raw",
        "final_class_ternary",
        "follow_proxy",
        "label_source",
        "label_quality",
    ]
    for column in required_cols:
        if column not in df.columns:
            df[column] = None

    validation = validate_truth_frame(df)
    if not validation["passed"]:
        raise ValueError("truth table validation failed: " + "; ".join(validation["errors"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical truth table")
    parser.add_argument("--input", default=None, help="External CSV/parquet truth table (highest priority)")
    parser.add_argument("--tns-crossmatch", default=None, help="TNS crossmatch parquet from crossmatch_tns.py")
    parser.add_argument("--labels", default=None, help="Weak-label CSV with object_id,label columns (lowest priority)")
    parser.add_argument("--output", default="data/truth/object_truth.parquet")
    parser.add_argument("--label-source", default=None)
    parser.add_argument("--label-quality", default=None)
    args = parser.parse_args()

    if not args.input and not args.labels and not args.tns_crossmatch:
        raise SystemExit("Provide --input, --tns-crossmatch, or --labels")

    labels_path = Path(args.labels) if args.labels else None
    if labels_path is not None and args.input is None and args.tns_crossmatch is None:
        _print_weak_label_warning(labels_path)

    tns_path = Path(args.tns_crossmatch) if args.tns_crossmatch else None

    path = build_truth_table(
        input_path=Path(args.input) if args.input else None,
        labels_path=labels_path,
        tns_crossmatch_path=tns_path,
        output_path=Path(args.output),
        label_source=args.label_source,
        label_quality=args.label_quality,
    )

    import pandas as pd
    df = pd.read_parquet(path)
    quality_counts = df["label_quality"].value_counts().to_dict()
    print(f"Wrote truth table → {path}  ({len(df)} objects, quality: {quality_counts})")


if __name__ == "__main__":
    main()
