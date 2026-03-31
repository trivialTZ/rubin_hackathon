"""Build the canonical truth table."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _read_table(path: Path):
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_truth_table(
    *,
    input_path: Path | None,
    labels_path: Path | None,
    output_path: Path,
    label_source: str | None = None,
    label_quality: str | None = None,
) -> Path:
    import pandas as pd

    if input_path is not None and input_path.exists():
        df = _read_table(input_path).copy()
        if "object_id" not in df.columns:
            raise ValueError("truth input must contain object_id")
        if "final_class_raw" not in df.columns:
            if "label" in df.columns:
                df["final_class_raw"] = df["label"]
            elif "final_class_ternary" in df.columns:
                df["final_class_raw"] = df["final_class_ternary"]
        if "final_class_ternary" not in df.columns:
            if "label" in df.columns:
                df["final_class_ternary"] = df["label"]
            else:
                raise ValueError("truth input must contain final_class_ternary or label")
        if "follow_proxy" not in df.columns:
            df["follow_proxy"] = (df["final_class_ternary"] == "snia").astype(int)
        if "label_source" not in df.columns:
            df["label_source"] = label_source or "external_curated"
        if "label_quality" not in df.columns:
            df["label_quality"] = label_quality or "strong"
    elif labels_path is not None and labels_path.exists():
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
        df = pd.DataFrame(rows)
    else:
        raise FileNotFoundError("provide --input or --labels")

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical truth table")
    parser.add_argument("--input", default=None, help="External CSV/parquet truth table")
    parser.add_argument("--labels", default="data/labels.csv", help="Fallback weak-label CSV")
    parser.add_argument("--output", default="data/truth/object_truth.parquet")
    parser.add_argument("--label-source", default=None)
    parser.add_argument("--label-quality", default=None)
    args = parser.parse_args()

    path = build_truth_table(
        input_path=Path(args.input) if args.input else None,
        labels_path=Path(args.labels) if args.labels else None,
        output_path=Path(args.output),
        label_source=args.label_source,
        label_quality=args.label_quality,
    )
    print(f"Wrote truth table → {path}")


if __name__ == "__main__":
    main()
