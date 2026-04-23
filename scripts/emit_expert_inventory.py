#!/usr/bin/env python3
"""Emit a final expert inventory JSON after the light-up campaign finishes.

Reports, for every registered expert in EXPERT_REGISTRY:
  - silver event count (from broker_events.parquet)
  - has_trust_head (model.pkl present in models/trust/<expert>/)
  - has_calibrator (calibrator.pkl present)
  - test AUC (raw + calibrated from expert_trust_metrics.json)
  - n_train / n_cal / n_test rows

Run after F3 to produce reports/expert_inventory.json and a
human-readable markdown at reports/expert_inventory.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from debass_meta.projectors.base import EXPERT_REGISTRY  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", default="data/silver/broker_events.parquet")
    parser.add_argument("--trust-dir", default="models/trust")
    parser.add_argument("--metrics-json", default="reports/metrics/expert_trust_metrics.json")
    parser.add_argument("--output-json", default="reports/expert_inventory.json")
    parser.add_argument("--output-md", default="reports/expert_inventory.md")
    args = parser.parse_args()

    silver = pd.read_parquet(args.silver) if Path(args.silver).exists() else pd.DataFrame()
    silver_counts: dict[str, int] = {}
    if not silver.empty:
        # Prefer the explicit expert_key column written by silver.py; fall back
        # to broker/classifier aliasing for older silver files.
        if "expert_key" in silver.columns:
            silver_counts = silver["expert_key"].value_counts().to_dict()
        else:
            for (broker, clf), g in silver.groupby(["broker", "classifier"]):
                key1 = f"{broker}/{clf}"
                if key1 in EXPERT_REGISTRY:
                    silver_counts[key1] = int(len(g))

    metrics: dict = {}
    if Path(args.metrics_json).exists():
        metrics = json.loads(Path(args.metrics_json).read_text())

    rows = []
    for expert_key, (survey, broker) in EXPERT_REGISTRY.items():
        trust_subdir = expert_key.replace("/", "__")
        trust_path = Path(args.trust_dir) / trust_subdir / "model.pkl"
        calib_path = Path(args.trust_dir) / trust_subdir / "calibrator.pkl"
        has_trust = trust_path.exists()
        has_calib = calib_path.exists()
        m = metrics.get(expert_key, {})
        raw = m.get("raw", m)
        cal = m.get("calibrated", {})
        rows.append(
            {
                "expert": expert_key,
                "broker": broker,
                "survey": survey,
                "silver_events": silver_counts.get(expert_key, 0),
                "has_trust_head": has_trust,
                "has_calibrator": has_calib,
                "test_auc_raw":  raw.get("roc_auc"),
                "test_auc_cal":  cal.get("roc_auc"),
                "test_ece_raw": raw.get("ece"),
                "test_ece_cal": cal.get("ece"),
                "n_train": m.get("n_train_rows"),
                "n_cal":   m.get("n_cal_rows"),
                "n_test":  m.get("n_test_rows"),
            }
        )
    inv = pd.DataFrame(rows)
    inv = inv.sort_values(["has_trust_head", "silver_events"], ascending=[False, False])
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(inv.to_json(orient="records", indent=2))
    # Markdown summary
    md_lines = [
        "# DEBASS expert inventory",
        "",
        f"Registered: {len(inv)}   ·   Live (trust head present): **{int(inv['has_trust_head'].sum())}**   ·   With calibrator: {int(inv['has_calibrator'].sum())}",
        "",
        "| Expert | Broker | Survey | Silver | Trust | Cal | AUC (raw→cal) | ECE (raw→cal) |",
        "|---|---|---|--:|:-:|:-:|---|---|",
    ]
    for _, r in inv.iterrows():
        auc = (
            f"{r['test_auc_raw']:.3f}" if r["test_auc_raw"] is not None else "—"
        ) + (
            f" → {r['test_auc_cal']:.3f}" if r["test_auc_cal"] is not None else ""
        )
        ece = (
            f"{r['test_ece_raw']:.3f}" if r["test_ece_raw"] is not None else "—"
        ) + (
            f" → {r['test_ece_cal']:.3f}" if r["test_ece_cal"] is not None else ""
        )
        trust = "✓" if r["has_trust_head"] else "·"
        cal_mark = "✓" if r["has_calibrator"] else "·"
        md_lines.append(
            f"| `{r['expert']}` | {r['broker']} | {r['survey']} | {r['silver_events']:,} | {trust} | {cal_mark} | {auc} | {ece} |"
        )
    Path(args.output_md).write_text("\n".join(md_lines) + "\n")
    print(f"wrote {out_json} and {args.output_md}")
    print(
        f"Live: {int(inv['has_trust_head'].sum())} / {len(inv)} experts with trust heads"
    )


if __name__ == "__main__":
    main()
