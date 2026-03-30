"""scripts/report.py — Evaluate all trained models and print metrics.

Metrics reported:
  - Macro F1, balanced accuracy
  - NLL (negative log-likelihood)
  - Brier score (multiclass)
  - ECE (expected calibration error)
  - Early-time curves (metrics vs epoch_index)

Usage:
    python scripts/report.py
    python scripts/report.py --model-dir models --gold-dir data/gold
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from debass.models.baselines import LABEL_MAP, LABEL_NAMES, N_CLASSES

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


def _load_gold(gold_dir: Path):
    import pandas as pd
    path = gold_dir / "snapshots.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run normalize.py first")
    return pd.read_parquet(path)


def _build_X(df, prob_cols):
    rows = []
    for _, r in df.iterrows():
        scores = r.get("broker_scores") or {}
        row = {}
        for c in prob_cols:
            broker = c.replace("_mean_prob", "")
            b_scores = {k: v for k, v in scores.items() if k.startswith(broker + "__")}
            vals = [
                v["canonical_projection"]
                for v in b_scores.values()
                if isinstance(v, dict) and v.get("canonical_projection") is not None
            ]
            row[c] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)[prob_cols].values


def _brier(probs, y, n_classes=N_CLASSES):
    y_oh = np.eye(n_classes)[y]
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))


def _nll(probs, y):
    probs = np.clip(probs, 1e-7, 1)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y])))


def _ece(probs, y, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            acc = correct[mask].mean()
            conf = confidences[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return float(ece / len(y))


def _eval_model(clf, X, y, epoch_idx, name):
    from sklearn.metrics import f1_score, balanced_accuracy_score
    try:
        if hasattr(clf, 'predict_proba') and callable(clf.predict_proba):
            # Check if it's a MetaClassifier (needs broker dict format)
            probs = clf.predict_proba(X)
        else:
            probs = clf.predict_proba(X)
        preds = probs.argmax(axis=1)
        return {
            "model": name,
            "macro_f1": round(f1_score(y, preds, average="macro", zero_division=0), 4),
            "bal_acc": round(balanced_accuracy_score(y, preds), 4),
            "nll": round(_nll(probs, y), 4),
            "brier": round(_brier(probs, y), 4),
            "ece": round(_ece(probs, y), 4),
        }
    except Exception as exc:
        return {"model": name, "error": str(exc)}


def _print_metrics(rows):
    if console:
        t = Table(title="DEBASS Model Evaluation")
        cols = ["model", "macro_f1", "bal_acc", "nll", "brier", "ece"]
        for c in cols:
            t.add_column(c)
        for r in rows:
            if "error" in r:
                t.add_row(r["model"], "—", "—", "—", "—", r["error"])
            else:
                t.add_row(*[str(r.get(c, "—")) for c in cols])
        console.print(t)
    else:
        for r in rows:
            print(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report DEBASS model metrics")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--gold-dir", default="data/gold")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    gold_dir = Path(args.gold_dir)

    print("Loading gold snapshots...")
    df = _load_gold(gold_dir)
    labels = []
    for lbl in df["target_label"]:
        labels.append(LABEL_MAP.get(lbl, -1))
    y = np.array(labels)
    labelled = y >= 0
    df = df[labelled]
    y = y[labelled]
    epoch_idx = df["epoch_index"].values
    print(f"  {len(df)} labelled snapshots")

    # Determine prob_cols from saved baseline or hardcode brokers
    _BROKERS = ["alerce", "fink", "lasair", "supernnova", "parsnip", "ampel"]
    prob_cols = [f"{b}_mean_prob" for b in _BROKERS]
    X = _build_X(df, prob_cols)

    results = []

    # Evaluate baselines
    bl_dir = model_dir / "baselines"
    if bl_dir.exists():
        for pkl_path in sorted(bl_dir.glob("*.pkl")):
            with open(pkl_path, "rb") as fh:
                clf = pickle.load(fh)
            results.append(_eval_model(clf, X, y, epoch_idx, pkl_path.stem))

    # Evaluate meta-classifier (full MetaClassifier object)
    meta_dir = model_dir / "meta"
    meta_full_path = meta_dir / "meta_full.pkl"
    if meta_full_path.exists():
        with open(meta_full_path, "rb") as fh:
            meta_obj = pickle.load(fh)
        # Build broker dict format expected by MetaClassifier
        class _MetaWrapper:
            def __init__(self, meta, prob_cols, epoch_idx):
                self._meta = meta
                self._prob_cols = prob_cols
                self._epoch_idx = epoch_idx
            def predict_proba(self, X):
                X_brokers = {b: X[:, [i]] for i, b in enumerate(self._prob_cols)}
                avail = {b: (~np.isnan(X[:, i])).astype(float) for i, b in enumerate(self._prob_cols)}
                return self._meta.predict_proba(X_brokers, avail, self._epoch_idx)
        results.append(_eval_model(
            _MetaWrapper(meta_obj, prob_cols, epoch_idx), X, y, epoch_idx, "meta_classifier"
        ))

    if not results:
        print("No trained models found. Run scripts/train.py first.")
        sys.exit(1)

    _print_metrics(results)

    # Save JSON report
    import json
    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "evaluation.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nMetrics saved → {out}")


if __name__ == "__main__":
    main()
