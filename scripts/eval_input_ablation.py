#!/usr/bin/env python3
"""Input-availability ablation on the frozen LSST live benchmark.

Blanks whole groups of expert inputs in a benchmark gold table (src/debass_meta/features/availability.py), scores each
variant with every requested fusion stack (scripts/score_fusion_v11.py, same flags as the benchmark job), and reports
per detection checkpoint (3, 5, 10, each object's latest):

  SN-vs-other AUC [95% bootstrap over objects], median P(SN) on SNe and on others, Ia|SN AUC

Variants: full (as scored), nobroker (local experts + lightcurve), nolocal (brokers + lightcurve), noexpert
(lightcurve only). Only benchmark ids with spectroscopic or catalogue-context truth are scored.

  python scripts/eval_input_ablation.py --gold data/label_refresh_20260924/bench/gold/bench_v12.parquet \\
      --truth data/label_refresh_20260924/bench/truth.parquet --model v12 --model v12w \\
      --out-dir data/label_refresh_20260924/ablate_v12
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from debass_meta.features.availability import mask_group  # noqa: E402

VARIANTS = {"full": None, "nobroker": "brokers", "nolocal": "local", "noexpert": "all"}
CHECKPOINTS = [3, 5, 10, "latest"]
SN = {"snia", "nonIa_snlike"}


def auc_ci(y, s, rng, n_boot=1000):
    y, s = np.asarray(y, int), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return None
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    return [round(float(roc_auc_score(y, s)), 3), round(float(np.percentile(boots, 2.5)), 3),
            round(float(np.percentile(boots, 97.5)), 3)]


def metrics(pred: pd.DataFrame, truth: pd.DataFrame, ids: set[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    p = pred.drop(columns=[c for c in ("target_class", "label_quality", "label_source") if c in pred.columns])
    p = p.assign(object_id=p["object_id"].astype(str)).join(truth[["final_class_ternary", "label_quality"]], on="object_id")
    p = p[p["object_id"].isin(ids) & p["label_quality"].isin(["spectroscopic", "context"])]
    out = {}
    for nd in CHECKPOINTS:
        s = p.sort_values("n_det").groupby("object_id").tail(1) if nd == "latest" else p[p["n_det"] == nd]
        y = s["final_class_ternary"].isin(SN).to_numpy(int)
        psn = (s["p_snia"] + s["p_nonia"]).to_numpy(float)
        sn = s[y == 1]
        yi = (sn["final_class_ternary"] == "snia").to_numpy(int)
        out[str(nd)] = {
            "n": int(len(s)), "n_sn": int(y.sum()),
            "sn_vs_other": auc_ci(y, psn, rng),
            "median_psn_sn": round(float(np.median(psn[y == 1])), 3) if y.any() else None,
            "median_psn_other": round(float(np.median(psn[y == 0])), 3) if (y == 0).any() else None,
            "ia_given_sn": auc_ci(yi, (sn["p_snia"] / (sn["p_snia"] + sn["p_nonia"])).to_numpy(float), rng),
        }
    return out


def score(gold: Path, model: str, tag: str, scores_dir: Path, models_root: Path) -> Path:
    m = models_root
    cmd = [sys.executable, str(REPO / "scripts/score_fusion_v11.py"), "--tag", tag, "--snapshots", str(gold),
           "--trust-dir", str(m / f"trust_fusion_{model}"), "--followup-dir", str(m / f"followup_fusion_{model}"),
           "--blend-dir", str(m / f"anchor_blend_{model}"),
           "--conformal", str(m / f"conformal_fusion_{model}/mondrian_aps.pkl"),
           "--scores-dir", str(scores_dir), "--no-priority"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"scoring failed: {model} on {gold}")
    return scores_dir / f"predictions_{tag}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, type=Path, help="benchmark gold (built with the current projectors)")
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=REPO / "data/gold/lsst_live_locked_test.json")
    ap.add_argument("--model", action="append", required=True, help="model suffix, e.g. v12 (models/*_fusion_v12)")
    ap.add_argument("--models-root", type=Path, default=REPO / "models")
    ap.add_argument("--variant", action="append", choices=list(VARIANTS), default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_parquet(a.truth).assign(object_id=lambda d: d["object_id"].astype(str)).set_index("object_id")
    ids = {str(x) for x in json.loads(a.manifest.read_text())["test_ids"]}
    gold = pd.read_parquet(a.gold)
    results: dict = {}
    for variant in a.variant or list(VARIANTS):
        grp = VARIANTS[variant]
        vg = a.out_dir / f"gold_{variant}.parquet"
        (gold if grp is None else mask_group(gold, grp)).to_parquet(vg, index=False)
        for model in a.model:
            pred = pd.read_parquet(score(vg, model, f"abl_{variant}_{model}", a.out_dir, a.models_root))
            results.setdefault(model, {})[variant] = metrics(pred, truth, ids, a.seed)
            print(f"  scored {model} / {variant}", flush=True)

    (a.out_dir / "ablation_metrics.json").write_text(json.dumps(results, indent=1))
    lines = ["| model | inputs | " + " | ".join(f"n = {c}" if c != "latest" else "latest" for c in CHECKPOINTS) + " |",
             "|---|---|" + "---|" * len(CHECKPOINTS),
             ]
    for model, byv in results.items():
        for variant, r in byv.items():
            cells = []
            for c in CHECKPOINTS:
                x = r[str(c)]
                auc = x["sn_vs_other"]
                cells.append("—" if auc is None else f"{auc[0]:.3f} / {x['median_psn_sn']:.2f} / {x['median_psn_other']:.2f}")
            lines.append(f"| {model} | {variant} | " + " | ".join(cells) + " |")
    md = ("Cells: SN-vs-other AUC / median P(SN) on SNe / on others (benchmark, spec + catalogue-context truth).\n\n"
          + "\n".join(lines) + "\n")
    (a.out_dir / "ablation_metrics.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
