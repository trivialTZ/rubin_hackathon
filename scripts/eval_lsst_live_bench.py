#!/usr/bin/env python3
"""Evaluate prediction sets on the frozen LSST-live benchmark manifest.

Follows the analyst conventions of data/live_eval_20260704/report/analysis.md
(and tools/v11_smoke_metrics.py):

  * Ia-vs-rest AUC ranked by p_snia, SPECTROSCOPIC objects only;
  * SN-vs-other AUC ranked by p_snia + p_nonia, spec + context objects
    (SN = spec snia/nonIa_snlike; other = context others + spec 'other');
  * epoch slices: exact n_det in {3, 5, 10} + 'max' (each object's last epoch);
  * bootstrap over rows (== objects; one row per object per slice),
    1000 resamples, percentile 95% CI, seed 42;
  * G2-spirit: median/max p_snia on spec-Ia rows.

The eval frame is ALWAYS restricted to the manifest ``test_ids`` (frozen
benchmark; stale_xmatch / tail_xmatch / tns_untyped objects never enter either
metric).  Labels come from the ``--truth`` parquet only — label columns baked
into prediction files are dropped (spec section 6: stale snapshot labels are
ignored).

Systems:
  * ``--pred name=path`` — GBM prediction parquet (score_fusion_v8/v11 output).
    If ``p_snia_model``/``p_snia_anchor`` columns exist, the blend / model /
    anchor variants are all reported; otherwise the single calibrated head.
  * ``--seq name=path``  — local_infer silver parquet (per-epoch
    ``class_probabilities`` JSON); evaluated standalone as its own system.

``--common-frame`` restricts every system to the intersection of all systems'
object coverage (within the manifest) so numbers are head-to-head comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
N_BOOT = 1000
SLICES = (3, 5, 10, "max")


def bootstrap_auc(y: np.ndarray, s: np.ndarray, n_boot: int = N_BOOT):
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2 or len(y) < 4:
        return float("nan"), float("nan"), float("nan"), int(len(y))
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(auc), float(lo), float(hi), int(n)


def epoch_slice(df: pd.DataFrame, gate) -> pd.DataFrame:
    if gate == "max":
        return (df.sort_values(["object_id", "n_det"])
                  .groupby("object_id", as_index=False).tail(1))
    return df[df["n_det"] == int(gate)]


_LABEL_COLS = ("final_class_ternary", "label_quality", "tns_name")


def load_pred(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["object_id"] = df["object_id"].astype(str)
    return df.drop(columns=[c for c in _LABEL_COLS if c in df.columns])


def load_seq(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["available"]].copy()
    df["object_id"] = df["object_id"].astype(str)
    probs = df["class_probabilities"].map(json.loads)
    df["p_snia"] = probs.map(lambda p: p.get("snia", np.nan)).astype(float)
    df["p_nonia"] = probs.map(lambda p: p.get("nonIa_snlike", np.nan)).astype(float)
    return df[["object_id", "n_det", "p_snia", "p_nonia"]]


def eval_system(df: pd.DataFrame, tmap: pd.DataFrame, variants) -> dict:
    df = df.join(tmap, on="object_id")
    spec = df[df["label_quality"] == "spectroscopic"]
    pool = df[df["label_quality"].isin(["spectroscopic", "context"])]
    out: dict = {}
    for vname, c_ia, c_non in variants:
        if c_ia not in df.columns:
            continue
        v: dict = {}
        for gate in SLICES:
            s_spec = epoch_slice(spec, gate)
            y_ia = (s_spec["final_class_ternary"] == "snia").to_numpy(int)
            a, lo, hi, n_sp = bootstrap_auc(y_ia, s_spec[c_ia].to_numpy(float))
            s_all = epoch_slice(pool, gate)
            y_sn = s_all["final_class_ternary"].isin(
                ["snia", "nonIa_snlike"]).to_numpy(int)
            sn_score = (s_all[c_ia].to_numpy(float)
                        + s_all[c_non].to_numpy(float))
            b, blo, bhi, n_all = bootstrap_auc(y_sn, sn_score)
            ia_rows = s_spec[s_spec["final_class_ternary"] == "snia"]
            med = float(np.nanmedian(ia_rows[c_ia])) if len(ia_rows) else float("nan")
            mx = float(np.nanmax(ia_rows[c_ia])) if len(ia_rows) else float("nan")
            v[str(gate)] = {
                "ia_vs_rest": {"auc": a, "ci": [lo, hi], "n": n_sp,
                               "n_ia": int(y_ia.sum())},
                "sn_vs_other": {"auc": b, "ci": [blo, bhi], "n": n_all,
                                "n_sn": int(y_sn.sum())},
                "spec_ia_p_snia": {"median": med, "max": mx, "n": int(len(ia_rows))},
            }
        out[vname] = v
    return out


def fmt_md(results: dict) -> str:
    lines = ["| system | variant | slice | Ia-vs-rest AUC [95% CI] (n_ia/n) | "
             "SN-vs-other AUC [95% CI] (n_sn/n) | med p_snia on Ia |",
             "|---|---|---|---|---|---|"]
    for sysname, variants in results["systems"].items():
        for vname, gates in variants.items():
            for gate in ("5", "max"):
                g = gates.get(gate)
                if not g:
                    continue
                ia, sn, gm = g["ia_vs_rest"], g["sn_vs_other"], g["spec_ia_p_snia"]

                def f(block):
                    if np.isnan(block["auc"]):
                        return f"— (n={block['n']})"
                    return (f"{block['auc']:.3f} [{block['ci'][0]:.3f},"
                            f"{block['ci'][1]:.3f}]")
                lines.append(
                    f"| {sysname} | {vname} | {gate} | {f(ia)} "
                    f"({ia['n_ia']}/{ia['n']}) | {f(sn)} ({sn['n_sn']}/{sn['n']}) | "
                    f"{gm['median']:.3f} (n={gm['n']}) |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/gold/lsst_live_locked_test.json")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pred", action="append", default=[],
                    metavar="NAME=PATH", help="GBM prediction parquet")
    ap.add_argument("--seq", action="append", default=[],
                    metavar="NAME=PATH", help="local_infer seq silver parquet")
    ap.add_argument("--common-frame", action="store_true",
                    help="restrict all systems to the common object coverage")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    test_ids = set(map(str, man["test_ids"]))

    truth = pd.read_parquet(args.truth)
    truth["object_id"] = truth["object_id"].astype(str)
    tmap = truth.set_index("object_id")[list(_LABEL_COLS)]

    frames: dict[str, tuple[pd.DataFrame, list]] = {}
    for spec_arg in args.pred:
        name, path = spec_arg.split("=", 1)
        df = load_pred(path)
        variants = [("blend", "p_snia", "p_nonia")]
        if "p_snia_model" in df.columns:
            variants += [("model", "p_snia_model", "p_nonia_model"),
                         ("anchor", "p_snia_anchor", "p_nonia_anchor")]
        frames[name] = (df, variants)
    for spec_arg in args.seq:
        name, path = spec_arg.split("=", 1)
        frames[name] = (load_seq(path), [("standalone", "p_snia", "p_nonia")])

    keep = set(test_ids)
    if args.common_frame:
        for df, _ in frames.values():
            keep &= set(df["object_id"].unique())

    results: dict = {"manifest": args.manifest, "truth": args.truth,
                     "n_manifest": len(test_ids),
                     "common_frame": bool(args.common_frame),
                     "n_frame_objects": len(keep) if args.common_frame else None,
                     "systems": {}, "coverage": {}}
    for name, (df, variants) in frames.items():
        sub = df[df["object_id"].isin(keep)]
        results["coverage"][name] = int(sub["object_id"].nunique())
        results["systems"][name] = eval_system(sub, tmap, variants)

    md = fmt_md(results)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=1))
        print(f"wrote {args.out_json}")
    if args.out_md:
        Path(args.out_md).write_text(md + "\n")
        print(f"wrote {args.out_md}")
    print(json.dumps(results["coverage"], indent=0))
    print(md)


if __name__ == "__main__":
    main()
