#!/usr/bin/env python3
"""Evaluate metaDEBASS predictions on named Rubin object sets (any model version).

Per set and slice (n_det = 3, 5, 10, and each object's latest epoch):

  * n objects, n SNe; SN-vs-other AUC with an object-level bootstrap 95% CI (fixed seed);
    Brier score of P(SN) = p_snia + p_nonia; median P(SN) on SNe and on others;
    for others, the fraction with P(SN) > 0.5;
  * best-single-input baseline: every expert's projected p_sn = p_snia + p_nonIa_snlike
    alone (an absent expert scores 0.5), its AUC, the best expert, and the PAIRED
    bootstrap difference metaDEBASS minus that expert (same resamples).  The best
    expert is chosen on the very objects it is scored on, which favours the baseline;
    the difference is therefore a conservative read of the gain;
  * the same metrics on the "broker-called-SN" subset: rows where any of
    ``--called-sn-expert`` (default alerce/stamp_classifier_rubin_beta, fink_lsst/snn)
    projects p_sn >= 0.5;
  * when the predictions carry ``call_trust__<expert>`` (scorer, v13g call-trust heads):
    per-expert reliability (10 equal-width bins, ECE) of call_trust against whether that
    expert's SN-vs-not call was correct.

Inputs: repeated ``--set name=<gold.parquet>,<predictions.parquet>[,<truth.parquet>][,<ids>]``.
Gold carries ``proj__<expert>__p_snia / __p_nonIa_snlike`` and ``avail__<expert>``;
predictions ``p_snia, p_nonia, p_other, target_class, n_det, object_id``.  Rows are paired
on (object_id, n_det).  The optional extras are told apart by their file type: a parquet
with ``object_id`` + ``final_class_ternary`` (and optionally ``label_quality``) overrides
the predictions' labels; a json (``test_ids`` / ``ids`` / ``object_ids`` list, or a plain
list) or csv (``object_id`` column) restricts the object ids.  ``--label-qualities``
(default spectroscopic,context, as scripts/eval_input_ablation.py; ``all`` = no filter)
applies wherever a ``label_quality`` is known.

``--positives-from DONOR``: a set with no SNe at all (hard negatives) is scored against the
donor set's SNe at the same slice (``TARGET=DONOR`` forces it for one set).  Its own SNe,
if any, are dropped there; reliability tables still use each set's own rows.

Object ids stay strings end to end (Rubin diaObjectIds ~1e17 corrupt through float64; a
float-typed id column is refused).

    python scripts/eval_rubin_sets.py \\
        --set bench=gold_full.parquet,predictions_abl_full_v13f.parquet,truth.parquet,ids.csv \\
        --set hardneg=gold_hn.parquet,pred_hn.parquet --positives-from bench --out reports/eval_v13f
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SN_CLASSES = {"snia", "nonIa_snlike"}
DEFAULT_SLICES = ("3", "5", "10", "latest")
DEFAULT_CALLED = ("alerce/stamp_classifier_rubin_beta", "fink_lsst/snn")
BASELINE_NOTE = ("The best single input is picked on the same objects it is scored on (a test-set "
                 "selection), which favours the baseline; the paired difference is conservative.")
N_BINS = 10


def _san(key: str) -> str:
    return str(key).replace("/", "__")


def string_ids(series: pd.Series, what: str = "object_id") -> pd.Series:
    """Object ids as strings; refuses a float column (ids ~1e17 lose digits in float64)."""
    if pd.api.types.is_float_dtype(series):
        raise ValueError(f"{what} is float-typed: Rubin diaObjectIds corrupt through float64 "
                         "(re-export the table with string ids)")
    return series.astype(str).str.strip()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _read_ids(path: Path) -> set[str]:
    if path.suffix == ".csv":
        return set(string_ids(pd.read_csv(path, dtype={"object_id": str})["object_id"]))
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        for key in ("test_ids", "ids", "object_ids"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raise ValueError(f"{path}: no test_ids / ids / object_ids list")
    return {str(x).strip() for x in raw}


def _expert_p_sn(gold: pd.DataFrame) -> pd.DataFrame:
    """``<san>`` -> projected p_sn (NaN where the expert is absent), keyed like ``gold``."""
    out: dict[str, np.ndarray] = {}
    for col in gold.columns:
        if not (col.startswith("proj__") and col.endswith("__p_snia")):
            continue
        san = col[len("proj__"):-len("__p_snia")]
        a = pd.to_numeric(gold[col], errors="coerce")
        bcol = f"proj__{san}__p_nonIa_snlike"
        b = pd.to_numeric(gold[bcol], errors="coerce") if bcol in gold.columns else pd.Series(np.nan, index=gold.index)
        psn = np.where(a.isna() & b.isna(), np.nan, a.fillna(0.0) + b.fillna(0.0))
        acol = f"avail__{san}"
        if acol in gold.columns:
            psn = np.where(pd.to_numeric(gold[acol], errors="coerce").fillna(0.0).to_numpy(float) > 0, psn, np.nan)
        out[san] = psn
    return pd.DataFrame(out, index=gold.index)


def load_set(
    name: str, gold_path: Path, pred_path: Path, extras: list[Path], label_qualities: set[str] | None,
) -> pd.DataFrame:
    """One row per (object, n_det) with ``object_id`` (str), ``n_det`` (int), ``y`` (1 = SN),
    ``psn``, ``label_quality``, ``e__<expert>`` (projected p_sn, NaN = absent) and the
    ``call_trust__`` / ``sn_call__`` columns of the predictions."""
    pred = pd.read_parquet(pred_path)
    gold = pd.read_parquet(gold_path)
    for frame, what in ((pred, f"{name} predictions"), (gold, f"{name} gold")):
        frame["object_id"] = string_ids(frame["object_id"], what)
        frame["n_det"] = np.rint(pd.to_numeric(frame["n_det"], errors="coerce")).astype("Int64")
    pred = pred.dropna(subset=["n_det"])
    gold = gold.dropna(subset=["n_det"]).drop_duplicates(["object_id", "n_det"])
    e = _expert_p_sn(gold)
    e.columns = [f"e__{c}" for c in e.columns]
    gold_part = pd.concat([gold[["object_id", "n_det"]], e], axis=1)
    gold_extra = [c for c in gold.columns if c.startswith(("call_trust__", "sn_call__"))]
    rows = pred.merge(pd.concat([gold_part, gold[gold_extra]], axis=1), on=["object_id", "n_det"], how="left",
                      suffixes=("", "_gold"), validate="many_to_one")
    rows["n_det"] = rows["n_det"].astype(int)
    rows = rows.drop_duplicates(["object_id", "n_det"], keep="last").reset_index(drop=True)
    rows["psn"] = pd.to_numeric(rows["p_snia"], errors="coerce") + pd.to_numeric(rows["p_nonia"], errors="coerce")

    label = rows["target_class"] if "target_class" in rows.columns else pd.Series(np.nan, index=rows.index)
    quality = rows["label_quality"] if "label_quality" in rows.columns else pd.Series(np.nan, index=rows.index)
    for path in extras:
        if path.suffix == ".parquet":
            truth = pd.read_parquet(path)
            truth["object_id"] = string_ids(truth["object_id"], f"{name} truth")
            truth = truth.drop_duplicates("object_id").set_index("object_id")
            label = rows["object_id"].map(truth["final_class_ternary"])
            if "label_quality" in truth.columns:
                quality = rows["object_id"].map(truth["label_quality"])
        else:
            rows = rows[rows["object_id"].isin(_read_ids(path))]
            label, quality = label.loc[rows.index], quality.loc[rows.index]
    rows = rows.assign(_label=label.loc[rows.index].to_numpy(), label_quality=quality.loc[rows.index].to_numpy())
    keep = rows["_label"].notna() & rows["psn"].notna()
    if label_qualities is not None and rows["label_quality"].notna().any():   # nothing to filter on otherwise
        keep &= rows["label_quality"].isin(label_qualities)
    rows = rows[keep].copy()
    rows["y"] = rows["_label"].isin(SN_CLASSES).astype(int)
    return rows.drop(columns=["_label"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def auc_columns(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ROC AUC of every column of ``scores`` (n x k) against ``y`` (ties by mid-rank); NaN if one class."""
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.full(scores.shape[1], np.nan)
    ranks = rankdata(scores, axis=0)
    return (ranks[y].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _r(x: Any, nd: int = 3) -> Any:
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def slice_metrics(frame: pd.DataFrame, experts: list[str], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Metrics of one slice (one row per object).  ``frame`` has ``y``, ``psn``, ``cluster``, ``e__<expert>``."""
    y = frame["y"].to_numpy(int)
    psn = frame["psn"].to_numpy(float)
    n, n_sn = len(frame), int(y.sum())
    out: dict[str, Any] = {"n": n, "n_sn": n_sn}
    if n == 0:
        return out
    out.update({
        "brier": _r(np.mean((psn - y) ** 2)),
        "median_psn_sn": _r(np.median(psn[y == 1])) if n_sn else None,
        "median_psn_other": _r(np.median(psn[y == 0])) if n_sn < n else None,
        "frac_other_psn_gt_half": _r(np.mean(psn[y == 0] > 0.5)) if n_sn < n else None,
    })
    cols = [f"e__{e}" for e in experts if f"e__{e}" in frame.columns]
    E = np.column_stack([frame[c].fillna(0.5).to_numpy(float) for c in cols]) if cols else np.zeros((n, 0))
    S = np.column_stack([psn, E])
    point = auc_columns(S, y)
    out["auc"], out["auc_ci"] = _r(point[0]), None
    if n_sn == 0 or n_sn == n:
        out["baseline"] = None
        return out
    clusters = pd.factorize(frame["cluster"])[0]
    assert len(np.unique(clusters)) == n, "slice must hold one row per object"
    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, S.shape[1]), np.nan)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        boot[b] = auc_columns(S[i], y[i])
    lo, hi = np.nanpercentile(boot[:, 0], [2.5, 97.5])
    out["auc_ci"] = [_r(lo), _r(hi)]
    base: dict[str, Any] = {"note": BASELINE_NOTE, "per_expert": {}}
    for j, col in enumerate(cols):
        avail = int(frame[col].notna().sum())
        base["per_expert"][col[len("e__"):]] = {"auc": _r(point[1 + j]), "n_available": avail}
    if cols and np.isfinite(point[1:]).any():
        j = int(np.nanargmax(point[1:]))
        diff = boot[:, 0] - boot[:, 1 + j]
        dlo, dhi = np.nanpercentile(diff, [2.5, 97.5])
        base.update({
            "best": cols[j][len("e__"):], "best_auc": _r(point[1 + j]),
            "best_auc_ci": [_r(v) for v in np.nanpercentile(boot[:, 1 + j], [2.5, 97.5])],
            "diff_model_minus_best": _r(point[0] - point[1 + j]), "diff_ci": [_r(dlo), _r(dhi)],
        })
    out["baseline"] = base
    return out


def last_per_object(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("n_det", kind="stable").groupby("object_id", sort=False).tail(1)


def take_slice(frame: pd.DataFrame, sl: str) -> pd.DataFrame:
    return last_per_object(frame) if sl == "latest" else frame[frame["n_det"] == int(sl)]


def called_sn(frame: pd.DataFrame, called: list[str]) -> np.ndarray:
    hit = np.zeros(len(frame), dtype=bool)
    for key in called:
        col = f"e__{_san(key)}"
        if col in frame.columns:
            hit |= frame[col].to_numpy(float) >= 0.5        # NaN (absent) compares False
    return hit


def reliability(conf: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    """10-bin reliability of ``conf`` against 0/1 ``correct`` and the (count-weighted) ECE."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    if n == 0:
        return {"n": 0}
    idx = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    bins, ece = [], 0.0
    for b in range(N_BINS):
        m = idx == b
        if m.any():
            bins.append({"bin": [round(b / N_BINS, 2), round((b + 1) / N_BINS, 2)], "n": int(m.sum()),
                         "mean_call_trust": _r(conf[m].mean()), "frac_correct": _r(correct[m].mean())})
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return {"n": n, "ece": _r(ece), "mean_call_trust": _r(conf.mean()), "frac_correct": _r(correct.mean()),
            "bins": bins}


def call_trust_tables(rows: pd.DataFrame, slices: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    experts = sorted(c[len("call_trust__"):] for c in rows.columns if c.startswith("call_trust__"))
    for san in experts:
        per = {}
        for sl in slices:
            s = take_slice(rows, sl)
            conf = pd.to_numeric(s[f"call_trust__{san}"], errors="coerce").to_numpy(float)
            if f"sn_call__{san}" in s.columns:
                call = pd.to_numeric(s[f"sn_call__{san}"], errors="coerce").to_numpy(float)
            else:                                           # the expert's call from its projection in the gold
                psn_e = (s[f"e__{san}"].to_numpy(float) if f"e__{san}" in s.columns else np.full(len(s), np.nan))
                call = np.where(np.isfinite(psn_e), (psn_e >= 0.5).astype(float), np.nan)
            ok = np.isfinite(conf) & np.isfinite(call)
            per[sl] = reliability(conf[ok], (call[ok] == s["y"].to_numpy(float)[ok]).astype(float))
        out[san] = per
    return out


def evaluate(
    sets: dict[str, pd.DataFrame], *, slices: list[str], called: list[str], donors: dict[str, str],
    n_boot: int, seed: int,
) -> dict[str, Any]:
    experts = sorted({c[len("e__"):] for df in sets.values() for c in df.columns if c.startswith("e__")})
    result: dict[str, Any] = {}
    for name, rows in sets.items():
        donor = donors.get(name)
        entry: dict[str, Any] = {"n_rows": int(len(rows)), "n_objects": int(rows["object_id"].nunique()),
                                 "positives_from": donor, "slices": {}}
        for sl in slices:
            own = take_slice(rows, sl)
            if donor:
                pos = take_slice(sets[donor], sl)
                pos = pos[pos["y"] == 1].assign(cluster=lambda d: donor + "/" + d["object_id"])
                neg = own[own["y"] == 0].assign(cluster=lambda d: name + "/" + d["object_id"])
                frame = pd.concat([neg, pos], ignore_index=True)
                extra = {"n_own_sn_dropped": int(own["y"].sum()), "positives_from": donor}
            else:
                frame, extra = own.assign(cluster=lambda d: name + "/" + d["object_id"]), {}
            cell = {"all": slice_metrics(frame, experts, n_boot=n_boot, seed=seed) | extra}
            sub = frame[called_sn(frame, called)]
            cell["broker_called_sn"] = slice_metrics(sub, experts, n_boot=n_boot, seed=seed)
            entry["slices"][sl] = cell
        entry["call_trust_reliability"] = call_trust_tables(rows, slices)
        result[name] = entry
    return result


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _ci(v: list | None) -> str:
    return "—" if not v else f"[{v[0]:.3f}, {v[1]:.3f}]"


def _f(x: Any, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def to_markdown(result: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = ["# metaDEBASS Rubin set evaluation", "",
             f"Bootstrap: {meta['n_boot']} object-level resamples, seed {meta['seed']}; 95% percentile intervals. "
             f"Label qualities: {meta['label_qualities']}. P(SN) = p_snia + p_nonia.", "",
             f"Baseline: {BASELINE_NOTE}", ""]
    for name, entry in result.items():
        head = f"## {name} ({entry['n_objects']} objects, {entry['n_rows']} rows)"
        if entry["positives_from"]:
            head += f" — SNe borrowed from `{entry['positives_from']}`"
        lines += [head, ""]
        for part, title in (("all", "All rows"), ("broker_called_sn",
                                                  "Broker-called-SN rows (" + ", ".join(meta["called"]) + ")")):
            lines += [f"### {title}", "",
                      "| slice | n | n SN | AUC [95% CI] | Brier | med P(SN) SNe | med P(SN) others | others P(SN)>0.5 "
                      "| best single input (AUC) | metaDEBASS − best [95% CI] |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for sl, cell in entry["slices"].items():
                m = cell[part]
                b = m.get("baseline") or {}
                best = f"{b['best']} ({_f(b['best_auc'])})" if b.get("best") else "—"
                diff = (f"{b['diff_model_minus_best']:+.3f} {_ci(b['diff_ci'])}" if b.get("best") else "—")
                lines.append(f"| {sl} | {m.get('n', 0)} | {m.get('n_sn', 0)} | {_f(m.get('auc'))} {_ci(m.get('auc_ci'))} "
                             f"| {_f(m.get('brier'))} | {_f(m.get('median_psn_sn'), 2)} | {_f(m.get('median_psn_other'), 2)} "
                             f"| {_f(m.get('frac_other_psn_gt_half'), 2)} | {best} | {diff} |")
            lines.append("")
        ct = entry["call_trust_reliability"]
        if ct:
            lines += ["### Call-trust reliability (ECE; n rows with an SN-vs-not call)", "",
                      "| expert | " + " | ".join(entry["slices"]) + " |", "|---|" + "---|" * len(entry["slices"])]
            for san, per in ct.items():
                lines.append(f"| {san} | " + " | ".join(
                    f"{_f(per[sl].get('ece'))} (n={per[sl].get('n', 0)})" for sl in entry["slices"]) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def parse_set(spec: str) -> tuple[str, Path, Path, list[Path]]:
    name, _, rest = spec.partition("=")
    parts = [p for p in rest.split(",") if p]
    if not name or len(parts) < 2:
        raise SystemExit(f"--set expects NAME=GOLD.parquet,PREDICTIONS.parquet[,TRUTH][,IDS]; got {spec!r}")
    return name, Path(parts[0]), Path(parts[1]), [Path(p) for p in parts[2:]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, dest="sets", metavar="NAME=GOLD,PRED[,TRUTH][,IDS]")
    ap.add_argument("--positives-from", action="append", default=[], metavar="DONOR | TARGET=DONOR",
                    help="borrow SN positives from DONOR for every other set with no SNe (or for TARGET)")
    ap.add_argument("--called-sn-expert", action="append", default=None, metavar="EXPERT",
                    help=f"broker-called-SN experts (default {', '.join(DEFAULT_CALLED)}); repeatable")
    ap.add_argument("--label-qualities", default="spectroscopic,context",
                    help="comma list of label qualities kept where known, or 'all'")
    ap.add_argument("--slices", default=",".join(DEFAULT_SLICES), help="n_det checkpoints and/or 'latest'")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path, help="directory, or a .json / .md path (both are written)")
    a = ap.parse_args(argv)

    qualities = None if a.label_qualities == "all" else {q.strip() for q in a.label_qualities.split(",") if q.strip()}
    slices = [s.strip() for s in a.slices.split(",") if s.strip()]
    called = list(a.called_sn_expert or DEFAULT_CALLED)
    sets: dict[str, pd.DataFrame] = {}
    for spec in a.sets:
        name, gold, pred, extras = parse_set(spec)
        if name in sets:
            raise SystemExit(f"duplicate set name {name!r}")
        sets[name] = load_set(name, gold, pred, extras, qualities)
        print(f"  {name}: {len(sets[name]):,} rows, {sets[name]['object_id'].nunique():,} objects, "
              f"{int(last_per_object(sets[name])['y'].sum())} SNe at latest", flush=True)
    donors: dict[str, str] = {}
    for spec in a.positives_from:
        target, sep, donor = spec.partition("=")
        donor = donor if sep else target
        if donor not in sets:
            raise SystemExit(f"--positives-from: unknown set {donor!r}")
        for name in ([target] if sep else [n for n in sets if n != donor and sets[n]["y"].sum() == 0]):
            if name not in sets:
                raise SystemExit(f"--positives-from: unknown set {name!r}")
            donors[name] = donor

    result = evaluate(sets, slices=slices, called=called, donors=donors, n_boot=a.n_boot, seed=a.seed)
    meta = {"n_boot": a.n_boot, "seed": a.seed, "slices": slices, "called": called,
            "label_qualities": "all" if qualities is None else sorted(qualities), "baseline_note": BASELINE_NOTE}
    base = a.out.with_suffix("") if a.out.suffix in (".json", ".md") else a.out / "eval_rubin_sets"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps({"meta": meta, "sets": result}, indent=1))
    md = to_markdown(result, meta)
    base.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
