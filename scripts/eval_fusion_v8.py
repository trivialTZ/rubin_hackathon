#!/usr/bin/env python3
"""fusion_v8 honest evaluation — Tables 1–7 + pre-registered headline & guards.

Protocol (design §6, spec corrections #2/#3/#4b):
  * object-level — exactly one row per object per n_det slice;
  * locked test IDs only (from data/gold/split_fusion_v8.json), spectroscopic
    labels only (spec+tns_untyped only where explicitly stated);
  * 1,000-resample object bootstrap 95% CIs on every cell;
  * best-single-expert baseline chosen on CAL, scored on test;
  * v6e2 comparator is RE-SCORED under this same protocol (models/trust +
    models/followup applied to the fusion snapshot), never its stale row-level
    numbers;
  * CI discipline (spec correction #3): the local run gates correctness +
    honesty — any delta whose CI95 includes 0 is reported as "not significant",
    never claimed as a win.  'other' OvR is informational-only locally.

Graceful degradation: every table runs with whatever subset of artifacts exists
and writes a clearly-marked unavailable status otherwise (local v6e2 artifacts
cover only 3 experts).

Outputs: reports/fusion_v8/tables_{1..7}.{md,json} + headline_guards.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

try:
    from debass_meta.models.multiclass_followup import CLASSES
except Exception:  # pragma: no cover — G4 module built in parallel
    CLASSES = ("snia", "nonIa_snlike", "other")

CLASS_KEYS = ("snia", "nonia", "other")          # short metric keys, CLASSES order
SPEC_QUALITIES = ("spectroscopic",)
INFO_QUALITIES = ("spectroscopic", "tns_untyped")
NDET_SLICES = (3, 5, 10, 20)
TRUST_NDET_BUCKETS = (3, 4, 5)                   # PI contract: trust at 3-5 detections
SURVEY_SLICES = ("all", "ztf", "lsst")
RANK_NDETS = (3, 5, 10)
RANK_BUDGETS = (20, 50, 100)
GOAL_TO_CLASS = {"ia": 0, "nonia": 1, "other": 2}
DP1_ECLBIN_REFERENCE_EF = 17.88                  # documented v6e2 DP1 headline


# ── generic helpers ──────────────────────────────────────────────────────────

def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_split_manifest(path: Path) -> tuple[set[str], set[str], set[str], dict]:
    with open(path) as fh:
        raw = json.load(fh)
    node = raw.get("split", raw) if isinstance(raw, dict) else raw
    parts: list[set[str]] = []
    for part in ("train", "cal", "test"):
        ids = node.get(f"{part}_ids", node.get(part))
        if ids is None:
            raise KeyError(f"Split manifest {path} missing '{part}_ids'")
        parts.append({str(x) for x in ids})
    train_ids, cal_ids, test_ids = parts
    assert not (train_ids & cal_ids), "split integrity: train/cal overlap"
    assert not (train_ids & test_ids), "split integrity: train/test overlap"
    assert not (cal_ids & test_ids), "split integrity: cal/test overlap"
    return train_ids, cal_ids, test_ids, raw


def class_index(target_class: pd.Series) -> np.ndarray:
    mapping = {c: i for i, c in enumerate(CLASSES)}
    return target_class.astype(str).map(mapping).fillna(-1).astype(int).to_numpy()


def ndet_bucket(n_det: np.ndarray) -> np.ndarray:
    n = np.asarray(n_det, dtype=float)
    out = np.full(n.shape, "10-20", dtype=object)
    out[(n >= 1) & (n <= 4)] = "1-4"
    out[(n >= 5) & (n <= 9)] = "5-9"
    return out


def _latest_per_object(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["object_id", "n_det"])
        .groupby("object_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _fmt(v: Any, prec: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        val, lo, hi = v.get("value"), v.get("lo"), v.get("hi")
        if val is None:
            return "—"
        if lo is None or hi is None:
            return f"{val:.{prec}f}"
        return f"{val:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"
    if isinstance(v, float):
        return "—" if np.isnan(v) else f"{v:.{prec}f}"
    return str(v)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _write_table(out_dir: Path, n: int, payload: dict, md: str) -> None:
    (out_dir / f"tables_{n}.json").write_text(json.dumps(_jsonify(payload), indent=2))
    (out_dir / f"tables_{n}.md").write_text(md + "\n")
    print(f"  wrote tables_{n}.md / tables_{n}.json")


def _unavailable(out_dir: Path, n: int, title: str, reason: str) -> None:
    _write_table(out_dir, n, {"status": "unavailable", "reason": reason},
                 f"## {title}\n\n**UNAVAILABLE** — {reason}")


# ── metric machinery ─────────────────────────────────────────────────────────

def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(p)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y[mask], p[mask]))
    except Exception:
        return None


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float | None:
    """Equal-width ECE with n_bins (design Table 2 spec: 15 bins)."""
    mask = np.isfinite(p)
    y, p = np.asarray(y, float)[mask], np.asarray(p, float)[mask]
    if len(y) < 10:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n, ece = len(y), 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def matrix_metrics(y_idx: np.ndarray, proba: np.ndarray, *, binary_only: bool = False) -> dict:
    """OvR AUC ×3, macro, balanced acc, log-loss for an (N,3) probability matrix.

    NaN columns (e.g. a binary v6e2 score in slot 0 only) yield None metrics;
    binary_only suppresses macro/balanced/log-loss so a 1-class score is never
    presented as a ternary classifier.
    """
    from sklearn.metrics import balanced_accuracy_score, log_loss

    out: dict[str, float | None] = {}
    aucs: list[float] = []
    for ci, key in enumerate(CLASS_KEYS):
        auc = _safe_auc((y_idx == ci).astype(int), proba[:, ci])
        out[f"auc_{key}"] = auc
        if auc is not None:
            aucs.append(auc)
    if binary_only:
        out["macro_auc"] = None
        out["balanced_acc"] = None
        out["log_loss"] = None
        return out
    out["macro_auc"] = float(np.mean(aucs)) if aucs else None
    full = np.isfinite(proba).all(axis=1)
    if full.sum() >= 10 and len(np.unique(y_idx[full])) >= 2:
        P = np.clip(proba[full], 1e-6, 1.0)
        P = P / P.sum(axis=1, keepdims=True)
        try:
            out["balanced_acc"] = float(
                balanced_accuracy_score(y_idx[full], P.argmax(axis=1)))
        except Exception:
            out["balanced_acc"] = None
        try:
            out["log_loss"] = float(
                log_loss(y_idx[full], P, labels=list(range(len(CLASSES)))))
        except Exception:
            out["log_loss"] = None
    else:
        out["balanced_acc"] = None
        out["log_loss"] = None
    return out


def bootstrap_dict(
    fn: Callable[[np.ndarray], dict],
    n_rows: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """Object bootstrap of a dict-of-metrics function (rows are object-level)."""
    point = fn(np.arange(n_rows))
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        m = fn(idx)
        for k, v in m.items():
            if v is not None:
                samples[k].append(v)
    out: dict[str, dict] = {}
    for k, v in point.items():
        s = samples[k]
        ok = len(s) >= max(20, n_boot // 4)
        out[k] = {
            "value": v,
            "lo": float(np.percentile(s, 2.5)) if ok else None,
            "hi": float(np.percentile(s, 97.5)) if ok else None,
            "n_boot_ok": len(s),
        }
    return out


def _purity_at_k(y_is_target: np.ndarray, score: np.ndarray, k: int) -> float | None:
    mask = np.isfinite(score)
    if mask.sum() < 10:
        return None
    order = np.argsort(-score[mask], kind="stable")
    top = order[: min(k, mask.sum())]
    return float(np.asarray(y_is_target, float)[mask][top].mean())


def _recall_at_k(y_is_target: np.ndarray, score: np.ndarray, k: int) -> float | None:
    mask = np.isfinite(score)
    n_target = int(np.asarray(y_is_target, bool)[mask].sum())
    if mask.sum() < 10 or n_target == 0:
        return None
    order = np.argsort(-score[mask], kind="stable")
    top = order[: min(k, mask.sum())]
    return float(np.asarray(y_is_target, float)[mask][top].sum() / n_target)


def _ndcg_at_k(y_is_target: np.ndarray, score: np.ndarray, k: int = 50) -> float | None:
    mask = np.isfinite(score)
    y = np.asarray(y_is_target, float)[mask]
    if mask.sum() < 10 or y.sum() == 0:
        return None
    order = np.argsort(-score[mask], kind="stable")
    rel = y[order][:k]
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float((rel * discounts).sum())
    ideal = np.sort(y)[::-1][:k]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else None


# ── method probability builders ──────────────────────────────────────────────

def proj_sans(df: pd.DataFrame) -> list[str]:
    return sorted({
        m.group(1) for c in df.columns
        for m in [re.match(r"proj__(.+)__p_snia$", c)] if m
    })


def expert_matrix(df: pd.DataFrame, san: str) -> np.ndarray:
    """(N,3) ternary projection of one expert; NaN where not available."""
    cols = [f"proj__{san}__p_snia", f"proj__{san}__p_nonIa_snlike", f"proj__{san}__p_other"]
    P = np.full((len(df), 3), np.nan)
    for i, c in enumerate(cols):
        if c in df.columns:
            P[:, i] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    avail_col = f"avail__{san}"
    if avail_col in df.columns:
        avail = df[avail_col].fillna(0).astype(bool).to_numpy()
        P[~avail] = np.nan
    return P


def mean_fusion_matrix(df: pd.DataFrame) -> np.ndarray:
    """Uniform average of available experts' ternary vectors, renormalized."""
    sans = proj_sans(df)
    acc = np.zeros((len(df), 3))
    cnt = np.zeros((len(df), 3))
    for san in sans:
        P = expert_matrix(df, san)
        ok = np.isfinite(P)
        acc[ok] += P[ok]
        cnt[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    rowsum = np.nansum(M, axis=1)
    valid = np.isfinite(M).all(axis=1) & (rowsum > 0)
    M[valid] = M[valid] / rowsum[valid, None]
    M[~valid] = np.nan
    return M


def trust_pool_matrix(df: pd.DataFrame) -> np.ndarray:
    """NaN-aware trust-weighted pool: Σ avail·q·p_c / Σ avail·q (design §1.5).

    Absent expert (avail=0, NaN proj, or NaN q) contributes nothing; missing q
    column degrades to weight 1.0 for that expert.
    """
    sans = proj_sans(df)
    num = np.zeros((len(df), 3))
    den = np.zeros(len(df))
    for san in sans:
        P = expert_matrix(df, san)
        q_col = f"q__{san}"
        q = (
            pd.to_numeric(df[q_col], errors="coerce").to_numpy(float)
            if q_col in df.columns else np.ones(len(df))
        )
        usable = np.isfinite(P).all(axis=1) & np.isfinite(q) & (q > 0)
        num[usable] += P[usable] * q[usable, None]
        den[usable] += q[usable]
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(den[:, None] > 0, num / np.maximum(den[:, None], 1e-12), np.nan)
    rowsum = M.sum(axis=1)
    valid = np.isfinite(M).all(axis=1) & (rowsum > 0)
    M[valid] = M[valid] / rowsum[valid, None]
    return M


def choose_best_single(frame: pd.DataFrame, cal_ids: set[str]) -> tuple[str | None, dict]:
    """Best-single expert chosen on CAL macro OvR AUC (fixes oracle-on-eval bug)."""
    cal = frame[
        frame["object_id"].astype(str).isin(cal_ids)
        & frame["label_quality"].astype(str).isin(SPEC_QUALITIES)
        & frame["target_class"].notna()
    ]
    slice5 = cal[cal["n_det"] == 5]
    pool = slice5 if len(slice5) >= 30 else _latest_per_object(cal)
    y = class_index(pool["target_class"])
    scores: dict[str, float] = {}
    for san in proj_sans(pool):
        m = matrix_metrics(y, expert_matrix(pool, san))
        if m["macro_auc"] is not None:
            scores[san] = m["macro_auc"]
    if not scores:
        return None, {}
    best = max(scores, key=lambda k: (scores[k], k))
    return best, scores


def v6e2_rescore(frame: pd.DataFrame, trust_dir: Path, followup_dir: Path) -> tuple[np.ndarray | None, str | None]:
    """Re-score the v6e2 stack (3 local trust heads + binary followup) on the
    fusion snapshot under the v8 protocol.  Returns (p_follow array, reason)."""
    if not (followup_dir / "metadata.json").exists():
        return None, f"v6e2 followup artifact missing at {followup_dir}"
    try:
        from debass_meta.models.expert_trust import ExpertTrustArtifact
        from debass_meta.models.followup import FollowupArtifact
        from debass_meta.projectors import sanitize_expert_key
    except Exception as exc:
        return None, f"v6e2 modules not importable: {exc}"

    work = frame.copy()
    n_heads = 0
    if trust_dir.is_dir():
        for sub in sorted(trust_dir.iterdir()):
            if not sub.is_dir() or not (sub / "metadata.json").exists():
                continue
            try:
                art = ExpertTrustArtifact.load(sub)
            except Exception:
                continue
            san = sanitize_expert_key(art.expert_key)
            q = np.asarray(art.predict_trust(work), dtype=float)
            avail_col = f"avail__{san}"
            if avail_col in work.columns:
                avail = work[avail_col].fillna(0).astype(bool).to_numpy()
                # v6e2 semantics: q=0.0 where unavailable (the documented conflation)
                q = np.where(avail, q, 0.0)
            work[f"q__{san}"] = q
            n_heads += 1
    if n_heads == 0:
        return None, f"no v6e2 trust heads under {trust_dir}"
    fu = FollowupArtifact.load(followup_dir)
    return np.asarray(fu.predict_followup(work), dtype=float), None


# ── Table 1 — classification per slice ───────────────────────────────────────

def build_table_1(
    frame: pd.DataFrame,
    test_ids: set[str],
    cal_ids: set[str],
    *,
    v6e2: np.ndarray | None,
    v6e2_reason: str | None,
    n_boot: int,
    rng: np.random.Generator,
    split_raw: dict,
) -> tuple[dict, str]:
    frame = frame.copy()
    if v6e2 is not None:
        frame["_p_v6e2"] = v6e2
    best_san, cal_scores = choose_best_single(frame, cal_ids)

    def _slice_rows(sub: pd.DataFrame, informational: bool) -> list[dict]:
        y = class_index(sub["target_class"])
        methods: list[tuple[str, np.ndarray, bool]] = []
        if {"p_snia", "p_nonia", "p_other"} <= set(sub.columns):
            methods.append(("fusion_v8",
                            sub[["p_snia", "p_nonia", "p_other"]].to_numpy(float), False))
        if "_p_v6e2" in sub.columns:
            P = np.full((len(sub), 3), np.nan)
            P[:, 0] = sub["_p_v6e2"].to_numpy(float)
            methods.append(("v6e2_rescored", P, True))
        if best_san is not None:
            methods.append((f"best_single_cal[{best_san}]",
                            expert_matrix(sub, best_san), False))
        methods.append(("mean_fusion", mean_fusion_matrix(sub), False))
        methods.append(("trust_weighted_pool", trust_pool_matrix(sub), False))
        rows = []
        for name, P, binary_only in methods:
            cis = bootstrap_dict(
                lambda idx, P=P, b=binary_only: matrix_metrics(y[idx], P[idx], binary_only=b),
                len(sub), n_boot, rng,
            )
            rows.append({
                "method": name, "binary_only": binary_only,
                "informational": informational,
                "n_objects": int(len(sub)),
                "n_per_class": {k: int((y == i).sum()) for i, k in enumerate(CLASS_KEYS)},
                "metrics": cis,
            })
        return rows

    def _run(ids: set[str], qualities: tuple[str, ...], informational: bool) -> list[dict]:
        base = frame[
            frame["object_id"].astype(str).isin(ids)
            & frame["label_quality"].astype(str).isin(qualities)
            & frame["target_class"].notna()
        ]
        out = []
        for n in NDET_SLICES:
            for survey in SURVEY_SLICES:
                sub = base[base["n_det"] == n]
                if survey != "all":
                    sub = sub[sub["_survey"] == survey]
                if len(sub) < 10:
                    out.append({"n_det": n, "survey": survey, "status": "too_few_rows",
                                "n_objects": int(len(sub)), "informational": informational})
                    continue
                assert sub["object_id"].nunique() == len(sub), \
                    "object-level violation: >1 row per object in a n_det slice"
                for row in _slice_rows(sub, informational):
                    out.append({"n_det": n, "survey": survey, **row})
        return out

    primary = _run(test_ids, SPEC_QUALITIES, informational=False)

    # Secondary, clearly-informational table (spec correction #3): expanded-test
    # IDs when the split manifest provides them, else locked test with
    # spec+tns_untyped labels.  'other' OvR is reportable locally ONLY here.
    node = split_raw.get("split", split_raw) if isinstance(split_raw, dict) else {}
    expanded = node.get("expanded_test_ids")
    if expanded:
        secondary = _run({str(x) for x in expanded}, INFO_QUALITIES, informational=True)
        secondary_label = "expanded-test IDs, spec+tns_untyped (INFORMATIONAL ONLY)"
    else:
        secondary = _run(test_ids, INFO_QUALITIES, informational=True)
        secondary_label = "locked test, spec+tns_untyped (INFORMATIONAL ONLY)"

    payload = {
        "table": 1, "title": "Classification per slice (object-level, locked test, spec-only)",
        "protocol": {"n_boot": n_boot, "qualities": list(SPEC_QUALITIES),
                     "best_single_cal_chosen": best_san,
                     "best_single_cal_macro_aucs": cal_scores,
                     "v6e2_rescored": v6e2 is not None, "v6e2_reason": v6e2_reason},
        "rows": primary,
        "secondary_informational": {"label": secondary_label, "rows": secondary},
    }
    headers = ["n_det", "survey", "method", "AUC snia", "AUC nonIa", "AUC other",
               "macro AUC", "bal. acc", "log-loss", "n"]
    md_rows = []
    for r in primary:
        if r.get("status"):
            md_rows.append([r["n_det"], r["survey"], f"({r['status']})",
                            "—", "—", "—", "—", "—", "—", r["n_objects"]])
            continue
        m = r["metrics"]
        md_rows.append([
            r["n_det"], r["survey"], r["method"],
            _fmt(m["auc_snia"]), _fmt(m["auc_nonia"]), _fmt(m["auc_other"]),
            _fmt(m["macro_auc"]), _fmt(m["balanced_acc"]), _fmt(m["log_loss"]),
            r["n_objects"],
        ])
    md = ("## Table 1 — Classification per slice (object-level, locked test, spec-only)\n\n"
          f"best-single expert chosen on cal: `{best_san}`; "
          f"v6e2 re-scored: {v6e2 is not None}"
          + (f" ({v6e2_reason})" if v6e2_reason else "") + "\n\n"
          + _md_table(headers, md_rows)
          + f"\n\n### Secondary — {secondary_label}\n\n"
          + _md_table(headers, [
              [r["n_det"], r["survey"], r.get("method", f"({r.get('status')})"),
               _fmt(r.get("metrics", {}).get("auc_snia")),
               _fmt(r.get("metrics", {}).get("auc_nonia")),
               _fmt(r.get("metrics", {}).get("auc_other")),
               _fmt(r.get("metrics", {}).get("macro_auc")),
               _fmt(r.get("metrics", {}).get("balanced_acc")),
               _fmt(r.get("metrics", {}).get("log_loss")),
               r["n_objects"]]
              for r in secondary if r.get("method") == "fusion_v8" or r.get("status")]))
    return payload, md


# ── Table 2 — calibration ────────────────────────────────────────────────────

def build_table_2(frame: pd.DataFrame, test_ids: set[str], n_boot: int,
                  rng: np.random.Generator) -> tuple[dict, str]:
    need = {"p_snia", "p_nonia", "p_other", "p_snia_raw", "p_nonia_raw", "p_other_raw"}
    if not need <= set(frame.columns):
        raise FileNotFoundError(f"predictions lack raw+calibrated columns {sorted(need)}")
    base = _latest_per_object(frame[
        frame["object_id"].astype(str).isin(test_ids)
        & frame["label_quality"].astype(str).isin(SPEC_QUALITIES)
        & frame["target_class"].notna()
    ])
    rows, reliability = [], {}
    for survey in SURVEY_SLICES:
        sub = base if survey == "all" else base[base["_survey"] == survey]
        if len(sub) < 10:
            rows.append({"survey": survey, "status": "too_few_rows", "n_objects": int(len(sub))})
            continue
        y_idx = class_index(sub["target_class"])
        for ci, (key, cls) in enumerate(zip(CLASS_KEYS, CLASSES)):
            y = (y_idx == ci).astype(int)
            for kind, col in (("raw", f"p_{key}_raw"), ("calibrated", f"p_{key}")):
                p = sub[col].to_numpy(float)

                def _m(idx: np.ndarray, y=y, p=p) -> dict:
                    from sklearn.metrics import brier_score_loss
                    mask = np.isfinite(p[idx])
                    if mask.sum() < 10:
                        return {"ece15": None, "brier": None}
                    return {
                        "ece15": _ece(y[idx][mask], p[idx][mask], 15),
                        "brier": float(brier_score_loss(y[idx][mask], np.clip(p[idx][mask], 0, 1))),
                    }

                cis = bootstrap_dict(_m, len(sub), n_boot, rng)
                rows.append({"survey": survey, "class": cls, "kind": kind,
                             "n_objects": int(len(sub)), "n_pos": int(y.sum()),
                             "metrics": cis})
            # reliability diagram data (calibrated)
            p = sub[f"p_{key}"].to_numpy(float)
            edges = np.linspace(0, 1, 16)
            bins = []
            for i in range(15):
                m = (p >= edges[i]) & (p <= edges[i + 1] if i == 14 else p < edges[i + 1])
                if m.any():
                    bins.append({"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                                 "conf": float(p[m].mean()), "acc": float(y[m].mean()),
                                 "n": int(m.sum())})
            reliability[f"{survey}/{cls}"] = bins
    payload = {"table": 2, "title": "Calibration (ECE-15 / Brier, raw vs Dirichlet-calibrated)",
               "protocol": {"rows": "latest epoch per test object, spec-only", "n_boot": n_boot},
               "rows": rows, "reliability_calibrated": reliability}
    md_rows = [[r["survey"], r.get("class", "—"), r.get("kind", "—"),
                _fmt(r.get("metrics", {}).get("ece15")),
                _fmt(r.get("metrics", {}).get("brier"), 4),
                r.get("n_pos", "—"), r["n_objects"]]
               for r in rows]
    md = ("## Table 2 — Calibration (raw vs Dirichlet-calibrated; latest epoch per test object)\n\n"
          + _md_table(["survey", "class", "kind", "ECE(15)", "Brier", "n_pos", "n"], md_rows))
    return payload, md


# ── Table 3 — ranking ────────────────────────────────────────────────────────

def build_table_3(frame: pd.DataFrame, test_ids: set[str], *, v6e2: np.ndarray | None,
                  n_boot: int, rng: np.random.Generator) -> tuple[dict, str]:
    frame = frame.copy()
    if v6e2 is not None:
        frame["_p_v6e2"] = v6e2
    base = frame[
        frame["object_id"].astype(str).isin(test_ids)
        & frame["label_quality"].astype(str).isin(SPEC_QUALITIES)
        & frame["target_class"].notna()
    ]
    rows = []
    for goal, cls_idx in GOAL_TO_CLASS.items():
        s_col = {"ia": "s_ia", "nonia": "s_nonia", "other": "s_other"}[goal]
        for n in RANK_NDETS:
            sub = base[base["n_det"] == n]
            if len(sub) < 10:
                rows.append({"goal": goal, "n_det": n, "status": "too_few_rows",
                             "n_objects": int(len(sub))})
                continue
            y = (class_index(sub["target_class"]) == cls_idx)
            scorers = []
            if s_col in sub.columns:
                scorers.append(("fusion_v8", sub[s_col].to_numpy(float)))
            elif f"p_{CLASS_KEYS[cls_idx]}" in sub.columns:
                scorers.append(("fusion_v8", sub[f"p_{CLASS_KEYS[cls_idx]}"].to_numpy(float)))
            if goal == "ia" and "_p_v6e2" in sub.columns:
                scorers.append(("v6e2_p_follow_proxy", sub["_p_v6e2"].to_numpy(float)))
            for name, score in scorers:
                def _m(idx: np.ndarray, y=y, score=score) -> dict:
                    out = {}
                    for k in RANK_BUDGETS:
                        out[f"purity@{k}"] = _purity_at_k(y[idx], score[idx], k)
                        out[f"recall@{k}"] = _recall_at_k(y[idx], score[idx], k)
                    out["ndcg@50"] = _ndcg_at_k(y[idx], score[idx], 50)
                    return out

                cis = bootstrap_dict(_m, len(sub), n_boot, rng)
                rows.append({"goal": goal, "n_det": n, "method": name,
                             "n_objects": int(len(sub)), "n_target": int(y.sum()),
                             "metrics": cis})
    payload = {"table": 3, "title": "Ranking per science goal (object-level test slices)",
               "protocol": {"budgets": list(RANK_BUDGETS), "n_boot": n_boot,
                            "note": "v6e2 comparator exists for the Ia goal only — "
                                    "the other goals have no v6e2 counterpart"},
               "rows": rows}
    md_rows = []
    for r in rows:
        if r.get("status"):
            md_rows.append([r["goal"], r["n_det"], f"({r['status']})"] + ["—"] * 7 + [r["n_objects"]])
            continue
        m = r["metrics"]
        md_rows.append([r["goal"], r["n_det"], r["method"],
                        _fmt(m["purity@20"]), _fmt(m["purity@50"]), _fmt(m["purity@100"]),
                        _fmt(m["recall@20"]), _fmt(m["recall@50"]), _fmt(m["recall@100"]),
                        _fmt(m["ndcg@50"]), f"{r['n_target']}/{r['n_objects']}"])
    md = ("## Table 3 — Ranking per goal (purity@K / recall@B / nDCG@50)\n\n"
          + _md_table(["goal", "n_det", "method", "pur@20", "pur@50", "pur@100",
                       "rec@20", "rec@50", "rec@100", "nDCG@50", "target/n"], md_rows))
    return payload, md


# ── Table 4 — DP1 enrichment factors (subprocess) ────────────────────────────

def build_table_4(pred_dp1: Path, out_dir: Path) -> tuple[dict, str]:
    if not pred_dp1.exists():
        raise FileNotFoundError(f"DP1 predictions missing: {pred_dp1}")
    script = Path(__file__).resolve().parent / "build_enrichment_metrics.py"
    results: dict[str, Any] = {}
    for tag, pred_path in (("overall", pred_dp1), ("ndet_ge3", None)):
        ef_dir = out_dir / f"dp1_enrichment_{tag}"
        ef_dir.mkdir(parents=True, exist_ok=True)
        if tag == "ndet_ge3":
            df = pd.read_parquet(pred_dp1)
            latest = _latest_per_object(df)
            keep = set(latest.loc[latest["n_det"] >= 3, "object_id"].astype(str))
            sub = df[df["object_id"].astype(str).isin(keep)]
            if len(sub) == 0:
                results[tag] = {"status": "unavailable", "reason": "no objects with n_det>=3"}
                continue
            pred_path = ef_dir / "predictions_ndet3.parquet"
            sub.to_parquet(pred_path, index=False)
        proc = subprocess.run(
            [sys.executable, str(script), "--pred", str(pred_path), "--out-dir", str(ef_dir)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            results[tag] = {"status": "error",
                            "reason": (proc.stderr or proc.stdout)[-2000:]}
            continue
        with open(ef_dir / "enrichment_metrics.json") as fh:
            results[tag] = json.load(fh)
    payload = {"table": 4, "title": "DP1 operational enrichment factors "
               "(via scripts/build_enrichment_metrics.py, unchanged)",
               "results": results}
    md_lines = ["## Table 4 — DP1 enrichment factors (EF@top-1%, bootstrap CI95)", ""]
    for tag, res in results.items():
        md_lines.append(f"### {tag}")
        if res.get("status"):
            md_lines.append(f"**{res['status'].upper()}** — {res.get('reason', '')[:500]}")
            md_lines.append("")
            continue
        hk = f"{res['headline_K']:.4f}"
        headers = ["class", "n"] + list(res["rankers"])
        md_rows = []
        for cname, blk in res["by_class"].items():
            cell = blk["by_K"][hk]
            row = [cname, blk["n"]]
            for rname in res["rankers"]:
                c = cell[rname]
                if c["ef_mean"] is None or (isinstance(c["ef_mean"], float) and np.isnan(c["ef_mean"])):
                    row.append("—")
                else:
                    row.append(f"{c['ef_mean']:.2f} [{c['ef_ci95_lo']:.2f}, {c['ef_ci95_hi']:.2f}]")
            md_rows.append(row)
        md_lines += [_md_table(headers, md_rows), ""]
    return payload, "\n".join(md_lines)


# ── Table 5 — trust quality per expert ───────────────────────────────────────

def trust_ndet_rows(frame: pd.DataFrame, test_ids: set[str]) -> list[dict]:
    """Per-expert calibrated trust AUC/ECE at the contract-relevant n_det
    slice (buckets 3 / 4 / 5 and pooled 3-5), locked test rows, spec-only.

    Rows where the expert fired (finite ``q__<san>``) are scored against the
    helpfulness-table target contract: ``is_topclass_correct``
    (``mapped_pred_class == target_class``, ``context_only`` rows excluded)
    for regular experts, ``is_sn`` (``target_class != 'other'``) for
    ``SN_FILTER_EXPERTS``.  Point estimates only (no bootstrap) — this is a
    diagnostic slice of the Table 5 aggregate, not a headline.
    """
    try:
        from debass_meta.models.expert_trust import SN_FILTER_EXPERTS
        from debass_meta.projectors import ALL_EXPERT_KEYS, sanitize_expert_key
    except Exception:
        SN_FILTER_EXPERTS = set()
        ALL_EXPERT_KEYS = []
        sanitize_expert_key = lambda k: str(k).replace("/", "__")

    base = frame[
        frame["object_id"].astype(str).isin(test_ids)
        & frame["label_quality"].astype(str).isin(SPEC_QUALITIES)
        & frame["target_class"].notna()
        & frame["n_det"].isin(TRUST_NDET_BUCKETS)
    ]
    if len(base) == 0:
        return []
    san_to_key = {sanitize_expert_key(k): k for k in ALL_EXPERT_KEYS}
    q_sans = sorted(c[len("q__"):] for c in base.columns if c.startswith("q__"))
    n_det = base["n_det"].astype(int).to_numpy()
    target_class = base["target_class"].astype(str)
    rows: list[dict] = []
    for san in q_sans:
        expert_key = san_to_key.get(san, san.replace("__", "/"))
        q = pd.to_numeric(base[f"q__{san}"], errors="coerce").to_numpy(float)
        is_sn_filter = expert_key in SN_FILTER_EXPERTS
        if is_sn_filter:
            target_kind = "is_sn"
            y = (target_class != "other").astype(int).to_numpy()
            valid = np.isfinite(q)
        else:
            target_kind = "is_topclass_correct"
            mp_col = f"mapped_pred_class__{san}"
            if mp_col not in base.columns:
                rows.append({"expert": expert_key, "target": target_kind,
                             "status": f"no {mp_col} column in frame"})
                continue
            mp = base[mp_col]
            y = (mp.astype(str) == target_class).astype(int).to_numpy()
            pt_col = f"prediction_type__{san}"
            not_ctx = (
                (base[pt_col].astype(str) != "context_only").to_numpy()
                if pt_col in base.columns else np.ones(len(base), dtype=bool)
            )
            valid = np.isfinite(q) & mp.notna().to_numpy() & not_ctx
        for bucket in [str(b) for b in TRUST_NDET_BUCKETS] + ["3-5"]:
            m = valid if bucket == "3-5" else valid & (n_det == int(bucket))
            yy, qq = y[m], q[m]
            rows.append({
                "expert": expert_key, "target": target_kind, "n_det": bucket,
                "n_rows": int(m.sum()), "n_pos": int(yy.sum()),
                "auc": _safe_auc(yy, qq), "ece15": _ece(yy, qq, 15),
            })
    return rows


def build_table_5(train_report: dict, v6e2_metrics_path: Path,
                  v6e2_trust_dir: Path, *,
                  frame: pd.DataFrame | None = None,
                  test_ids: set[str] | None = None) -> tuple[dict, str]:
    pooled = (train_report.get("stage_a") or {}).get("metrics") or {}
    ndet_rows: list[dict] = []
    ndet_reason: str | None = None
    if frame is not None and test_ids is not None:
        try:
            ndet_rows = trust_ndet_rows(frame, test_ids)
            if not ndet_rows:
                ndet_reason = "no spec-labeled locked-test rows at n_det 3-5"
        except Exception as exc:
            ndet_reason = f"n_det-sliced trust rows failed: {exc}"
    else:
        ndet_reason = "evaluation frame / test ids unavailable"
    if not pooled and not ndet_rows:
        raise FileNotFoundError("no stage_a metrics in fusion_v8_train.json")
    v6e2: dict[str, dict] = {}
    if v6e2_metrics_path.exists():
        with open(v6e2_metrics_path) as fh:
            v6e2 = json.load(fh)
    san_of = lambda k: str(k).replace("/", "__")
    v6e2_by_san = {san_of(k): v for k, v in v6e2.items()}
    # local per-head metadata fallback
    if v6e2_trust_dir.is_dir():
        for sub in v6e2_trust_dir.iterdir():
            if sub.is_dir() and (sub / "metadata.json").exists() and sub.name not in v6e2_by_san:
                try:
                    v6e2_by_san[sub.name] = json.load(open(sub / "metadata.json"))
                except Exception:
                    pass
    rows = []
    for key in sorted(pooled, key=str):
        m = pooled[key] if isinstance(pooled[key], dict) else {}
        san = san_of(key)
        ref = v6e2_by_san.get(san, {})
        ref_raw = (ref.get("raw") or {}).get("roc_auc") if isinstance(ref.get("raw"), dict) else None
        ref_cal = (ref.get("calibrated") or {}).get("roc_auc") if isinstance(ref.get("calibrated"), dict) else None
        rows.append({
            "expert": str(key),
            "pooled_raw_auc": m.get("raw_auc"), "pooled_cal_auc": m.get("cal_auc"),
            "pooled_brier": m.get("brier"), "pooled_ece": m.get("ece"),
            "n_test": m.get("n_test"), "calibrator_kind": m.get("calibrator_kind"),
            "fallback_used": m.get("fallback_used"),
            "v6e2_raw_auc": ref_raw, "v6e2_cal_auc": ref_cal,
            "v6e2_n_test": ref.get("n_test_rows"),
            "v6e2_available": bool(ref),
        })
    payload = {"table": 5, "title": "Trust quality per expert (pooled fusion_v8 vs v6e2 heads)",
               "note": "local v6e2 artifacts cover only 3 experts; SCC json covers 11",
               "rows": rows,
               "ndet_sliced": {
                   "protocol": "locked test rows, spec-only labels, calibrated q "
                               "per expert claim (finite q__<expert>), n_det "
                               "buckets 3/4/5 + pooled 3-5; point estimates",
                   "reason_unavailable": ndet_reason,
                   "rows": ndet_rows,
               }}
    md_rows = [[r["expert"], _fmt(r["pooled_raw_auc"]), _fmt(r["pooled_cal_auc"]),
                _fmt(r["pooled_brier"], 4), _fmt(r["pooled_ece"], 4),
                r["n_test"] if r["n_test"] is not None else "—",
                r["calibrator_kind"] or "—",
                "yes" if r["fallback_used"] else "no",
                _fmt(r["v6e2_raw_auc"]), _fmt(r["v6e2_cal_auc"]),
                r["v6e2_n_test"] if r["v6e2_n_test"] is not None else "—"]
               for r in rows]
    md = ("## Table 5 — Trust quality per expert (pooled fusion_v8 vs v6e2)\n\n"
          + _md_table(["expert", "v8 raw AUC", "v8 cal AUC", "v8 Brier", "v8 ECE",
                       "n_test", "calibrator", "fallback", "v6e2 raw AUC",
                       "v6e2 cal AUC", "v6e2 n_test"], md_rows))
    md += ("\n\n### Table 5b — Trust at 3-5 detections "
           "(calibrated q, locked test, spec-only)\n\n")
    if ndet_rows:
        md += _md_table(
            ["expert", "target", "n_det", "AUC", "ECE(15)", "pos/n"],
            [[r["expert"], r.get("target", "—"),
              r.get("n_det", "—") if not r.get("status") else f"({r['status']})",
              _fmt(r.get("auc")), _fmt(r.get("ece15")),
              f"{r.get('n_pos', '—')}/{r.get('n_rows', '—')}"]
             for r in ndet_rows])
    else:
        md += f"**UNAVAILABLE** — {ndet_reason}"
    return payload, md


# ── Table 6 — conformal coverage ─────────────────────────────────────────────

def build_table_6(frame: pd.DataFrame, test_ids: set[str], alpha: float = 0.1) -> tuple[dict, str]:
    need = {"set_snia", "set_nonia", "set_other", "set_size"}
    if not need <= set(frame.columns):
        raise FileNotFoundError("predictions lack conformal set columns")
    base = frame[
        frame["object_id"].astype(str).isin(test_ids)
        & frame["label_quality"].astype(str).isin(INFO_QUALITIES)
        & frame["target_class"].notna()
    ].copy()
    # object-level within each bucket: latest n_det row per (object, bucket)
    base["_bucket"] = ndet_bucket(base["n_det"].to_numpy())
    base = (base.sort_values(["object_id", "n_det"])
            .groupby(["object_id", "_bucket"], as_index=False).tail(1))
    if len(base) < 10:
        raise FileNotFoundError("too few labeled test rows for coverage")
    y = class_index(base["target_class"])
    set_cols = ["set_snia", "set_nonia", "set_other"]
    covered = np.zeros(len(base), dtype=bool)
    for ci, col in enumerate(set_cols):
        covered |= (y == ci) & base[col].fillna(False).astype(bool).to_numpy()
    base["_covered"] = covered
    rows = []
    grouped = base.groupby([base["target_class"].astype(str), base["_survey"], base["_bucket"]])
    for (cls, survey, bucket), g in grouped:
        rows.append({"class": cls, "survey": survey, "n_det_bucket": bucket,
                     "n": int(len(g)), "coverage": float(g["_covered"].mean()),
                     "mean_set_size": float(g["set_size"].mean())})
    overall = {"n": int(len(base)), "coverage": float(base["_covered"].mean()),
               "mean_set_size": float(base["set_size"].mean()),
               "nominal": 1 - alpha}
    payload = {"table": 6, "title": "Conformal coverage vs nominal 0.9 (Mondrian APS)",
               "protocol": {"qualities": list(INFO_QUALITIES),
                            "rows": "latest row per (object, n_det bucket), locked test"},
               "overall": overall, "rows": rows}
    md_rows = [[r["class"], r["survey"], r["n_det_bucket"], f"{r['coverage']:.3f}",
                f"{r['mean_set_size']:.2f}", r["n"]] for r in rows]
    md = (f"## Table 6 — Conformal coverage (nominal {1 - alpha:.2f})\n\n"
          f"Overall: coverage={overall['coverage']:.3f}, "
          f"mean set size={overall['mean_set_size']:.2f}, n={overall['n']}\n\n"
          + _md_table(["class", "survey", "n_det bucket", "coverage", "mean |set|", "n"], md_rows))
    return payload, md


# ── Table 7 — component gates + headline/guards ──────────────────────────────

def _paired_auc_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                      n_boot: int, rng: np.random.Generator) -> dict:
    """Paired object-bootstrap delta of OvR AUC: method A − method B."""
    def _d(idx: np.ndarray) -> float | None:
        a = _safe_auc(y[idx], p_a[idx])
        b = _safe_auc(y[idx], p_b[idx])
        return None if a is None or b is None else a - b

    point = _d(np.arange(len(y)))
    if point is None:
        return {"delta": None, "lo": None, "hi": None}
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        d = _d(idx)
        if d is not None:
            ds.append(d)
    ok = len(ds) >= max(20, n_boot // 4)
    return {"delta": point,
            "lo": float(np.percentile(ds, 2.5)) if ok else None,
            "hi": float(np.percentile(ds, 97.5)) if ok else None}


def build_headline_guards(
    frame: pd.DataFrame,
    test_ids: set[str],
    *,
    v6e2: np.ndarray | None,
    train_report: dict,
    table4_payload: dict | None,
    v6e2_dp1_enrichment: Path,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Pre-registered headline + hard guards (spec correction #4b), test-only."""
    out: dict[str, Any] = {"preregistered": {
        "headline": "object-level spec-only macro OvR AUC @ n_det=5, locked test, "
                    "fusion_v8 vs re-scored v6e2",
        "guards": ["LSST-spec slice non-regression",
                   "DP1 EclBin+RRLyrae EF non-regression",
                   "LightGBM bagging-seed spread on headline < 0.02 (5 seeds)"],
    }}
    frame = frame.copy()
    if v6e2 is not None:
        frame["_p_v6e2"] = v6e2
    base = frame[
        frame["object_id"].astype(str).isin(test_ids)
        & frame["label_quality"].astype(str).isin(SPEC_QUALITIES)
        & frame["target_class"].notna()
    ]
    slice5 = base[base["n_det"] == 5]

    # headline ---------------------------------------------------------------
    headline: dict[str, Any] = {"n_objects": int(len(slice5))}
    if len(slice5) >= 10 and {"p_snia", "p_nonia", "p_other"} <= set(slice5.columns):
        y = class_index(slice5["target_class"])
        P = slice5[["p_snia", "p_nonia", "p_other"]].to_numpy(float)
        cis = bootstrap_dict(lambda idx: matrix_metrics(y[idx], P[idx]), len(slice5), n_boot, rng)
        headline["fusion_v8_macro_auc"] = cis["macro_auc"]
        headline["fusion_v8_auc_snia"] = cis["auc_snia"]
        if "_p_v6e2" in slice5.columns:
            d = _paired_auc_delta((y == 0).astype(int), P[:, 0],
                                  slice5["_p_v6e2"].to_numpy(float), n_boot, rng)
            sig = d["lo"] is not None and d["lo"] > 0
            headline["vs_v6e2_snia_auc_delta"] = d
            headline["vs_v6e2_significant_win"] = bool(sig)
            headline["claim"] = (
                "fusion_v8 beats re-scored v6e2 on snia OvR AUC @ n_det=5 (CI95 excludes 0)"
                if sig else
                "NO significant win claimed — delta CI95 includes 0 or is negative "
                "(local run gates correctness+honesty; SCC is the performance gate)"
            )
            headline["note"] = ("macro OvR AUC has no v6e2 counterpart (binary head); "
                                "the snia OvR axis is the comparable one")
        else:
            headline["claim"] = "v6e2 comparator unavailable — headline reported without comparison"
    else:
        headline["status"] = "unavailable (slice too small or fusion predictions missing)"
    out["headline"] = headline

    guards: list[dict] = []

    # guard 1: LSST-spec slice non-regression ---------------------------------
    lsst5 = slice5[slice5["_survey"] == "lsst"]
    g: dict[str, Any] = {"guard": "lsst_spec_non_regression", "n_objects": int(len(lsst5))}
    if len(lsst5) >= 10 and "_p_v6e2" in lsst5.columns and "p_snia" in lsst5.columns:
        y = (class_index(lsst5["target_class"]) == 0).astype(int)
        d = _paired_auc_delta(y, lsst5["p_snia"].to_numpy(float),
                              lsst5["_p_v6e2"].to_numpy(float), n_boot, rng)
        g["snia_auc_delta_vs_v6e2"] = d
        if d["delta"] is None:
            g["status"] = "N/A (AUC not computable)"
        else:
            g["pass"] = bool(d["delta"] >= 0 or (d["hi"] is not None and d["hi"] >= 0))
            g["rule"] = "pass iff delta >= 0 or CI95 upper >= 0 (no significant regression)"
    else:
        g["status"] = "N/A (no LSST spec test slice locally, or v6e2 unavailable)"
    guards.append(g)

    # guard 2: DP1 EclBin+RRLyrae EF non-regression ---------------------------
    g = {"guard": "dp1_eclbin_rrlyrae_ef_non_regression"}
    try:
        res = (table4_payload or {}).get("results", {}).get("overall", {})
        hk = f"{res['headline_K']:.4f}"
        cell = res["by_class"]["EclBin+RRLyrae"]["by_K"][hk]["p_follow_proxy"]
        fusion_ef = cell["ef_mean"]
        ref_ef, ref_src = DP1_ECLBIN_REFERENCE_EF, "documented v6e2 headline (17.88)"
        if v6e2_dp1_enrichment.exists():
            with open(v6e2_dp1_enrichment) as fh:
                ref = json.load(fh)
            rk = f"{ref['headline_K']:.4f}"
            ref_ef = ref["by_class"]["EclBin+RRLyrae"]["by_K"][rk]["p_follow_proxy"]["ef_mean"]
            ref_src = str(v6e2_dp1_enrichment)
        g.update({"fusion_v8_ef": fusion_ef, "v6e2_ef": ref_ef, "v6e2_source": ref_src,
                  "pass": bool(fusion_ef is not None and fusion_ef <= ref_ef),
                  "rule": "pass iff fusion_v8 EF@top-1% <= v6e2 EF (lower = better suppression)"})
    except Exception as exc:
        g["status"] = f"N/A (DP1 EF unavailable: {exc})"
    guards.append(g)

    # guard 3: bagging-seed spread on the headline ----------------------------
    g = {"guard": "seed_spread_lt_0.02"}
    sv = train_report.get("seed_variants")
    try:
        if not sv:
            raise RuntimeError("no seed variants recorded in fusion_v8_train.json")
        from debass_meta.models.multiclass_followup import MulticlassFollowupArtifact

        if len(slice5) < 10:
            raise RuntimeError("headline slice too small")
        y5 = class_index(slice5["target_class"])
        aucs: dict[str, float | None] = {}
        for label, d in [("main", sv["main_dir"])] + [
                (f"seed{s}", p) for s, p in zip(sv["seeds"][1:], sv["variant_dirs"])]:
            art = MulticlassFollowupArtifact.load(str(d))
            P = np.asarray(art.predict_proba(slice5), dtype=float)
            aucs[label] = matrix_metrics(y5, P)["macro_auc"]
        vals = [v for v in aucs.values() if v is not None]
        if len(vals) < 2:
            raise RuntimeError("macro AUC not computable for seed variants")
        spread = float(max(vals) - min(vals))
        g.update({"per_seed_macro_auc": aucs, "spread": spread,
                  "pass": bool(spread < 0.02), "rule": "pass iff max-min macro AUC < 0.02"})
    except Exception as exc:
        g["status"] = f"N/A ({exc})"
    guards.append(g)

    out["guards"] = guards
    return out


def build_table_7(train_report: dict, headline_guards: dict) -> tuple[dict, str]:
    gates = train_report.get("gates")
    if gates is None:
        raise FileNotFoundError("no gate ledger in fusion_v8_train.json")
    payload = {"table": 7, "title": "Component-gate ledger (decided on cal) + "
               "pre-registered headline & guards",
               "gates": gates, "aug_frac_final": train_report.get("aug_frac_final"),
               "headline_guards": headline_guards}
    md_rows = []
    for gate in gates:
        md_rows.append([
            gate.get("component", "?"), gate.get("decision", "?"),
            _fmt(gate.get("delta_macro_auc")),
            f"[{_fmt(gate.get('delta_macro_auc_ci95', [None, None])[0])}, "
            f"{_fmt(gate.get('delta_macro_auc_ci95', [None, None])[1])}]"
            if gate.get("delta_macro_auc_ci95") else "—",
            _fmt(gate.get("delta_purity50")),
            (gate.get("reason") or "")[:80],
        ])
    md_lines = [
        "## Table 7 — Component gates (cal-only decisions) + headline & guards", "",
        _md_table(["component", "decision", "Δ macro AUC (cal)", "Δ CI95",
                   "Δ purity@50 (cal)", "note"], md_rows),
        "", "### Pre-registered headline", "",
        "```json", json.dumps(_jsonify(headline_guards.get("headline", {})), indent=2), "```",
        "", "### Guards", "",
    ]
    for g in headline_guards.get("guards", []):
        status = g.get("status") or ("PASS" if g.get("pass") else "FAIL")
        md_lines.append(f"- **{g['guard']}**: {status}")
    return payload, "\n".join(md_lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="fusion_v8 evaluation — Tables 1-7 + guards")
    parser.add_argument("--pred", default="data/scores/predictions_fusion_v8.parquet")
    parser.add_argument("--pred-dp1", default="data/scores/predictions_fusion_v8_dp1.parquet")
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_fusion_v8_trust.parquet")
    parser.add_argument("--split", default="data/gold/split_fusion_v8.json")
    parser.add_argument("--train-metrics", default="reports/metrics/fusion_v8_train.json")
    parser.add_argument("--v6e2-trust-dir", default="models/trust")
    parser.add_argument("--v6e2-followup-dir", default="models/followup")
    parser.add_argument("--v6e2-trust-metrics", default="reports/metrics/expert_trust_metrics_safe_v6e2.json")
    parser.add_argument("--v6e2-dp1-enrichment", default="reports/v6_dp1_50k/enrichment_metrics.json")
    parser.add_argument("--out-dir", default="reports/fusion_v8")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dp1", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="n_boot=100 quick pass")
    args = parser.parse_args()

    n_boot = 100 if args.smoke else args.n_boot
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_report: dict = {}
    if Path(args.train_metrics).exists():
        with open(args.train_metrics) as fh:
            train_report = json.load(fh)
    else:
        print(f"[warn] {args.train_metrics} missing — Tables 5/7 + seed guard degrade")

    try:
        train_ids, cal_ids, test_ids, split_raw = load_split_manifest(Path(args.split))
    except Exception as exc:
        raise SystemExit(f"Cannot evaluate without split manifest: {exc}")
    print(f"Split: train={len(train_ids):,} cal={len(cal_ids):,} test={len(test_ids):,}")

    # ── assemble the evaluation frame (snapshot + predictions merged) ────────
    frame: pd.DataFrame | None = None
    pred: pd.DataFrame | None = None
    if Path(args.pred).exists():
        pred = pd.read_parquet(args.pred)
    if Path(args.snapshots).exists():
        frame = pd.read_parquet(args.snapshots)
        if pred is not None:
            pred_cols = [c for c in pred.columns
                         if c not in frame.columns and c not in ("object_id", "n_det")]
            frame = frame.merge(pred[["object_id", "n_det"] + pred_cols],
                                on=["object_id", "n_det"], how="left")
    elif pred is not None:
        print("[warn] snapshot parquet missing — baselines limited to prediction columns")
        frame = pred.copy()
    if frame is None:
        raise SystemExit(f"Neither {args.snapshots} nor {args.pred} exists — nothing to evaluate")
    frame["_survey"] = frame["survey"].astype(str).str.lower()
    print(f"Evaluation frame: {len(frame):,} rows ({frame['object_id'].nunique():,} objects)")

    # v6e2 re-scored comparator (graceful — local artifacts cover 3 experts)
    v6e2_scores, v6e2_reason = v6e2_rescore(
        frame, Path(args.v6e2_trust_dir), Path(args.v6e2_followup_dir))
    if v6e2_reason:
        print(f"[warn] v6e2 rescoring unavailable: {v6e2_reason}")

    # ── Table 1 ───────────────────────────────────────────────────────────
    try:
        payload1, md1 = build_table_1(
            frame, test_ids, cal_ids, v6e2=v6e2_scores, v6e2_reason=v6e2_reason,
            n_boot=n_boot, rng=rng, split_raw=split_raw)
        _write_table(out_dir, 1, payload1, md1)
    except Exception as exc:
        _unavailable(out_dir, 1, "Table 1 — Classification", str(exc))

    # ── Table 2 ───────────────────────────────────────────────────────────
    try:
        payload2, md2 = build_table_2(frame, test_ids, n_boot, rng)
        _write_table(out_dir, 2, payload2, md2)
    except Exception as exc:
        _unavailable(out_dir, 2, "Table 2 — Calibration", str(exc))

    # ── Table 3 ───────────────────────────────────────────────────────────
    try:
        payload3, md3 = build_table_3(frame, test_ids, v6e2=v6e2_scores,
                                      n_boot=n_boot, rng=rng)
        _write_table(out_dir, 3, payload3, md3)
    except Exception as exc:
        _unavailable(out_dir, 3, "Table 3 — Ranking", str(exc))

    # ── Table 4 (DP1) ─────────────────────────────────────────────────────
    payload4: dict | None = None
    if args.skip_dp1:
        _unavailable(out_dir, 4, "Table 4 — DP1 enrichment", "--skip-dp1")
    else:
        try:
            payload4, md4 = build_table_4(Path(args.pred_dp1), out_dir)
            _write_table(out_dir, 4, payload4, md4)
        except Exception as exc:
            _unavailable(out_dir, 4, "Table 4 — DP1 enrichment", str(exc))

    # ── Table 5 ───────────────────────────────────────────────────────────
    try:
        payload5, md5 = build_table_5(train_report, Path(args.v6e2_trust_metrics),
                                      Path(args.v6e2_trust_dir),
                                      frame=frame, test_ids=test_ids)
        _write_table(out_dir, 5, payload5, md5)
    except Exception as exc:
        _unavailable(out_dir, 5, "Table 5 — Trust quality", str(exc))

    # ── Table 6 ───────────────────────────────────────────────────────────
    try:
        payload6, md6 = build_table_6(frame, test_ids)
        _write_table(out_dir, 6, payload6, md6)
    except Exception as exc:
        _unavailable(out_dir, 6, "Table 6 — Conformal coverage", str(exc))

    # ── headline + guards, then Table 7 ───────────────────────────────────
    headline_guards = build_headline_guards(
        frame, test_ids, v6e2=v6e2_scores, train_report=train_report,
        table4_payload=payload4, v6e2_dp1_enrichment=Path(args.v6e2_dp1_enrichment),
        n_boot=n_boot, rng=rng)
    (out_dir / "headline_guards.json").write_text(
        json.dumps(_jsonify(headline_guards), indent=2))
    hg_md = ["# fusion_v8 — pre-registered headline & guards", "",
             "```json", json.dumps(_jsonify(headline_guards), indent=2), "```"]
    (out_dir / "headline_guards.md").write_text("\n".join(hg_md) + "\n")
    print("  wrote headline_guards.md / headline_guards.json")

    try:
        payload7, md7 = build_table_7(train_report, headline_guards)
        _write_table(out_dir, 7, payload7, md7)
    except Exception as exc:
        _unavailable(out_dir, 7, "Table 7 — Component gates", str(exc))

    print(f"Done — reports under {out_dir}")


if __name__ == "__main__":
    main()
