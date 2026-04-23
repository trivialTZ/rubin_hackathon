"""Phase 0.3/0.4 — latency-aware metrics, bootstrap CIs, ensemble baselines.

Given an object-epoch snapshot (with q__{expert} trust + proj__{expert}__p_snia
columns) and a truth column, compute for each n_det ∈ n_dets:

  • per-expert AUC of P(Ia) on available rows
  • three ensemble baselines:
      - best_single_expert : choose expert with best overall test AUC, use its p_snia
      - mean_ensemble       : uniform average of available experts' p_snia
      - trust_weighted      : Σ p_snia · q / Σ q  (over available experts)
  • Brier, ECE, purity-completeness curve
  • bootstrap percentile CI on each AUC (default 200 resamples, 95%)

Outputs a long-format DataFrame suitable for parquet + seaborn plots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from debass_meta.projectors import ALL_EXPERT_KEYS, sanitize_expert_key


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = np.isfinite(y_prob)
    if mask.sum() < 3 or len(np.unique(y_true[mask])) < 2:
        return None
    try:
        return float(roc_auc_score(y_true[mask], y_prob[mask]))
    except Exception:
        return None


def _safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = np.isfinite(y_prob)
    if mask.sum() < 3:
        return None
    try:
        return float(brier_score_loss(y_true[mask], y_prob[mask]))
    except Exception:
        return None


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, float | None]:
    """Percentile-CI bootstrap on AUC. Returns {'auc', 'lo', 'hi'} or Nones."""
    rng = rng or np.random.default_rng(42)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    n = len(y_true)
    if n < 10 or len(np.unique(y_true)) < 2:
        return {"auc": _safe_auc(y_true, y_prob), "lo": None, "hi": None, "n_boot_ok": 0}
    point = _safe_auc(y_true, y_prob)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sampled_true = y_true[idx]
        sampled_prob = y_prob[idx]
        auc = _safe_auc(sampled_true, sampled_prob)
        if auc is not None:
            aucs.append(auc)
    if len(aucs) < n_boot // 4:
        return {"auc": point, "lo": None, "hi": None, "n_boot_ok": len(aucs)}
    lo = float(np.percentile(aucs, 100 * (alpha / 2)))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return {"auc": point, "lo": lo, "hi": hi, "n_boot_ok": len(aucs)}


def purity_completeness_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    targets: Sequence[float] = (0.5, 0.7, 0.9, 0.95),
) -> dict[str, float | None]:
    """At each completeness target, the max purity reachable by thresholding p_prob,
    and at each purity target, the max completeness reachable.

    Purity = TP / (TP + FP); completeness = TP / (TP + FN).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_prob) & np.isin(y_true, [0, 1])
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    out: dict[str, float | None] = {}
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        for t in targets:
            out[f"purity@comp{t}"] = None
            out[f"completeness@pur{t}"] = None
        return out

    # Sort by descending p
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    # Cumulative TP and FP
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    total_pos = int((y_true == 1).sum())
    completeness = tp_cum / max(total_pos, 1)
    predicted = tp_cum + fp_cum
    purity = np.where(predicted > 0, tp_cum / np.maximum(predicted, 1), 0.0)

    for t in targets:
        # Max purity at completeness >= t
        reach = completeness >= t
        out[f"purity@comp{t}"] = float(np.max(purity[reach])) if reach.any() else None
        # Max completeness at purity >= t
        reach_p = purity >= t
        out[f"completeness@pur{t}"] = float(np.max(completeness[reach_p])) if reach_p.any() else None
    return out


# ── Ensemble baselines ───────────────────────────────────────────────────────

def _available_cols_for_expert(expert_key: str, df: pd.DataFrame) -> tuple[str, str, str] | None:
    san = sanitize_expert_key(expert_key)
    avail = f"avail__{san}"
    trust = f"q__{san}"
    psnia = f"proj__{san}__p_snia"
    if avail not in df.columns or psnia not in df.columns:
        return None
    return avail, trust, psnia


def ensemble_predictions(
    df: pd.DataFrame,
    expert_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with same index as df, adding columns:
        p_snia_mean_ensemble  : unweighted mean of available experts' p_snia
        p_snia_trust_weighted : Σ p_snia·q / Σ q, over available experts
        n_experts_available   : count per row
    """
    if expert_keys is None:
        expert_keys = ALL_EXPERT_KEYS
    n_rows = len(df)
    p_sum_u = np.zeros(n_rows, dtype=float)
    n_u = np.zeros(n_rows, dtype=float)
    p_sum_w = np.zeros(n_rows, dtype=float)
    q_sum_w = np.zeros(n_rows, dtype=float)
    for expert_key in expert_keys:
        cols = _available_cols_for_expert(expert_key, df)
        if cols is None:
            continue
        avail, trust, psnia = cols
        avail_arr = df[avail].fillna(0).astype(bool).to_numpy()
        p_arr = df[psnia].astype(float).to_numpy()
        q_arr = (
            df[trust].astype(float).to_numpy()
            if trust in df.columns
            else np.ones(n_rows, dtype=float)
        )
        # Filter finite
        p_valid = np.isfinite(p_arr)
        q_valid = np.isfinite(q_arr)
        usable = avail_arr & p_valid
        p_sum_u[usable] += p_arr[usable]
        n_u[usable] += 1
        usable_w = usable & q_valid
        p_sum_w[usable_w] += p_arr[usable_w] * q_arr[usable_w]
        q_sum_w[usable_w] += q_arr[usable_w]

    with np.errstate(invalid="ignore", divide="ignore"):
        p_mean = np.where(n_u > 0, p_sum_u / np.maximum(n_u, 1), np.nan)
        p_tw = np.where(q_sum_w > 0, p_sum_w / np.maximum(q_sum_w, 1e-12), np.nan)
    return pd.DataFrame({
        "p_snia_mean_ensemble": p_mean,
        "p_snia_trust_weighted": p_tw,
        "n_experts_available": n_u.astype(int),
    }, index=df.index)


def best_single_expert_oracle(
    df: pd.DataFrame,
    *,
    expert_aucs: dict[str, float],
    expert_keys: Sequence[str] | None = None,
) -> tuple[np.ndarray, str | None]:
    """Return per-row p_snia from the globally-best-AUC expert (oracle baseline)."""
    if expert_keys is None:
        expert_keys = ALL_EXPERT_KEYS
    ranked = sorted(
        (k for k in expert_keys if expert_aucs.get(k) is not None),
        key=lambda k: expert_aucs[k],
        reverse=True,
    )
    if not ranked:
        return np.full(len(df), np.nan), None
    for expert_key in ranked:
        cols = _available_cols_for_expert(expert_key, df)
        if cols is None:
            continue
        avail, _, psnia = cols
        avail_arr = df[avail].fillna(0).astype(bool).to_numpy()
        p_arr = df[psnia].astype(float).to_numpy()
        out = np.where(avail_arr & np.isfinite(p_arr), p_arr, np.nan)
        if np.isfinite(out).any():
            return out, expert_key
    return np.full(len(df), np.nan), None


# ── Latency-aware table ──────────────────────────────────────────────────────

@dataclass
class LatencyResult:
    per_ndet: pd.DataFrame            # one row per (n_det, method) with AUC/Brier/CI/PC
    per_expert_ndet: pd.DataFrame     # one row per (n_det, expert) with AUC/CI
    earliest_n_det: dict[str, int | None]   # {method: earliest n_det to reach AUC>=0.9}


def auc_vs_ndet(
    snapshot_df: pd.DataFrame,
    *,
    n_dets: Iterable[int] = (3, 4, 5, 7, 10, 15, 20),
    target_auc: float = 0.9,
    expert_keys: Sequence[str] | None = None,
    label_qualities: Sequence[str] | None = ("spectroscopic", "tns_untyped"),
    bootstrap_n: int = 200,
) -> LatencyResult:
    """Compute per-n_det AUC/Brier/purity-completeness for each ensemble + per expert.

    Uses `target_class == "snia"` as binary truth.  Filters to label qualities if given.
    """
    if expert_keys is None:
        expert_keys = ALL_EXPERT_KEYS

    df = snapshot_df.copy()
    if label_qualities is not None and "label_quality" in df.columns:
        df = df[df["label_quality"].isin(list(label_qualities))]
    if "target_class" not in df.columns:
        raise KeyError("snapshot_df must contain target_class column")
    df = df[df["target_class"].notna()]

    df["is_snia"] = (df["target_class"].astype(str) == "snia").astype(int)

    # First pass: global per-expert AUC (for oracle baseline)
    global_expert_aucs: dict[str, float] = {}
    for expert_key in expert_keys:
        cols = _available_cols_for_expert(expert_key, df)
        if cols is None:
            continue
        avail, _, psnia = cols
        sub = df[df[avail].fillna(0).astype(bool)]
        auc = _safe_auc(sub["is_snia"].to_numpy(), sub[psnia].astype(float).to_numpy())
        if auc is not None:
            global_expert_aucs[expert_key] = auc

    ens_cols = ensemble_predictions(df, expert_keys=expert_keys)
    df["p_snia_mean_ensemble"] = ens_cols["p_snia_mean_ensemble"]
    df["p_snia_trust_weighted"] = ens_cols["p_snia_trust_weighted"]
    best_arr, best_key = best_single_expert_oracle(df, expert_aucs=global_expert_aucs, expert_keys=expert_keys)
    df["p_snia_best_single"] = best_arr

    rng = np.random.default_rng(42)

    per_ndet_rows: list[dict] = []
    per_expert_rows: list[dict] = []
    earliest: dict[str, int | None] = {m: None for m in ("mean_ensemble", "trust_weighted", "best_single")}

    for n in sorted(set(int(x) for x in n_dets)):
        sub = df[df["n_det"] == n]
        y = sub["is_snia"].to_numpy()
        if len(sub) == 0:
            continue
        for method, col in [
            ("mean_ensemble", "p_snia_mean_ensemble"),
            ("trust_weighted", "p_snia_trust_weighted"),
            ("best_single", "p_snia_best_single"),
        ]:
            p = sub[col].to_numpy(dtype=float)
            ci = bootstrap_auc_ci(y, p, n_boot=bootstrap_n, rng=rng)
            pc = purity_completeness_curve(y, p)
            row = {
                "n_det": n,
                "method": method,
                "n_rows": int(len(sub)),
                "n_pos": int((y == 1).sum()),
                "auc": ci["auc"],
                "auc_lo": ci["lo"],
                "auc_hi": ci["hi"],
                "brier": _safe_brier(y, p),
                "ece": _expected_calibration_error(y, p),
                **pc,
            }
            per_ndet_rows.append(row)
            if ci["auc"] is not None and ci["auc"] >= target_auc and earliest[method] is None:
                earliest[method] = n

        for expert_key in expert_keys:
            cols = _available_cols_for_expert(expert_key, sub)
            if cols is None:
                continue
            avail, _, psnia = cols
            available_mask = sub[avail].fillna(0).astype(bool)
            ssub = sub[available_mask]
            if len(ssub) < 3:
                continue
            yy = ssub["is_snia"].to_numpy()
            pp = ssub[psnia].astype(float).to_numpy()
            ci = bootstrap_auc_ci(yy, pp, n_boot=bootstrap_n, rng=rng)
            per_expert_rows.append({
                "n_det": n,
                "expert": expert_key,
                "n_rows": int(len(ssub)),
                "n_pos": int((yy == 1).sum()),
                "auc": ci["auc"],
                "auc_lo": ci["lo"],
                "auc_hi": ci["hi"],
                "brier": _safe_brier(yy, pp),
            })

    per_ndet = pd.DataFrame(per_ndet_rows)
    per_expert = pd.DataFrame(per_expert_rows)
    per_ndet.attrs["best_single_expert_key"] = best_key
    per_ndet.attrs["global_expert_aucs"] = global_expert_aucs
    return LatencyResult(per_ndet=per_ndet, per_expert_ndet=per_expert, earliest_n_det=earliest)


def plot_auc_vs_ndet(latency: LatencyResult, out_path: str) -> None:
    """Plot AUC-vs-n_det for the three ensemble baselines with CI bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    per = latency.per_ndet
    if per.empty:
        ax.text(0.5, 0.5, "no data", ha="center")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return
    for method, color, marker in [
        ("trust_weighted", "#c0392b", "o"),
        ("mean_ensemble", "#2980b9", "s"),
        ("best_single", "#27ae60", "^"),
    ]:
        sub = per[per["method"] == method].sort_values("n_det")
        if sub.empty:
            continue
        ax.plot(sub["n_det"], sub["auc"], color=color, marker=marker, label=method, lw=1.5)
        has_ci = sub["auc_lo"].notna() & sub["auc_hi"].notna()
        if has_ci.any():
            ax.fill_between(
                sub["n_det"][has_ci],
                sub["auc_lo"][has_ci],
                sub["auc_hi"][has_ci],
                color=color,
                alpha=0.15,
            )
    ax.axhline(0.9, color="gray", ls="--", lw=0.7, label="AUC = 0.9")
    ax.set_xlabel("n_det")
    ax.set_ylabel("AUC (is_SN Ia)")
    ax.set_title("DEBASS — AUC vs n_det (95% CI bootstrap)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
