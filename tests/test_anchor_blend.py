"""fusion_v11 anchored blend — anchor, Ia-capability mask, α fit + apply."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from debass_meta.models import anchor_blend
from debass_meta.models.anchor_blend import (
    ANCHOR_EXCLUDED,
    EPS,
    IA_CAPABLE,
    BlendSpec,
    apply,
    compute_anchor,
    fit_alpha,
)
from debass_meta.projectors.base import (
    ALL_EXPERT_KEYS,
    project_expert_events,
    sanitize_expert_key,
)


# ── Ia-capability mask self-verification ─────────────────────────────────────

def _ratio(proj: dict) -> float | None:
    """p_snia / (p_snia + p_nonIa_snlike); None when there is no SN mass."""
    ps = proj.get("p_snia")
    pn = proj.get("p_nonIa_snlike")
    if ps is None or pn is None:
        return None
    denom = ps + pn
    if denom <= 1e-9:
        return None
    return ps / denom


# Synthetic-event generators for every NON-Ia-capable ternary projector.  Each
# yields two event sets that would move an Ia-capable expert's Ia|SN ratio; a
# non-capable projector must hold the ratio constant.
def _non_member_events() -> dict[str, list[list[dict]]]:
    def stamp(cls_sn: str, other: str):
        return [
            [{"class_name": cls_sn, "canonical_projection": v},
             {"class_name": other, "canonical_projection": 1.0 - v}]
            for v in (0.2, 0.85)
        ]

    return {
        "fink/slsn": [[{"field": "slsn_score", "canonical_projection": v}]
                      for v in (0.2, 0.9)],
        "fink_lsst/snn": [[{"canonical_projection": v}] for v in (0.3, 0.8)],
        "alerce/stamp_classifier": stamp("SN", "AGN"),
        "alerce/stamp_classifier_2025_beta": stamp("SN", "AGN"),
        "alerce/stamp_classifier_rubin_beta": stamp("SN", "AGN"),
        "alerce/lc_classifier_BHRF_forced_phot_top": stamp("Transient", "Periodic"),
        "ampel/snguess": [[{"canonical_projection": v}] for v in (0.25, 0.75)],
        "pittgoogle/upsilon_lsst": [[{"class_name": "rr_lyrae", "canonical_projection": v}]
                                    for v in (0.3, 0.7)],
    }


def test_generators_cover_all_non_members():
    """The self-verifying test must exercise EVERY non-member ternary projector
    (documented forfeit `fink_lsst/cats` and context-only sherlock/babamul aside)."""
    ternary = set(ALL_EXPERT_KEYS) - set(ANCHOR_EXCLUDED)
    non_members = ternary - set(IA_CAPABLE) - {"fink_lsst/cats"}
    assert set(_non_member_events()) == non_members


@pytest.mark.parametrize("expert_key", list(_non_member_events()))
def test_non_member_constant_ia_ratio(expert_key):
    """Every non-Ia-capable projector holds a CONSTANT Ia|SN ratio (drift guard)."""
    ratios = []
    for events in _non_member_events()[expert_key]:
        proj = project_expert_events(expert_key, events)
        r = _ratio(proj)
        if r is not None:
            ratios.append(r)
    assert ratios, f"{expert_key}: no SN-mass rows produced"
    assert max(ratios) - min(ratios) < 1e-9, f"{expert_key} ratio drifted: {ratios}"


def test_fink_snn_is_ia_capable_ratio_moves():
    """Sanity foil: fink/snn (an Ia-capable member) DOES move the Ia|SN ratio."""
    ratios = []
    for snia_vs_nonia in (0.2, 0.8):
        events = [
            {"field": "snn_snia_vs_nonia", "canonical_projection": snia_vs_nonia},
            {"field": "snn_sn_vs_all", "canonical_projection": 0.9},
        ]
        r = _ratio(project_expert_events("fink/snn", events))
        assert r is not None
        ratios.append(r)
    assert max(ratios) - min(ratios) > 0.4


def test_cats_forfeited_not_capable():
    """fink_lsst/cats is row-conditional (documented v11 forfeit): OUT of IA_CAPABLE."""
    assert "fink_lsst/cats" not in IA_CAPABLE


def test_ia_capable_subset_of_registry():
    """IA_CAPABLE ⊆ ALL_EXPERT_KEYS (seq_v11 registered by sibling P5)."""
    anchor_blend.assert_ia_capable_subset()  # tolerates pending seq_v11
    strict_missing = set(IA_CAPABLE) - {"seq_v11"} - set(ALL_EXPERT_KEYS)
    assert not strict_missing, strict_missing
    # nothing but seq_v11 may silently be missing from the live registry
    assert (set(IA_CAPABLE) - set(ALL_EXPERT_KEYS)) <= {"seq_v11"}


def test_sherlock_babamul_excluded():
    assert {"lasair/sherlock", "babamul"} <= set(ANCHOR_EXCLUDED)
    assert not (set(ANCHOR_EXCLUDED) & set(IA_CAPABLE))


# ── anchor construction ──────────────────────────────────────────────────────

def _san(key: str) -> str:
    return sanitize_expert_key(key)


def _row_with_experts(experts: dict[str, tuple[float, float, float]], *,
                      survey: str = "ztf", q: dict[str, float] | None = None) -> dict:
    """Build one snapshot row with proj__/avail__/q__ columns for `experts`."""
    row: dict = {"survey": survey}
    q = q or {}
    for key, (ps, pn, po) in experts.items():
        s = _san(key)
        row[f"proj__{s}__p_snia"] = ps
        row[f"proj__{s}__p_nonIa_snlike"] = pn
        row[f"proj__{s}__p_other"] = po
        row[f"avail__{s}"] = True
        row[f"q__{s}"] = q.get(key, 1.0)
    return row


def test_anchor_two_axis_pool():
    """SN axis pools all fired experts; Ia axis pools Ia-capable experts only."""
    # fink/snn (Ia-capable): Ia|SN = 0.8 ; fink_lsst/snn (not capable): Ia|SN=0.5
    df = pd.DataFrame([_row_with_experts({
        "fink/snn": (0.72, 0.18, 0.10),          # p_sn=0.9, r=0.8
        "fink_lsst/snn": (0.30, 0.30, 0.40),      # p_sn=0.6, r=0.5 (ignored on Ia axis)
    })])
    out = compute_anchor(df)
    assert int(out["n_experts_fired"].iloc[0]) == 2
    p_snia = out["p_snia_anchor"].iloc[0]
    p_other = out["p_other_anchor"].iloc[0]
    # SN axis: equal q → P(SN) = mean(0.9, 0.6) = 0.75 → p_other = 0.25
    assert p_other == pytest.approx(0.25, abs=1e-6)
    # Ia axis uses only fink/snn → r=0.8 → p_snia = 0.75*0.8 = 0.6
    assert p_snia == pytest.approx(0.6, abs=1e-6)
    assert out[["p_snia_anchor", "p_nonia_anchor", "p_other_anchor"]].iloc[0].sum() \
        == pytest.approx(1.0, abs=1e-9)


def test_anchor_base_rate_when_no_ia_capable():
    """No Ia-capable expert fired → Ia axis falls back to per-survey base rate."""
    df = pd.DataFrame([_row_with_experts({
        "alerce/stamp_classifier": (0.0, 0.8, 0.2),  # p_sn=0.8, non-capable
    })])
    out = compute_anchor(df, base_rate_by_survey={"ztf": 0.4})
    # P(SN)=0.8, r=base_rate 0.4 → p_snia = 0.32
    assert out["p_snia_anchor"].iloc[0] == pytest.approx(0.32, abs=1e-6)


def test_anchor_zero_experts_is_nan():
    """No fired anchor-eligible expert → NaN anchor, n_experts_fired=0."""
    df = pd.DataFrame([{"survey": "ztf", "p_snia": 0.3, "p_nonia": 0.3, "p_other": 0.4}])
    out = compute_anchor(df)
    assert int(out["n_experts_fired"].iloc[0]) == 0
    assert out["p_snia_anchor"].isna().iloc[0]


def test_anchor_excludes_sherlock_babamul():
    """sherlock/babamul never enter the anchor even when they carry proj columns."""
    df = pd.DataFrame([_row_with_experts({
        "lasair/sherlock": (0.5, 0.3, 0.2),
        "babamul": (0.1, 0.1, 0.8),
    })])
    out = compute_anchor(df)
    assert int(out["n_experts_fired"].iloc[0]) == 0
    assert out["p_snia_anchor"].isna().iloc[0]


def test_eps_clip_gives_finite_logloss_on_stamp_only_cell():
    """Stamp-only anchor (p_snia=0 pre-clip) → ε-clip keeps log-loss finite even
    when the true class is snia (the v11 G3 stamp-cell requirement)."""
    df = pd.DataFrame([_row_with_experts({
        "alerce/stamp_classifier": (0.0, 0.9, 0.1),
    })])
    out = compute_anchor(df, base_rate_by_survey={"ztf": 0.05})
    p_snia = out["p_snia_anchor"].iloc[0]
    assert p_snia >= EPS
    assert np.isfinite(-np.log(p_snia))


# ── α fit + apply ────────────────────────────────────────────────────────────

def _cal_frame(n: int, *, anchor_good: bool, model_good: bool, seed: int = 0) -> pd.DataFrame:
    from debass_meta.features.lightcurve import FEATURE_NAMES

    rng = np.random.default_rng(seed)
    is_ia = rng.random(n) < 0.5
    tc = np.where(is_ia, "snia", np.where(rng.random(n) < 0.5, "nonIa_snlike", "other"))
    is_sn = np.isin(tc, ["snia", "nonIa_snlike"])

    def signal(good):
        base = np.where(is_ia, 0.85, 0.15)
        noise = rng.uniform(0, 1, n)
        return np.clip(base, 0, 1) if good else noise

    # two anchor experts so n_experts_fired == 2 (both Ia-capable so Ia axis fires)
    rows = []
    a_sig = signal(anchor_good)
    m_sig = signal(model_good)
    for i in range(n):
        r = _row_with_experts({
            "fink/snn": (a_sig[i] * 0.9, (1 - a_sig[i]) * 0.9, 0.1),
            "pittgoogle/supernnova_ztf": (a_sig[i] * 0.9, (1 - a_sig[i]) * 0.63, 0.27),
        })
        r["target_class"] = tc[i]
        r["label_source"] = "spectroscopic"
        # model probs
        ps = m_sig[i]
        r["p_snia"], r["p_nonia"], r["p_other"] = ps * 0.9, (1 - ps) * 0.7, (1 - ps) * 0.3 + 0.1 * 0.0
        # full lc coverage → cov bucket ">=0.25"
        for c in FEATURE_NAMES:
            r[c] = 1.0
        rows.append(r)
    df = pd.DataFrame(rows)
    # renormalize model probs
    m = df[["p_snia", "p_nonia", "p_other"]].to_numpy(float)
    df[["p_snia", "p_nonia", "p_other"]] = m / m.sum(axis=1, keepdims=True)
    return df


def test_fit_alpha_prefers_anchor_when_anchor_better():
    df = _cal_frame(240, anchor_good=True, model_good=False, seed=1)
    spec = fit_alpha(df)
    cells = spec.alpha_cells
    assert cells, "expected at least one fitted cell"
    # anchor is the informative signal → 1-SE rule pulls α toward 0
    assert all(c["alpha"] <= 0.5 for c in cells.values())
    # G3 per-cell guarantee: blend log-loss <= anchor log-loss
    for cell in spec.g3["per_cell"].values():
        assert cell["pass"]


def test_fit_alpha_prefers_model_when_model_better():
    df = _cal_frame(240, anchor_good=False, model_good=True, seed=2)
    spec = fit_alpha(df)
    assert any(c["alpha"] >= 0.75 for c in spec.alpha_cells.values())


def test_apply_blends_and_emits_columns():
    df = _cal_frame(240, anchor_good=True, model_good=False, seed=3)
    spec = fit_alpha(df)
    out = apply(df, spec)
    for c in ("p_snia_model", "p_nonia_model", "p_other_model",
              "p_snia_anchor", "alpha", "alpha_fallback_level"):
        assert c in out.columns
    # deployed probs are a simplex
    deployed = out[["p_snia", "p_nonia", "p_other"]].to_numpy(float)
    assert np.allclose(deployed.sum(axis=1), 1.0, atol=1e-6)
    # model copies preserved
    assert np.isfinite(out["p_snia_model"]).all()
    # blend identity: deployed == α·model + (1-α)·anchor
    a = out["alpha"].to_numpy(float)[:, None]
    model = out[["p_snia_model", "p_nonia_model", "p_other_model"]].to_numpy(float)
    anchor = out[["p_snia_anchor", "p_nonia_anchor", "p_other_anchor"]].to_numpy(float)
    expected = a * model + (1 - a) * np.where(np.isfinite(anchor), anchor, model)
    assert np.allclose(deployed, expected, atol=1e-9)


def test_apply_zero_expert_row_pinned_to_model():
    """A 0-expert row (no anchor) is pinned α=1 → deployed == model."""
    from debass_meta.features.lightcurve import FEATURE_NAMES

    row = {"survey": "ztf", "target_class": "other",
           "p_snia": 0.2, "p_nonia": 0.3, "p_other": 0.5}
    for c in FEATURE_NAMES:
        row[c] = 1.0
    df = pd.DataFrame([row])
    out = apply(df, BlendSpec())
    assert out["alpha"].iloc[0] == 1.0
    assert out["alpha_fallback_level"].iloc[0] == "no_anchor"
    assert out["p_snia_anchor"].isna().iloc[0]
    assert out["p_snia"].iloc[0] == pytest.approx(0.2)


def test_apply_fallback_ladder_to_survey():
    """A cell absent from the table falls back to the per-survey α."""
    spec = BlendSpec(alpha_survey={"ztf": {"alpha": 0.25, "n": 100}},
                     alpha_global={"alpha": 0.75, "n": 200})
    a, lvl = spec.lookup_alpha("ztf", "2+", ">=0.25", has_anchor=True)
    assert (a, lvl) == (0.25, "survey")
    # Survey-guarded global rung: lsst contributed nothing to this fit, so it
    # must NOT inherit the (ztf-fit) global α — it falls back to the anchor.
    a2, lvl2 = spec.lookup_alpha("lsst", "1", "<0.25", has_anchor=True)
    assert (a2, lvl2) == (0.0, "anchor_default")


def test_global_rung_reachable_only_for_fitted_surveys():
    """The global α applies to a fitted survey's unfitted buckets, never to a
    survey absent from the fit (the smoke-found ZTF→LSST α=1 carry-over)."""
    spec = BlendSpec(alpha_cells={"ztf|2+|>=0.25": {"alpha": 1.0, "n": 400}},
                     alpha_global={"alpha": 1.0, "n": 400})
    assert spec.fitted_surveys() == {"ztf"}
    # ztf, different bucket -> global rung is fine (same survey as the fit)
    a, lvl = spec.lookup_alpha("ztf", "1", "<0.25", has_anchor=True)
    assert (a, lvl) == (1.0, "global")
    # lsst never seen by the fit -> anchor fallback, NOT the ztf-fit α=1
    a2, lvl2 = spec.lookup_alpha("lsst", "2+", ">=0.25", has_anchor=True)
    assert (a2, lvl2) == (0.0, "anchor_default")
    # no-anchor rows keep the α=1 pin regardless of survey
    a3, lvl3 = spec.lookup_alpha("lsst", "0", "<0.25", has_anchor=False)
    assert (a3, lvl3) == (1.0, "no_anchor")


def test_blendspec_roundtrip(tmp_path):
    df = _cal_frame(240, anchor_good=True, model_good=False, seed=4)
    spec = fit_alpha(df, out_dir=tmp_path)
    assert (tmp_path / "blend.json").exists()
    reloaded = BlendSpec.load(tmp_path)
    assert reloaded.alpha_cells == spec.alpha_cells
    assert reloaded.base_rates == spec.base_rates


def test_honesty_filter_drops_broker_and_member_labels():
    df = _cal_frame(60, anchor_good=True, model_good=False, seed=5)
    df.loc[:9, "label_source"] = "broker_consensus"
    df.loc[10:19, "label_source"] = "fink/snn"  # anchor-member label
    mask = anchor_blend._honesty_mask(df)
    assert mask.sum() == len(df) - 20
    assert not mask[:20].any()
