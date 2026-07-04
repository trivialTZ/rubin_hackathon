"""v10 seq-classifier tests: K-fold OOF routing, grouped weak-label loss,
4-way → ternary folding (model, expert, projector), fine-label mapping, and
the honest-eval split helpers (inner-val carve + fold assignment).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from debass_meta.features.sequence_dataset import NormStats, sequence_arrays  # noqa: E402
from debass_meta.models.seq_classifier import (  # noqa: E402
    CLASSES,
    SN_GROUP,
    TERNARY_CLASSES,
    SeqClassifier,
    SeqClassifierArtifact,
    classification_loss,
    fine_class_from_raw,
    fit_temperature,
    fold_probs_to_ternary,
    supervision_mask,
)
from debass_meta.projectors import base as proj_base  # noqa: E402

_TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_seq_classifier.py"


def _load_train_module():
    if "tsc_v10" in sys.modules:
        return sys.modules["tsc_v10"]
    spec = importlib.util.spec_from_file_location("tsc_v10", _TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tsc_v10"] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def _det(mjd, mag, band="g", magerr=0.1, snr=20.0):
    return {"mjd": mjd, "mag": mag, "band": band, "magerr": magerr, "snr": snr,
            "is_positive": True, "survey": "ZTF"}


def _lc(n=10, seed=0):
    rng = np.random.default_rng(seed)
    mjd, mag, dets = 60000.0, 19.5, []
    for k in range(n):
        mjd += float(rng.uniform(0.5, 2.5))
        mag += float(rng.normal(-0.1, 0.08))
        dets.append(_det(mjd, mag, band="g" if k % 2 else "r"))
    return dets


def _clf(seed=3, classes=CLASSES):
    torch.manual_seed(seed)
    return SeqClassifier(classes=classes).eval()


def _save_artifact(tmp_path: Path, name: str, seed: int, temperature: float = 1.0) -> Path:
    art = SeqClassifierArtifact(classifier=_clf(seed), norm_stats=NormStats(),
                                temperature=temperature, meta={})
    art.save(tmp_path / name)
    return tmp_path / name


# ── grouped weak-label loss ──────────────────────────────────────────────


def test_grouped_loss_masks_sn_subclass_gradient():
    """A weak stamp-SN row supervises the grouped sn-like mass only: zero
    signal on the snia-vs-snii distinction."""
    logits = torch.tensor([[[1.3, -0.4, 0.2, 0.7]]], requires_grad=True)
    lengths = torch.tensor([1])
    w = torch.tensor([1.0])
    sn_mask = torch.tensor([[c in SN_GROUP for c in CLASSES]])

    loss, n_terms = classification_loss(logits, None, lengths, w, class_masks=sn_mask)
    assert int(n_terms) == 1
    # Analytically the loss is −log Σ_{sn-like} p_c.
    p = torch.softmax(logits.detach()[0, 0], dim=-1)
    m = float(p[:3].sum())
    assert float(loss) == pytest.approx(-np.log(m), abs=1e-6)

    loss.backward()
    g = logits.grad[0, 0]
    # All sn-like logits pushed UP together (≤0), non_sn pushed down (≥0):
    # never one SN subclass against another.
    assert (g[:3] <= 1e-8).all() and float(g[3]) >= -1e-8
    # Swapping snia/snii logits leaves the loss unchanged — the loss carries
    # no information about which SN subclass.
    swapped = logits.detach().clone()
    swapped[0, 0, [0, 1]] = swapped[0, 0, [1, 0]]
    loss_sw, _ = classification_loss(swapped, None, lengths, w, class_masks=sn_mask)
    assert float(loss_sw) == pytest.approx(float(loss), abs=1e-6)

    # Contrast: a full snii label DOES push snia down (positive gradient).
    logits2 = torch.tensor([[[1.3, -0.4, 0.2, 0.7]]], requires_grad=True)
    loss2, _ = classification_loss(logits2, torch.tensor([1]), lengths, w)
    loss2.backward()
    assert float(logits2.grad[0, 0, 0]) > 0


def test_grouped_loss_reduces_to_plain_ce_for_onehot():
    torch.manual_seed(0)
    logits = torch.randn(3, 4, len(CLASSES))
    labels = torch.tensor([0, 2, 3])
    lengths = torch.tensor([2, 4, 3])
    w = torch.tensor([1.0, 0.5, 2.0])
    loss_labels, n1 = classification_loss(logits, labels, lengths, w)
    onehot = torch.nn.functional.one_hot(labels, len(CLASSES)).bool()
    loss_masks, n2 = classification_loss(logits, None, lengths, w, class_masks=onehot)
    assert float(loss_labels) == pytest.approx(float(loss_masks), abs=1e-6)
    assert int(n1) == int(n2) == 9


def test_classification_loss_rejects_empty_mask():
    logits = torch.zeros(1, 2, len(CLASSES))
    with pytest.raises(ValueError):
        classification_loss(logits, None, torch.tensor([2]), torch.tensor([1.0]),
                            class_masks=torch.zeros(1, len(CLASSES), dtype=torch.bool))


def test_fit_temperature_with_grouped_masks():
    rng = np.random.default_rng(1)
    n = 300
    masks = np.zeros((n, len(CLASSES)), dtype=bool)
    y = rng.integers(0, len(CLASSES), size=n)
    masks[np.arange(n), y] = True
    masks[: n // 3, :3] = True  # a third of rows grouped over sn-like
    base = rng.normal(0, 1, size=(n, len(CLASSES)))
    base[np.arange(n), y] += 1.0
    t = fit_temperature(base * 4.0, None, class_masks=masks)
    assert 2.0 < t < 8.0


# ── label machinery ──────────────────────────────────────────────────────


def test_supervision_mask():
    assert list(CLASSES) == ["snia", "snii", "other_sn", "non_sn"]
    np.testing.assert_array_equal(
        supervision_mask("nonIa_snlike", grouped_sn=True), [True, True, True, False])
    np.testing.assert_array_equal(
        supervision_mask("nonIa_snlike"), [False, True, True, False])
    np.testing.assert_array_equal(
        supervision_mask("nonIa_snlike", fine="snii"), [False, True, False, False])
    np.testing.assert_array_equal(
        supervision_mask("snia"), [True, False, False, False])
    np.testing.assert_array_equal(
        supervision_mask("other"), [False, False, False, True])
    with pytest.raises(ValueError):
        supervision_mask("??")


def test_fine_class_from_raw():
    assert fine_class_from_raw("SN Ia") == "snia"
    assert fine_class_from_raw("SN Ia-91T") == "snia"
    assert fine_class_from_raw("SN Iax[02cx-like]") == "snia"
    for s in ("SN II", "SN IIn", "SN IIb", "SN IIP", "SN IIL", "SN II-pec",
              "SN IIn-pec", "SLSN-II"):
        assert fine_class_from_raw(s) == "snii", s
    for s in ("SN Ib", "SN Ic", "SN Ib/c", "SN Ic-BL", "SN Ibn", "SN Icn",
              "SLSN-I", "SLSN-R", "SN I"):
        assert fine_class_from_raw(s) == "other_sn", s
    for s in ("TDE", "TDE-He", "AGN", "CV", "Nova", "Varstar", "Galaxy", "QSO", "LBV"):
        assert fine_class_from_raw(s) == "non_sn", s
    # Ambiguous / ternary-token inputs stay unresolved.
    for s in ("SN", "AT", "-", "", None, "nonIa_snlike", "SLSN", float("nan")):
        assert fine_class_from_raw(s) is None, s


def test_fold_probs_to_ternary():
    p4 = np.array([0.1, 0.3, 0.2, 0.4])
    p3 = fold_probs_to_ternary(p4, CLASSES)
    np.testing.assert_allclose(p3, [0.1, 0.5, 0.4], atol=1e-12)
    # Batched + ternary passthrough.
    batch = np.tile(p4, (5, 1))
    np.testing.assert_allclose(fold_probs_to_ternary(batch, CLASSES).sum(axis=1), 1.0)
    p3_in = np.array([0.6, 0.3, 0.1])
    np.testing.assert_allclose(fold_probs_to_ternary(p3_in, TERNARY_CLASSES), p3_in)


# ── projector folding ────────────────────────────────────────────────────


def test_projector_folds_fourway_to_ternary():
    events = [
        {"class_name": "snia", "canonical_projection": 0.1},
        {"class_name": "snii", "canonical_projection": 0.3},
        {"class_name": "other_sn", "canonical_projection": 0.2},
        {"class_name": "non_sn", "canonical_projection": 0.4},
    ]
    out = proj_base._dispatch_projector("seq_v9", events)
    assert out["p_snia"] == pytest.approx(0.1, abs=1e-6)
    assert out["p_nonIa_snlike"] == pytest.approx(0.5, abs=1e-6)
    assert out["p_other"] == pytest.approx(0.4, abs=1e-6)
    assert out["mapped_pred_class"] == "nonIa_snlike"

    # p4_-prefixed extras fold identically.
    events_p4 = [{"class_name": f"p4_{c}", "canonical_projection": v}
                 for c, v in zip(CLASSES, (0.1, 0.3, 0.2, 0.4))]
    out_p4 = proj_base._dispatch_projector("seq_v9", events_p4)
    assert out_p4["p_nonIa_snlike"] == pytest.approx(0.5, abs=1e-6)

    # When canonical ternary events are present (the expert already folded),
    # they win and the p4 extras are not double-counted.
    mixed = [
        {"class_name": "snia", "canonical_projection": 0.1},
        {"class_name": "nonIa_snlike", "canonical_projection": 0.5},
        {"class_name": "other", "canonical_projection": 0.4},
    ] + events_p4
    out_mixed = proj_base._dispatch_projector("seq_v9", mixed)
    assert out_mixed["p_snia"] == pytest.approx(0.1, abs=1e-6)
    assert out_mixed["p_nonIa_snlike"] == pytest.approx(0.5, abs=1e-6)
    assert out_mixed["p_other"] == pytest.approx(0.4, abs=1e-6)


# ── K-fold OOF routing ───────────────────────────────────────────────────


def test_fold_routing_uses_model_that_never_saw_the_object(tmp_path: Path):
    from debass_meta.experts.local.seq_v9 import SeqV9Expert

    root = tmp_path / "m"
    _save_artifact(tmp_path, "m", seed=1)
    _save_artifact(root, "fold_0", seed=2)
    _save_artifact(root, "fold_1", seed=3)
    (root / "fold_map.json").write_text(json.dumps({
        "n_folds": 2, "seed": 42,
        "assignments": {"OBJ_A": 0, "OBJ_B": 1},
    }))

    lc = _lc(6)
    cont, bands = sequence_arrays(lc)
    expected = {
        name: SeqClassifierArtifact.load(root / name).predict_proba_prefixes(cont, bands)[-1]
        for name in ("fold_0", "fold_1")
    }
    expected_full = SeqClassifierArtifact.load(root).predict_proba_prefixes(cont, bands)[-1]

    ex = SeqV9Expert(model_dir=root)
    out_a = ex.predict_epoch("OBJ_A", lc, epoch_jd=2460000.5)
    out_b = ex.predict_epoch("OBJ_B", lc, epoch_jd=2460000.5)
    out_c = ex.predict_epoch("OBJ_C", lc, epoch_jd=2460000.5)  # unmapped → full

    # OBJ_A (fold 0) must be scored by fold_0 — the model trained WITHOUT
    # fold 0 — and must NOT match the full-data model.
    for out, name in ((out_a, "fold_0"), (out_b, "fold_1")):
        p3 = fold_probs_to_ternary(expected[name], CLASSES)
        assert out.class_probabilities["snia"] == pytest.approx(p3[0], abs=1e-6)
        assert out.class_probabilities["nonIa_snlike"] == pytest.approx(p3[1], abs=1e-6)
    assert out_a.raw_output["fold_route"] == 0
    assert out_b.raw_output["fold_route"] == 1
    assert out_a.class_probabilities["snia"] != pytest.approx(
        float(fold_probs_to_ternary(expected_full, CLASSES)[0]), abs=1e-6)

    p3_full = fold_probs_to_ternary(expected_full, CLASSES)
    assert out_c.raw_output["fold_route"] is None
    assert out_c.class_probabilities["snia"] == pytest.approx(p3_full[0], abs=1e-6)

    md = ex.metadata()
    assert md["oof_fold_routing"] is True and md["n_fold_mapped_objects"] == 2


def test_fold_routing_falls_back_to_full_when_fold_dir_missing(tmp_path: Path):
    from debass_meta.experts.local.seq_v9 import SeqV9Expert

    root = tmp_path / "m"
    _save_artifact(tmp_path, "m", seed=1)
    (root / "fold_map.json").write_text(json.dumps({"assignments": {"OBJ_A": 0}}))
    ex = SeqV9Expert(model_dir=root)
    out = ex.predict_epoch("OBJ_A", _lc(5), epoch_jd=2460000.5)
    assert out.available is True
    assert out.raw_output["fold_route"] == "missing_fold_0_fallback_full"


def test_no_fold_map_behaves_like_single_artifact(tmp_path: Path):
    from debass_meta.experts.local.seq_v9 import SeqV9Expert

    root = _save_artifact(tmp_path, "m", seed=1)
    ex = SeqV9Expert(model_dir=root)
    out = ex.predict_epoch("ANY", _lc(5), epoch_jd=2460000.5)
    assert out.available is True and out.raw_output["fold_route"] is None
    assert ex.metadata()["oof_fold_routing"] is False


def test_corrupt_fold_map_fails_loudly(tmp_path: Path):
    """A fold_map.json that EXISTS but cannot be parsed must raise — a silent
    {} fallback would route every train object to the full-data root model
    (the v9c stacking trap) with no error anywhere in the chain."""
    from debass_meta.experts.local.seq_v9 import SeqV9Expert

    root = _save_artifact(tmp_path, "m", seed=1)
    (root / "fold_map.json").write_text('{"assignments": {"OBJ_A": ')  # truncated
    ex = SeqV9Expert(model_dir=root)
    with pytest.raises(RuntimeError, match="corrupt fold_map.json"):
        ex.predict_epoch("OBJ_A", _lc(5), epoch_jd=2460000.5)


# ── backward compatibility: legacy ternary (v9c) artifacts ──────────────


def test_legacy_ternary_artifact_roundtrip_and_expert(tmp_path: Path):
    from debass_meta.experts.local.seq_v9 import SeqV9Expert

    clf3 = _clf(seed=5, classes=TERNARY_CLASSES)
    art = SeqClassifierArtifact(classifier=clf3, norm_stats=NormStats(),
                                temperature=1.2, meta={})
    art.save(tmp_path / "m3")
    meta = json.loads((tmp_path / "m3" / "config.json").read_text())
    assert meta["classes"] == list(TERNARY_CLASSES)

    art2 = SeqClassifierArtifact.load(tmp_path / "m3")
    assert art2.classes == TERNARY_CLASSES
    cont, bands = sequence_arrays(_lc(6))
    assert art2.predict_proba_prefixes(cont, bands).shape[1] == 3

    ex = SeqV9Expert(model_dir=tmp_path / "m3")
    out = ex.predict_epoch("ZTFtest", _lc(5), epoch_jd=2460000.5)
    assert out.available is True
    assert set(out.class_probabilities) == {"snia", "nonIa_snlike", "other"}  # no p4 extras
    assert sum(out.class_probabilities.values()) == pytest.approx(1.0, abs=1e-5)


# ── honest-eval split helpers ────────────────────────────────────────────


def test_inner_val_split_disjoint_and_deterministic():
    mod = _load_train_module()
    ids = [f"obj{i:03d}" for i in range(87)]
    fit1, val1 = mod.split_inner_val(ids, 0.10, seed=7)
    fit2, val2 = mod.split_inner_val(list(reversed(ids)), 0.10, seed=7)
    assert fit1 == fit2 and val1 == val2                      # deterministic, order-free
    assert set(fit1) & set(val1) == set()                     # disjoint
    assert set(fit1) | set(val1) == set(ids)                  # exhaustive
    assert len(val1) == round(0.10 * len(ids))
    _, val_other = mod.split_inner_val(ids, 0.10, seed=8)
    assert val_other != val1                                  # seed-sensitive


def test_assign_folds_grouped_deterministic_balanced():
    mod = _load_train_module()
    ids = [f"obj{i:03d}" for i in range(103)]
    fm1 = mod.assign_folds(ids, 5, seed=42)
    fm2 = mod.assign_folds(list(reversed(ids)), 5, seed=42)
    assert fm1 == fm2
    assert set(fm1) == set(ids)                               # every object mapped once
    counts = np.bincount(list(fm1.values()), minlength=5)
    assert counts.min() >= 1 and counts.max() - counts.min() <= 1


def test_label_loaders_grouped_supervision():
    mod = _load_train_module()
    # Weak stamp-SN → grouped sn-like supervision (NEVER a forced subclass).
    csv_path = Path(_TRAIN_SCRIPT).parent.parent / "nonexistent.csv"
    assert mod.load_lsst_candidate_labels(csv_path) == {}

    labels = {
        "spec_nonia": mod.RowLabel("nonIa_snlike", "spectroscopic"),
        "consensus_nonia": mod.RowLabel("nonIa_snlike", "consensus"),
        "weak_nonia": mod.RowLabel("nonIa_snlike", "weak"),
        "spec_snia": mod.RowLabel("snia", "spectroscopic"),
        "spec_snii": mod.RowLabel("nonIa_snlike", "spectroscopic"),
        "conflict": mod.RowLabel("other", "spectroscopic"),
    }
    fine_map = {"spec_snii": "snii", "conflict": "snii"}  # snii contradicts ternary 'other'
    n_fine, n_grouped = mod.attach_fine_labels(labels, fine_map)
    assert labels["spec_snii"].fine == "snii" and n_fine == 1
    assert labels["conflict"].fine is None                    # contradiction dropped
    assert labels["weak_nonia"].grouped_sn is True            # weak SN → grouped mass
    # ANY nonIa row without a parseable subtype gets grouped supervision —
    # quality tier alone never earns "definitely not Ia" (generic TNS "SN",
    # legacy BTS '-', consensus claims cannot exclude Ia).
    assert labels["spec_nonia"].grouped_sn is True
    assert labels["consensus_nonia"].grouped_sn is True
    assert labels["spec_snii"].grouped_sn is False            # parsed subtype → one-hot
    assert n_grouped == 3


def test_lsst_candidate_loader_grouped_sn(tmp_path: Path):
    mod = _load_train_module()
    csv_path = tmp_path / "cand.csv"
    csv_path.write_text(
        "object_id,alerce_stamp_class\n111,SN\n222,AGN\n333,\n")
    out = mod.load_lsst_candidate_labels(csv_path)
    assert out["111"].ternary == "nonIa_snlike" and out["111"].grouped_sn is True
    np.testing.assert_array_equal(
        supervision_mask(out["111"].ternary, out["111"].fine, out["111"].grouped_sn),
        [True, True, True, False])                            # snia mass NOT excluded
    assert out["222"].fine == "non_sn" and out["222"].grouped_sn is False
    assert "333" not in out
