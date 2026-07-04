"""seq_v9 classifier + expert + projector tests.

Covers: per-prefix causality of the classifier, loss masking/weighting,
temperature fitting, artifact round-trip, the LocalExpert wrapper contract
(incl. graceful unavailability), registry/projector wiring, and the LSST
weak-label mapping (SN → nonIa_snlike, never snia).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from debass_meta.features.sequence_dataset import NormStats, sequence_arrays  # noqa: E402
from debass_meta.models.seq_classifier import (  # noqa: E402
    CLASSES,
    SeqClassifier,
    SeqClassifierArtifact,
    classification_loss,
    fit_temperature,
    predict_prefix_proba,
)
from debass_meta.projectors import base as proj_base  # noqa: E402


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


def _clf(seed=3):
    torch.manual_seed(seed)
    return SeqClassifier().eval()


def test_classifier_prefix_causality():
    clf = _clf()
    stats = NormStats()
    full = _lc(12)
    cont_f, bands_f = sequence_arrays(full)
    p_full = predict_prefix_proba(clf, stats.apply(cont_f), bands_f)
    for k in (1, 4, 9):
        cont_p, bands_p = sequence_arrays(full[:k])
        p_pre = predict_prefix_proba(clf, stats.apply(cont_p), bands_p)
        np.testing.assert_allclose(p_full[:k], p_pre, atol=1e-5)
    np.testing.assert_allclose(p_full.sum(axis=1), 1.0, atol=1e-5)


def test_classification_loss_masks_and_object_weighting():
    clf = _clf()
    seqs = [sequence_arrays(_lc(n, seed=n)) for n in (4, 9)]
    L = max(len(c) for c, _ in seqs)
    cont = np.zeros((2, L, seqs[0][0].shape[1]), dtype=np.float32)
    bands = np.zeros((2, L), dtype=np.int64)
    for i, (c, b) in enumerate(seqs):
        cont[i, : len(c)] = c
        bands[i, : len(b)] = b
    lengths = torch.tensor([4, 9])
    logits = clf(torch.from_numpy(cont), torch.from_numpy(bands))
    labels = torch.tensor([0, 2])
    w = torch.tensor([1.0, 1.0])
    loss, n_terms = classification_loss(logits, labels, lengths, w)
    assert int(n_terms) == 13  # 4 + 9 real steps
    assert float(loss) > 0
    # Doubling one object's weight changes the loss (weighting is live).
    loss2, _ = classification_loss(logits, labels, lengths, torch.tensor([2.0, 1.0]))
    assert abs(float(loss2) - float(loss)) > 1e-8


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=400)
    # well-calibrated logits, then artificially sharpened ×4 → T should be ≈4.
    base = rng.normal(0, 1, size=(400, 3))
    base[np.arange(400), y] += 1.0
    t = fit_temperature(base * 4.0, y)
    assert 2.5 < t < 6.5
    # already-calibrated logits → T stays near 1 (guarded fallback allows 1.0)
    t1 = fit_temperature(base, y)
    assert 0.5 < t1 < 2.0


def test_artifact_roundtrip(tmp_path: Path):
    clf = _clf()
    art = SeqClassifierArtifact(
        classifier=clf, norm_stats=NormStats(), temperature=1.7,
        meta={"n_train_objects": 10},
    )
    art.save(tmp_path / "m")
    art2 = SeqClassifierArtifact.load(tmp_path / "m")
    assert art2.temperature == pytest.approx(1.7)
    cont, bands = sequence_arrays(_lc(6))
    np.testing.assert_allclose(
        art.predict_proba_prefixes(cont, bands),
        art2.predict_proba_prefixes(cont, bands), atol=1e-6,
    )
    meta = json.loads((tmp_path / "m" / "config.json").read_text())
    assert meta["classes"] == list(CLASSES)


def test_expert_wrapper_contract(tmp_path: Path):
    from debass_meta.experts.local.seq_v9 import SeqV9Expert

    # Unavailable artifact → available=False, never raises.
    ex = SeqV9Expert(model_dir=tmp_path / "missing")
    out = ex.predict_epoch("ZTFtest", _lc(5), epoch_jd=2460000.5)
    assert out.available is False and out.expert == "seq_v9"

    # Real (4-way) artifact → ternary contract keys summing to 1, plus the
    # p4_* fine-head extras persisted for the silver payload.
    art = SeqClassifierArtifact(classifier=_clf(), norm_stats=NormStats(),
                                temperature=1.0, meta={})
    art.save(tmp_path / "m")
    ex2 = SeqV9Expert(model_dir=tmp_path / "m")
    out2 = ex2.predict_epoch("ZTFtest", _lc(5), epoch_jd=2460000.5)
    assert out2.available is True
    ternary = {"snia", "nonIa_snlike", "other"}
    p4 = {f"p4_{c}" for c in CLASSES}
    assert set(out2.class_probabilities) == ternary | p4
    assert sum(out2.class_probabilities[k] for k in ternary) == pytest.approx(1.0, abs=1e-5)
    assert sum(out2.class_probabilities[k] for k in p4) == pytest.approx(1.0, abs=1e-5)
    # The folded ternary must equal the 4-way fold.
    assert out2.class_probabilities["nonIa_snlike"] == pytest.approx(
        out2.class_probabilities["p4_snii"] + out2.class_probabilities["p4_other_sn"], abs=1e-6)
    assert out2.class_probabilities["other"] == pytest.approx(
        out2.class_probabilities["p4_non_sn"], abs=1e-6)
    assert out2.raw_output["n_det_used"] == 5
    assert out2.raw_output["fold_route"] is None  # no fold_map.json → full model
    assert set(out2.raw_output["class_probabilities_4way"]) == set(CLASSES)
    md = ex2.metadata()
    assert md["available"] is True and "LSST" in md["surveys"]
    assert md["classes"] == ["snia", "nonIa_snlike", "other"]  # pipeline contract
    assert md["oof_fold_routing"] is False


def test_registry_and_projector_wiring():
    assert "seq_v9" in proj_base.EXPERT_REGISTRY
    assert proj_base.EXPERT_REGISTRY["seq_v9"][0] == "any"
    assert "seq_v9" in proj_base.ALL_EXPERT_KEYS
    events = [
        {"class_name": "snia", "canonical_projection": 0.6},
        {"class_name": "nonIa_snlike", "canonical_projection": 0.3},
        {"class_name": "other", "canonical_projection": 0.1},
    ]
    out = proj_base._dispatch_projector("seq_v9", events)
    assert out["p_snia"] == pytest.approx(0.6, abs=1e-6)
    assert out["p_nonIa_snlike"] == pytest.approx(0.3, abs=1e-6)
    assert out["mapped_pred_class"] == "snia"
    empty = proj_base._dispatch_projector("seq_v9", [{"class_name": "??", "canonical_projection": None}])
    assert "reason" in empty


def test_lsst_candidate_label_mapping(tmp_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bsf", Path(__file__).resolve().parents[1] / "scripts" / "build_snapshots_fusion.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    csv_path = tmp_path / "cand.csv"
    csv_path.write_text(
        "object_id,alerce_stamp_class\n"
        "111,SN\n222,AGN\n333,asteroid\n444,VS\n555,bogus\n666,\n"
    )
    truth = mod.load_lsst_candidate_truth(csv_path)
    assert truth["111"]["final_class_ternary"] == "nonIa_snlike"   # NEVER snia
    assert truth["111"]["label_quality"] == "weak"
    assert truth["111"]["label_source"] == "alerce_self_label"     # provenance masking hook
    for oid in ("222", "333", "444", "555"):
        assert truth[oid]["final_class_ternary"] == "other"
    assert "666" not in truth
    assert all(e["follow_proxy"] == 0 for e in truth.values())
