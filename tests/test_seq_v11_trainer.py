"""WP2 seq-classifier trainer tests: the marginalized coarse-label loss +
auditable tier→allowed-set table, length-deconfounded sampling, random-phase
windows (train-only; eval/OOF/inference prove prefix-only), the per-survey eval
assertion, and the --extra-truth train-only merge.

All offline, tiny synthetic data.  The trainer is loaded as a module by path
(mirrors tests/test_seq_classifier_v10.py) so its dataclasses register.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from debass_meta.features.sequence_dataset import (  # noqa: E402
    NormStats,
    epoch_window_tokens,
    load_object_tokens,
    sequence_arrays,
)
from debass_meta.models.seq_classifier import (  # noqa: E402
    CLASSES,
    supervision_mask,
)

_TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_seq_classifier.py"


def _load_train_module():
    if "tsc_wp2" in sys.modules:
        return sys.modules["tsc_wp2"]
    spec = importlib.util.spec_from_file_location("tsc_wp2", _TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tsc_wp2"] = mod
    spec.loader.exec_module(mod)
    return mod


def _det(mjd, mag, band="g", magerr=0.1, snr=20.0, survey="ZTF"):
    return {"mjd": mjd, "mag": mag, "band": band, "magerr": magerr, "snr": snr,
            "is_positive": True, "survey": survey, "flux": 500.0,
            "fluxerr": 20.0, "quality": 0.9}


def _lc(n=10, seed=0, survey="ZTF"):
    rng = np.random.default_rng(seed)
    mjd, mag, dets = 60000.0, 19.5, []
    for k in range(n):
        mjd += float(rng.uniform(0.5, 2.5))
        mag += float(rng.normal(-0.1, 0.08))
        dets.append(_det(mjd, mag, band="g" if k % 2 else "r", survey=survey))
    return dets


# ── 1. marginalized coarse-label loss + auditable tier→set table ─────────────


def test_tier_allowed_classes_table_is_auditable():
    mod = _load_train_module()
    # Every SN-ish tier admits the whole SN group (SN-vs-non_sn, never a
    # forced subclass, NEVER excludes Ia); non-SN tiers admit only non_sn.
    assert mod.TIER_ALLOWED_CLASSES["spectroscopic"] == ("snia", "snii", "other_sn")
    assert mod.TIER_ALLOWED_CLASSES["tns_untyped"] == ("snia", "snii", "other_sn")
    assert mod.TIER_ALLOWED_CLASSES["bts_untyped"] == ("snia", "snii", "other_sn")
    assert mod.TIER_ALLOWED_CLASSES["context"] == ("non_sn",)
    assert mod.TIER_ALLOWED_CLASSES["weak"] is None  # decided by label source


def test_allowed_classes_matches_supervision_mask():
    """allowed_classes_for is a faithful re-expression of supervision_mask."""
    mod = _load_train_module()
    cases = [
        ("snia", None, False),           # spec Ia
        ("nonIa_snlike", "snii", False),  # parsed subtype → singleton
        ("nonIa_snlike", None, True),     # grouped SN → SN-vs-non_sn
        ("other", None, False),           # non-SN
        ("other", "non_sn", False),       # parsed non_sn
    ]
    for ternary, fine, grouped in cases:
        allowed = mod.allowed_classes_for(ternary, fine, grouped)
        got = mod.mask_from_allowed(allowed)
        want = supervision_mask(ternary, fine, grouped)
        np.testing.assert_array_equal(got, want, err_msg=str((ternary, fine, grouped)))


def test_exclude_tiers_drops_row_from_classification_loss():
    mod = _load_train_module()
    assert mod.allowed_classes_for("nonIa_snlike", None, True, "weak") == mod.SN_LIKE_CLASSES
    # Excluding the 'weak' tier gives an empty allowed set → no cls supervision.
    assert mod.allowed_classes_for("nonIa_snlike", None, True, "weak",
                                   exclude_tiers=frozenset({"weak"})) == ()


def test_marginalized_nll_handcomputed_singleton_and_group():
    """The marginalized loss = −log Σ_{c∈allowed} softmax(logits)_c.

    Singleton allowed set == plain cross-entropy; a 2-class allowed set ==
    −log(p_a + p_b) (hand-computed)."""
    mod = _load_train_module()
    logits = torch.tensor([[2.0, 0.5, -1.0, 0.25]])
    p = torch.softmax(logits[0], dim=-1)

    # singleton {snia} == CE on class 0
    m1 = torch.tensor([[True, False, False, False]])
    nll1 = mod.marginalized_nll(logits, m1)[0]
    assert float(nll1) == pytest.approx(float(-torch.log(p[0])), abs=1e-6)
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    assert float(nll1) == pytest.approx(float(ce), abs=1e-6)

    # allowed {snii, other_sn} == −log(p1 + p2)
    m2 = torch.tensor([[False, True, True, False]])
    nll2 = mod.marginalized_nll(logits, m2)[0]
    assert float(nll2) == pytest.approx(float(-torch.log(p[1] + p[2])), abs=1e-6)

    # SN group {snia,snii,other_sn} == −log(1 − p_non_sn)
    m3 = torch.tensor([[True, True, True, False]])
    nll3 = mod.marginalized_nll(logits, m3)[0]
    assert float(nll3) == pytest.approx(float(-torch.log(1.0 - p[3])), abs=1e-6)


# ── 2. length-deconfounded sampling ──────────────────────────────────────────


def _skewed_seqset(mod, seed=0):
    """A synthetic set where class correlates with length: class A objects are
    SHORT (~SNe), class B objects are LONG (~persistent variables), with a few
    counter-examples so both classes reach every bucket."""
    rng = np.random.default_rng(seed)
    s = mod.SeqSet()
    y3, is_lsst = [], []
    specs = ([("A", 3)] * 20 + [("B", 15)] * 20          # dominant length↔class link
             + [("A", 15)] * 3 + [("B", 3)] * 3          # a few counter-examples
             + [("A", 8)] * 6 + [("B", 8)] * 6)          # mid-bucket coverage
    for k, (cls, L) in enumerate(specs):
        cont, bands = sequence_arrays(_lc(L, seed=int(rng.integers(1e6))))
        s.oids.append(f"o{k}")
        s.seqs.append((cont, bands))
        y3.append(0 if cls == "A" else 1)
        is_lsst.append(False)
    s.y3 = np.array(y3, dtype=np.int64)
    s.quals = np.ones(len(s.oids))
    s.is_lsst = np.array(is_lsst, dtype=bool)
    s.masks = np.zeros((len(s.oids), 4), dtype=bool)
    return s


def _sampled_bucket_class_fracs(mod, rows, weights, tr, n_draw=200_000, seed=1):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(rows), size=n_draw, replace=True, p=weights)
    picked = rows[sel]
    cls = tr.y3[picked[:, 0]]
    bkt = np.array([mod.ndet_bucket(int(n)) for n in picked[:, 1]], dtype=object)
    fracs = {}
    for b in set(map(tuple, bkt)):
        sel_b = np.array([tuple(x) == b for x in bkt])
        if sel_b.sum() > 0:
            fracs[b] = float((cls[sel_b] == 0).mean())  # P(class A | bucket)
    return fracs


def test_balanced_sampling_flattens_class_across_buckets():
    mod = _load_train_module()
    tr = _skewed_seqset(mod)
    fit_idx = np.arange(len(tr.oids))

    rows_u, w_u = mod.build_training_rows(tr, fit_idx, balanced=False)
    rows_b, w_b = mod.build_training_rows(tr, fit_idx, balanced=True)
    assert len(rows_b) == len(rows_u) > 0
    np.testing.assert_allclose(w_b.sum(), 1.0, atol=1e-9)

    fr_u = _sampled_bucket_class_fracs(mod, rows_u, w_u, tr)
    fr_b = _sampled_bucket_class_fracs(mod, rows_b, w_b, tr)

    # UNBALANCED: the short bucket is dominated by class A, the long bucket by
    # class B — the length shortcut is visible.
    assert fr_u[(1, 3)] > 0.7 and fr_u[(11, 20)] < 0.3
    # BALANCED: within every bucket the two classes are ~50/50, so class no
    # longer predicts length.
    for b, frac in fr_b.items():
        assert abs(frac - 0.5) < 0.08, f"bucket {b} not flat: P(A)={frac:.3f}"


def test_balanced_weights_per_object_mass_is_one_over_nrows():
    """Base weight is quality/n_rows so each object contributes once."""
    mod = _load_train_module()
    tr = _skewed_seqset(mod)
    fit_idx = np.arange(len(tr.oids))
    rows, w = mod.build_training_rows(tr, fit_idx, balanced=False)
    # per-object summed weight equal across objects (all quality=1), = 1/N_obj
    per_obj = np.zeros(len(tr.oids))
    for (i, _), wi in zip(rows, w):
        per_obj[int(i)] += wi
    np.testing.assert_allclose(per_obj, 1.0 / len(tr.oids), atol=1e-9)


# ── 3. random-phase windows (train-only) — eval/inference invariance ─────────


def _tokens_seqset(mod, n=18):
    """One long object with cached tokens (v9) so random windows can fire."""
    s = mod.SeqSet()
    lc = _lc(n, seed=7)
    cont, bands = sequence_arrays(lc, schema="v9")
    s.oids.append("long")
    s.seqs.append((cont, bands))
    s.tokens.append(epoch_window_tokens(lc, schema="v9"))
    s.y3 = np.array([0], dtype=np.int64)
    s.quals = np.ones(1)
    s.is_lsst = np.zeros(1, dtype=bool)
    s.masks = np.zeros((1, 4), dtype=bool)
    s.masks[0, 0] = True
    return s


def test_random_windows_disabled_is_exact_prefix():
    """With random_windows=False the batch is byte-identical to the prefix
    slice — the eval/OOF/inference contract."""
    mod = _load_train_module()
    tr = _tokens_seqset(mod)
    stats = NormStats()
    rows = np.array([[0, 5]], dtype=np.int64)
    rng = np.random.default_rng(0)
    cont, bands, lengths, masks = mod.build_row_batch(
        tr, rows, stats, torch.device("cpu"), schema="v9", rng=rng,
        random_windows=False)
    assert int(lengths[0]) == 5
    prefix = stats.apply(tr.seqs[0][0][:5])
    np.testing.assert_allclose(cont[0, :5].numpy(), prefix, atol=1e-6)


def test_random_windows_enabled_changes_some_rows_but_stays_causal():
    mod = _load_train_module()
    tr = _tokens_seqset(mod)
    stats = NormStats()
    rows = np.array([[0, 5]] * 40, dtype=np.int64)
    rng = np.random.default_rng(3)
    cont, bands, lengths, masks = mod.build_row_batch(
        tr, rows, stats, torch.device("cpu"), schema="v9", rng=rng,
        random_windows=True, rw_prob=1.0)  # always attempt a window
    assert (lengths == 5).all()
    prefix = stats.apply(tr.seqs[0][0][:5])
    # At least one row is a shifted (re-anchored) window ≠ the prefix.
    diffs = [not np.allclose(cont[j, :5].numpy(), prefix, atol=1e-5)
             for j in range(len(rows))]
    assert any(diffs), "random windows never fired"
    # Re-anchoring proof: a window starting at index 2 re-tensorizes as a FRESH
    # causal window — is_first (dim 5, RAW) is on row 0, not a stale mid-sequence
    # slice with the original positional features.
    win = sequence_arrays(tr.tokens[0][2:7], pre_truncated=True, schema="v9")[0]
    assert win[0, 5] == 1.0 and np.all(win[1:, 5] == 0.0)


def test_epoch_window_tokens_and_load_object_tokens(tmp_path):
    """The additive token helpers round-trip 1:1 through sequence_arrays."""
    mod = _load_train_module()  # noqa: F841 (ensures module import path works)
    import json as _json

    lc = _lc(6, seed=1)
    toks = epoch_window_tokens(lc, schema="v9")
    assert len(toks) == 6
    # tokens re-tensorize to the same tensor as the full sequence
    c_tok, _ = sequence_arrays(toks, pre_truncated=True, schema="v9")
    c_full, _ = sequence_arrays(lc, schema="v9")
    np.testing.assert_array_equal(c_tok, c_full)

    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    (lc_dir / "ZTFtok.json").write_text(_json.dumps(lc))
    loaded = load_object_tokens(lc_dir, "ZTFtok", schema="v9")
    assert loaded is not None and len(loaded) == 6
    assert load_object_tokens(lc_dir, "MISSING", schema="v9") is None


# ── 4. per-survey eval assertion ─────────────────────────────────────────────


def test_per_survey_eval_assertion_fires_and_overrides(tmp_path, monkeypatch):
    """A surveys='both' final-eval with a single-survey test set must abort,
    unless --allow-missing-survey-eval is passed.  Driven end-to-end through
    main() on a tiny synthetic corpus."""
    mod = _load_train_module()
    import json as _json

    import pandas as pd

    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    n_train = 140
    train_ids, cal_ids, test_ids = [], [], []
    truth_rows, snap_rows = [], []

    def add(oid, survey, ternary, quality, bucket):
        (lc_dir / f"{oid}.json").write_text(
            _json.dumps(_lc(6, seed=hash(oid) % 100000, survey=survey)))
        snap_rows.append({"object_id": oid, "target_class": ternary,
                          "label_quality": quality})
        truth_rows.append({"object_id": oid, "tns_type": "SN Ia" if ternary == "snia" else "",
                           "bts_type": "", "final_class_raw": ""})
        bucket.append(oid)

    # ZTF-only train + cal + TEST (no LSST anywhere) → 'both' must abort.
    for k in range(n_train):
        add(f"ZTFtr{k}", "ZTF", "snia" if k % 2 else "other", "spectroscopic", train_ids)
    for k in range(40):
        add(f"ZTFca{k}", "ZTF", "snia" if k % 2 else "other", "spectroscopic", cal_ids)
    for k in range(20):
        add(f"ZTFte{k}", "ZTF", "snia" if k % 2 else "other", "spectroscopic", test_ids)

    snap = tmp_path / "snap.parquet"
    truth = tmp_path / "truth.parquet"
    pd.DataFrame(snap_rows).to_parquet(snap, index=False)
    pd.DataFrame(truth_rows).to_parquet(truth, index=False)
    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"train_ids": train_ids, "cal_ids": cal_ids,
                                  "test_ids": test_ids}))

    base_argv = [
        "train_seq_classifier.py",
        "--snapshots", str(snap), "--split", str(split), "--truth-table", str(truth),
        "--lc-dir", str(lc_dir), "--encoder", str(tmp_path / "no_encoder"),
        "--no-lsst", "--out", str(tmp_path / "artifact"),
        "--surveys", "both", "--oof-folds", "0", "--final-eval",
        "--epochs", "1", "--freeze-epochs", "0", "--batch", "64", "--seed", "0",
    ]
    monkeypatch.setattr(sys, "argv", base_argv)
    with pytest.raises(SystemExit, match="per-survey blind spot"):
        mod.main()

    # With the override it completes and writes an artifact.
    monkeypatch.setattr(sys, "argv", base_argv + ["--allow-missing-survey-eval"])
    mod.main()
    assert (tmp_path / "artifact" / "classifier.pt").exists()


# ── 5. --extra-truth train-only merge ────────────────────────────────────────


def test_load_extra_truth_train_only_merge(tmp_path):
    mod = _load_train_module()
    import pandas as pd

    p = tmp_path / "extra.parquet"
    pd.DataFrame([
        {"object_id": "X1", "final_class_ternary": "snia", "label_quality": "spectroscopic",
         "tns_type": "SN Ia", "bts_type": "", "final_class_raw": ""},
        {"object_id": "X2", "final_class_ternary": "nonIa_snlike", "label_quality": "weak",
         "tns_type": "", "bts_type": "", "final_class_raw": ""},   # no subtype → grouped SN
        {"object_id": "X3", "final_class_ternary": "other", "label_quality": "context",
         "tns_type": "", "bts_type": "", "final_class_raw": "AGN"},
        {"object_id": "X4", "final_class_ternary": "garbage", "label_quality": "weak",
         "tns_type": "", "bts_type": "", "final_class_raw": ""},   # bad ternary → dropped
    ]).to_parquet(p, index=False)

    out = mod.load_extra_truth([str(p)])
    assert set(out) == {"X1", "X2", "X3"}
    assert out["X1"].fine == "snia" and out["X1"].grouped_sn is False
    assert out["X2"].grouped_sn is True and out["X2"].fine is None
    assert out["X3"].fine == "non_sn"

    # A missing parquet is skipped without crashing.
    assert mod.load_extra_truth([str(tmp_path / "nope.parquet")]) == {}


def test_extra_lc_dir_and_truth_flow_into_train_only(tmp_path):
    """Extra objects load from an extra lc dir via the SAME tokenizer and land
    ONLY in the train SeqSet (never cal/test)."""
    mod = _load_train_module()
    import json as _json

    main_dir = tmp_path / "lc"
    extra_dir = tmp_path / "extra_lc"
    main_dir.mkdir()
    extra_dir.mkdir()
    (main_dir / "ZTFmain.json").write_text(_json.dumps(_lc(6, seed=1)))
    (extra_dir / "LSSTextra.json").write_text(_json.dumps(_lc(7, seed=2, survey="LSST")))

    labels = {
        "ZTFmain": mod.RowLabel(ternary="snia", quality="spectroscopic"),
        "LSSTextra": mod.RowLabel(ternary="other", quality="weak", fine="non_sn"),
    }
    train_ids = {"ZTFmain"}
    all_split = {"ZTFmain"}  # LSSTextra absent from every split → allow_extra admits it

    tr = mod.collect(labels, train_ids, lc_dir=main_dir, max_len=20,
                     all_split_ids=all_split, allow_extra=True,
                     extra_lc_dirs=[extra_dir], keep_tokens=True)
    assert set(tr.oids) == {"ZTFmain", "LSSTextra"}
    assert tr.is_lsst[tr.oids.index("LSSTextra")]
    # keep_tokens populated aligned with seqs
    assert len(tr.tokens) == len(tr.oids)

    # cal/test collect (no extra dir, extra id not in those sets) → excluded.
    ca = mod.collect(labels, set(), lc_dir=main_dir, max_len=20)
    assert "LSSTextra" not in ca.oids
