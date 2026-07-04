"""v10 split-hygiene + transfer infrastructure tests.

Covers the four v10 seams owned by the infra track:
  * per-survey NormStats (survey-keyed z-scoring, pooled fallback, roundtrip)
  * --surveys corpus/train filtering (encoder build_corpus, classifier collect)
  * association-aware exclusions + association-grouped splitting
  * the seq-train-id diversion guard in build_split_manifest
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from debass_meta.features.sequence_dataset import (  # noqa: E402
    _SURVEY_DIM,
    _ZSCORE_DIMS,
    SEQ_CONTINUOUS_DIM,
    NormStats,
    sequence_arrays,
    sequence_survey,
)
from debass_meta.models.splitters import group_train_cal_test_split  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _det(mjd, mag, band="g", magerr=0.1, snr=20.0, survey="ZTF"):
    return {"mjd": mjd, "mag": mag, "band": band, "magerr": magerr, "snr": snr,
            "is_positive": True, "survey": survey,
            "flux": None, "fluxerr": None, "quality": 0.9}


def _lc(n=6, seed=0, survey="ZTF"):
    rng = np.random.default_rng(seed)
    mjd, mag, dets = 60000.0, 19.5, []
    for k in range(n):
        mjd += float(rng.uniform(0.5, 2.5))
        mag += float(rng.normal(-0.1, 0.08))
        dets.append(_det(mjd, mag, band="g" if k % 2 else "r", survey=survey))
    return dets


def _arrays(n_seqs, seed0, survey):
    return [sequence_arrays(_lc(6, seed=seed0 + i, survey=survey))[0]
            for i in range(n_seqs)]


# ── per-survey NormStats ─────────────────────────────────────────────────


def test_sequence_survey_helper():
    assert sequence_survey(sequence_arrays(_lc(4, survey="ZTF"))[0]) == "ztf"
    assert sequence_survey(sequence_arrays(_lc(4, survey="LSST"))[0]) == "lsst"


def test_per_survey_norm_stats_fit_and_apply():
    ztf = _arrays(4, 0, "ZTF")
    lsst = _arrays(4, 100, "LSST")
    # Shift the LSST distribution so survey stats are visibly different.
    for a in lsst:
        a[:, 0] += 5.0
    stats = NormStats.fit(ztf + lsst, per_survey=True, min_survey_sequences=2)
    assert set(stats.per_survey) == {"ztf", "lsst"}
    assert stats.per_survey["lsst"]["mean"][0] > stats.per_survey["ztf"]["mean"][0]

    # Row-wise dispatch: LSST rows use the lsst stats, ZTF rows the ztf stats.
    z = stats.apply(ztf[0])
    l = stats.apply(lsst[0])
    m_z, s_z = stats.per_survey["ztf"]["mean"], stats.per_survey["ztf"]["std"]
    m_l, s_l = stats.per_survey["lsst"]["mean"], stats.per_survey["lsst"]["std"]
    np.testing.assert_allclose(z[:, 0], (ztf[0][:, 0] - m_z[0]) / s_z[0], rtol=1e-5)
    np.testing.assert_allclose(l[:, 0], (lsst[0][:, 0] - m_l[0]) / s_l[0], rtol=1e-5)
    # Flags stay raw.
    np.testing.assert_array_equal(l[:, _SURVEY_DIM], lsst[0][:, _SURVEY_DIM])


def test_per_survey_norm_stats_pooled_fallback_below_threshold():
    ztf = _arrays(6, 0, "ZTF")
    lsst = _arrays(2, 100, "LSST")
    stats = NormStats.fit(ztf + lsst, per_survey=True, min_survey_sequences=3)
    assert "ztf" in stats.per_survey and "lsst" not in stats.per_survey
    # LSST rows fall back to the POOLED stats.
    out = stats.apply(lsst[0])
    for d in _ZSCORE_DIMS:
        np.testing.assert_allclose(
            out[:, d], (lsst[0][:, d] - stats.mean[d]) / stats.std[d], rtol=1e-5)


def test_per_survey_norm_stats_json_roundtrip_and_legacy_payload():
    stats = NormStats.fit(_arrays(3, 0, "ZTF") + _arrays(3, 50, "LSST"),
                          per_survey=True, min_survey_sequences=2)
    payload = json.loads(json.dumps(stats.to_json()))
    stats2 = NormStats.from_json(payload)
    assert stats2.per_survey.keys() == stats.per_survey.keys()
    a = sequence_arrays(_lc(5, survey="LSST"))[0]
    np.testing.assert_allclose(stats.apply(a), stats2.apply(a), rtol=1e-6)

    # Pre-v10 payloads (no per_survey key) load and behave pooled.
    legacy = NormStats.from_json({"mean": [0.5] * SEQ_CONTINUOUS_DIM,
                                  "std": [2.0] * SEQ_CONTINUOUS_DIM})
    assert legacy.per_survey == {}
    out = legacy.apply(a)
    for d in _ZSCORE_DIMS:
        np.testing.assert_allclose(out[:, d], (a[:, d] - 0.5) / 2.0, rtol=1e-6)
    # And pooled fit without the flag emits no per_survey key at all.
    pooled = NormStats.fit(_arrays(3, 0, "ZTF"))
    assert "per_survey" not in pooled.to_json()


# ── encoder script: --surveys corpus filter, DP1 corpus, association excl ─


def _write_json_lc(lc_dir: Path, stem: str, survey: str, n=6, seed=0):
    lc_dir.mkdir(parents=True, exist_ok=True)
    (lc_dir / f"{stem}.json").write_text(json.dumps(_lc(n, seed=seed, survey=survey)))


def _write_dp1_parquet(dp1_dir: Path, stem: str, n=6):
    dp1_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    mjd = 61000.0
    for k in range(n):
        mjd += 1.5
        rows.append({"midpointMjdTai": mjd, "band": "r" if k % 2 else "i",
                     "psfFlux": 3000.0 + 100.0 * k, "psfFluxErr": 50.0,
                     "snr": 25.0, "reliability": 0.95})
    pd.DataFrame(rows).to_parquet(dp1_dir / f"{stem}.parquet", index=False)


def test_build_corpus_survey_filter_and_dp1(tmp_path: Path):
    mod = _load_script("train_seq_encoder.py", "tse_v10_infra")
    lc_dir = tmp_path / "lc"
    _write_json_lc(lc_dir, "ZTFaaa", "ZTF", seed=1)
    _write_json_lc(lc_dir, "ZTFbbb", "ZTF", seed=2)
    _write_json_lc(lc_dir, "3100000001", "LSST", seed=3)
    _write_json_lc(lc_dir, "ZTFexcl", "ZTF", seed=4)
    dp1_dir = tmp_path / "dp1"
    _write_dp1_parquet(dp1_dir, "3200000001")
    _write_dp1_parquet(dp1_dir, "3200000002")  # excluded below

    exclude = {"ZTFexcl", "3200000002"}
    ids_both, seqs = mod.build_corpus(lc_dir, exclude, max_len=20,
                                      surveys="both", dp1_dir=dp1_dir)
    assert set(ids_both) == {"ZTFaaa", "ZTFbbb", "3100000001", "3200000001"}
    dp1_seq = seqs[ids_both.index("3200000001")]
    assert sequence_survey(dp1_seq[0]) == "lsst"  # DP1 normalized as LSST

    ids_ztf, _ = mod.build_corpus(lc_dir, exclude, max_len=20,
                                  surveys="ztf", dp1_dir=dp1_dir)
    assert set(ids_ztf) == {"ZTFaaa", "ZTFbbb"}  # DP1 skipped entirely for ztf

    ids_lsst, _ = mod.build_corpus(lc_dir, exclude, max_len=20,
                                   surveys="lsst", dp1_dir=dp1_dir)
    assert set(ids_lsst) == {"3100000001", "3200000001"}


def test_load_split_exclusions_association_counterparts(tmp_path: Path):
    mod = _load_script("train_seq_encoder.py", "tse_v10_infra")
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "train_ids": ["ZTFtrain", "3100000009"],
        "cal_ids": ["ZTFcal", "3100000001"],
        "test_ids": ["ZTFtest"],
    }))
    assoc = tmp_path / "assoc.csv"
    assoc.write_text(
        "lsst_object_id,ztf_object_id,match_status,sep_arcsec\n"
        "3100000001,ZTFcounterA,matched,0.4\n"     # LSST cal → ZTF counterpart excluded
        "3100000002,ZTFtest,matched,0.6\n"          # ZTF test → LSST counterpart excluded
        "3100000003,ZTFfar,matched,3.5\n"           # > 2 arcsec: rejected upstream
        "3100000004,ZTFunmatched,nomatch,0.2\n"     # not matched: ignored
        "3100000009,ZTFtrainbuddy,matched,0.1\n"    # train-side: NOT excluded
    )
    excl = mod.load_split_exclusions(manifest, association_csv=assoc)
    assert {"ZTFcal", "3100000001", "ZTFtest"} <= excl
    assert "ZTFcounterA" in excl and "3100000002" in excl
    assert "ZTFfar" not in excl and "ZTFunmatched" not in excl
    assert "ZTFtrainbuddy" not in excl and "3100000009" not in excl

    # Missing association CSV → plain cal/test exclusions (no crash).
    excl2 = mod.load_split_exclusions(manifest, association_csv=tmp_path / "nope.csv")
    assert excl2 == {"ZTFcal", "3100000001", "ZTFtest"}


# ── classifier script: --surveys train filter + --init-from guard ───────


def test_classifier_collect_survey_filter(tmp_path: Path):
    mod = _load_script("train_seq_classifier.py", "tsc_v10_infra")
    lc_dir = tmp_path / "lc"
    _write_json_lc(lc_dir, "ZTFaaa", "ZTF", seed=1)
    _write_json_lc(lc_dir, "3100000001", "LSST", seed=2)
    labels = {
        "ZTFaaa": mod.RowLabel(ternary="snia", quality="spectroscopic"),
        "3100000001": mod.RowLabel(ternary="other", quality="weak", fine="non_sn"),
    }
    ids = {"ZTFaaa", "3100000001"}
    both = mod.collect(labels, ids, lc_dir=lc_dir, max_len=20)
    assert set(both.oids) == ids
    ztf = mod.collect(labels, ids, lc_dir=lc_dir, max_len=20, surveys="ztf")
    assert ztf.oids == ["ZTFaaa"]
    lsst = mod.collect(labels, ids, lc_dir=lc_dir, max_len=20, surveys="lsst")
    assert lsst.oids == ["3100000001"]
    assert lsst.is_lsst.all() and not ztf.is_lsst.any()


def _tiny_seqset(mod, n=6):
    s = mod.SeqSet()
    for i in range(n):
        cont, bands = sequence_arrays(_lc(5, seed=i))
        s.oids.append(f"o{i}")
        s.seqs.append((cont, bands))
    s.y3 = np.zeros(n, dtype=np.int64)
    s.y4 = np.zeros(n, dtype=np.int64)
    s.masks = np.zeros((n, 4), dtype=bool)
    s.masks[:, 0] = True
    s.quals = np.ones(n)
    s.is_lsst = np.zeros(n, dtype=bool)
    return s


def _fit_args(mod, **over):
    import argparse
    base = dict(lr=1e-3, encoder_lr=1e-4, weight_decay=0.0, epochs=0,
                freeze_epochs=0, batch=4, patience=2, init_from=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_init_from_warm_starts_classifier_weights(tmp_path: Path):
    from debass_meta.models.seq_classifier import SeqClassifier, SeqClassifierArtifact

    mod = _load_script("train_seq_classifier.py", "tsc_v10_infra")
    torch.manual_seed(11)
    src = SeqClassifier()
    art = SeqClassifierArtifact(classifier=src, norm_stats=NormStats(),
                                temperature=1.0, meta={})
    art.save(tmp_path / "warm")

    tr = _tiny_seqset(mod)
    idx = np.arange(len(tr))
    # epochs=0: fit_classifier returns the INITIAL weights → must equal src's.
    clf, _, _ = mod.fit_classifier(
        tr, np.ones(len(tr)), idx[:4], idx[4:],
        args=_fit_args(mod, init_from=str(tmp_path / "warm")),
        stats=NormStats(), device=torch.device("cpu"),
        encoder_dir=tmp_path / "no_encoder", seed=0, tag="t")
    for k, v in src.state_dict().items():
        torch.testing.assert_close(clf.state_dict()[k].cpu(), v, rtol=0, atol=0)


def test_init_from_rejects_legacy_ternary_artifact(tmp_path: Path):
    from debass_meta.models.seq_classifier import (
        TERNARY_CLASSES,
        SeqClassifier,
        SeqClassifierArtifact,
    )

    mod = _load_script("train_seq_classifier.py", "tsc_v10_infra")
    art = SeqClassifierArtifact(
        classifier=SeqClassifier(classes=TERNARY_CLASSES),
        norm_stats=NormStats(), temperature=1.0, meta={})
    art.save(tmp_path / "legacy3")
    tr = _tiny_seqset(mod)
    idx = np.arange(len(tr))
    with pytest.raises(SystemExit):
        mod.fit_classifier(
            tr, np.ones(len(tr)), idx[:4], idx[4:],
            args=_fit_args(mod, init_from=str(tmp_path / "legacy3")),
            stats=NormStats(), device=torch.device("cpu"),
            encoder_dir=tmp_path / "no_encoder", seed=0, tag="t")


# ── association-grouped splitting ────────────────────────────────────────


def _legacy_group_split(object_to_label, *, train_frac=0.6, cal_frac=0.2, seed=42):
    """The pre-v10 implementation, verbatim — the parity oracle."""
    rng = np.random.default_rng(seed)
    by_label, unlabeled = {}, []
    for object_id, label in object_to_label.items():
        if label is None:
            unlabeled.append(str(object_id))
            continue
        by_label.setdefault(str(label), []).append(str(object_id))
    train_ids, cal_ids, test_ids = set(), set(), set()

    def _split_ids(ids):
        ids = ids.copy()
        rng.shuffle(ids)
        n = len(ids)
        if n <= 2:
            if n == 1:
                return ids, [], []
            return [ids[0]], [ids[1]], []
        n_train = max(1, int(round(n * train_frac)))
        n_cal = max(1, int(round(n * cal_frac)))
        if n_train + n_cal >= n:
            overflow = n_train + n_cal - (n - 1)
            if overflow > 0:
                n_train = max(1, n_train - overflow)
        n_test = n - n_train - n_cal
        if n_test < 1:
            n_cal = max(0, n_cal - 1)
            n_test = n - n_train - n_cal
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = n - n_train - n_cal
        return ids[:n_train], ids[n_train:n_train + n_cal], ids[n_train + n_cal:]

    for ids in by_label.values():
        a, b, c = _split_ids(ids)
        train_ids.update(a); cal_ids.update(b); test_ids.update(c)
    if unlabeled:
        a, b, c = _split_ids(unlabeled)
        train_ids.update(a); cal_ids.update(b); test_ids.update(c)
    return train_ids, cal_ids, test_ids


def test_group_split_exact_parity_without_associations():
    rng = np.random.default_rng(0)
    classes = ["snia", "nonIa_snlike", "other", None]
    for seed in (1, 7, 42):
        labels = {f"obj{i:04d}": classes[int(rng.integers(0, 4))] for i in range(211)}
        legacy = _legacy_group_split(labels, seed=seed)
        for assoc in (None, {}):
            split = group_train_cal_test_split(labels, seed=seed, association_map=assoc)
            assert (split.train_ids, split.cal_ids, split.test_ids) == legacy


def test_group_split_association_counterparts_stay_together():
    labels = {f"obj{i:03d}": "snia" for i in range(60)}
    labels.update({f"lsst{i:03d}": "snia" for i in range(20)})
    assoc = {f"lsst{i:03d}": f"obj{i:03d}" for i in range(20)}
    split = group_train_cal_test_split(labels, seed=42, association_map=assoc)
    buckets = {"train": split.train_ids, "cal": split.cal_ids, "test": split.test_ids}
    assert set().union(*buckets.values()) == set(labels)
    for i in range(20):
        homes = {name for name, ids in buckets.items()
                 if f"lsst{i:03d}" in ids or f"obj{i:03d}" in ids}
        assert len(homes) == 1, f"pair {i} straddles {homes}"


def test_group_split_transitive_association_through_absent_counterpart():
    # Two LSST ids sharing one ZTF counterpart that is NOT in the split
    # population still form one cluster (same physical photometry).
    labels = {f"pad{i:02d}": "other" for i in range(30)}
    labels.update({"lsstA": "other", "lsstB": "other"})
    assoc = {"lsstA": "ZTFshared", "lsstB": "ZTFshared"}
    for seed in range(5):
        split = group_train_cal_test_split(labels, seed=seed, association_map=assoc)
        for ids in (split.train_ids, split.cal_ids, split.test_ids):
            assert not ({"lsstA", "lsstB"} & ids) or {"lsstA", "lsstB"} <= ids


# ── seq-train diversion guard (build_split_manifest) ─────────────────────


@pytest.fixture(scope="module")
def bsf():
    return _load_script("build_snapshots_fusion.py", "bsf_v10_infra")


def _trust_metadata(tmp_path: Path) -> Path:
    md = {
        "train_ids": [f"OT{i}" for i in range(6)],
        "cal_ids": ["OC0", "OC1"],
        "test_ids": ["OX0", "OX1"],
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(md))
    return p


def _manifest_inputs(tmp_path: Path, n_new=40):
    md_path = _trust_metadata(tmp_path)
    orig = [f"OT{i}" for i in range(6)] + ["OC0", "OC1", "OX0", "OX1"]
    new = [f"NEW{i:03d}" for i in range(n_new)]
    objs = orig + new
    truth = {o: {"final_class_ternary": "snia"} for o in objs}
    return objs, truth, md_path


def test_seq_train_guard_diverts_new_cal_to_train(tmp_path: Path, bsf):
    objs, truth, md_path = _manifest_inputs(tmp_path)
    base = bsf.build_split_manifest(objs, truth, md_path, seed=42)
    new_cal = [o for o in base["cal_ids"] if o.startswith("NEW")]
    assert new_cal, "fixture must place some new objects in cal"
    victim = new_cal[0]

    guarded = bsf.build_split_manifest(
        objs, truth, md_path, seed=42, seq_train_ids={victim, "NEWNOTINSPLIT"})
    assert victim in guarded["train_ids"]
    assert victim not in guarded["cal_ids"]
    assert guarded["n_seq_train_diverted_cal_to_train"] == 1
    assert guarded["seq_train_guard"] is True
    # Everything else identical to the unguarded manifest.
    assert set(guarded["cal_ids"]) == set(base["cal_ids"]) - {victim}
    assert set(guarded["train_ids"]) == set(base["train_ids"]) | {victim}
    assert guarded["test_ids"] == base["test_ids"]


def test_seq_train_guard_hard_fails_on_locked_test(tmp_path: Path, bsf):
    objs, truth, md_path = _manifest_inputs(tmp_path)
    with pytest.raises(AssertionError, match="LOCKED TEST"):
        bsf.build_split_manifest(objs, truth, md_path, seed=42,
                                 seq_train_ids={"OX0"})


def test_seq_train_guard_warns_but_keeps_locked_cal(tmp_path: Path, bsf, capsys):
    objs, truth, md_path = _manifest_inputs(tmp_path)
    manifest = bsf.build_split_manifest(objs, truth, md_path, seed=42,
                                        seq_train_ids={"OC0"})
    assert "OC0" in manifest["cal_ids"]  # verbatim preservation wins
    assert manifest["n_seq_train_in_locked_cal"] == 1
    assert "LOCKED (v6e2) cal" in capsys.readouterr().out


def test_manifest_association_grouping_of_new_objects(tmp_path: Path, bsf):
    objs, truth, md_path = _manifest_inputs(tmp_path)
    assoc = {"NEW000": "NEW001", "NEW002": "NEW003"}
    manifest = bsf.build_split_manifest(objs, truth, md_path, seed=42,
                                        association_map=assoc)
    assert manifest["association_grouped_split"] is True
    for a, b in assoc.items():
        in_train = {a, b} <= set(manifest["train_ids"])
        in_cal = {a, b} <= set(manifest["cal_ids"])
        assert in_train or in_cal, f"pair ({a},{b}) split across train/cal"


def test_manifest_locked_counterparts_never_split_independently(tmp_path: Path, bsf):
    """A NEW object whose association counterpart sits in the LOCKED split
    must not be assigned independently: test counterpart → quarantined,
    cal counterpart → forced cal, train counterpart → forced train."""
    objs, truth, md_path = _manifest_inputs(tmp_path)
    assoc = {"NEW000": "OX0", "NEW001": "OC0", "NEW002": "OT0"}
    manifest = bsf.build_split_manifest(objs, truth, md_path, seed=42,
                                        association_map=assoc)

    train, cal = set(manifest["train_ids"]), set(manifest["cal_ids"])
    test, quar = set(manifest["test_ids"]), set(manifest["quarantined_ids"])

    # locked-TEST counterpart: excluded from EVERY split (train and cal
    # would both leak locked-test photometry)
    assert "NEW000" in quar
    assert "NEW000" not in train | cal | test
    assert manifest["n_new_quarantined_test_counterparts"] == 1
    assert manifest["counts"]["quarantined"] == 1

    # locked-CAL counterpart: co-located with the counterpart, no straddle
    assert "NEW001" in cal and "NEW001" not in train
    assert manifest["n_new_forced_cal_by_association"] == 1

    # locked-TRAIN counterpart: forced into train
    assert "NEW002" in train and "NEW002" not in cal
    assert manifest["n_new_forced_train_by_association"] == 1

    # coverage: every snapshot object in exactly one of the four buckets
    assert train | cal | test | quar == set(objs)
    # locked assignments stay verbatim
    assert "OX0" in test and "OC0" in cal and "OT0" in train


def test_manifest_seq_train_counterpart_of_locked_test_hard_fails(tmp_path: Path, bsf):
    """Training on the counterpart of a locked-test object is training on
    locked-test photometry — the extended assert must fire."""
    objs, truth, md_path = _manifest_inputs(tmp_path)
    assoc = {"NEW000": "OX0"}
    with pytest.raises(AssertionError, match="LOCKED TEST"):
        bsf.build_split_manifest(objs, truth, md_path, seed=42,
                                 association_map=assoc,
                                 seq_train_ids={"NEW000"})


def test_manifest_seq_train_guard_diverts_whole_association_cluster(tmp_path: Path, bsf):
    """cal→train diversion moves WHOLE clusters: diverting only the
    seq-train member would leave its counterpart's (identical) photometry
    in cal — re-creating the straddle the association grouping prevents."""
    objs, truth, md_path = _manifest_inputs(tmp_path)
    assoc = {"NEW000": "NEW001", "NEW002": "NEW003", "NEW004": "NEW005"}
    cal_pairs, seed = [], None
    for seed in range(20):  # find a seed that places an associated pair in cal
        base = bsf.build_split_manifest(objs, truth, md_path, seed=seed,
                                        association_map=assoc)
        cal_pairs = [(a, b) for a, b in assoc.items()
                     if {a, b} <= set(base["cal_ids"])]
        if cal_pairs:
            break
    assert cal_pairs, "no seed placed an associated pair in cal"
    victim, partner = cal_pairs[0]

    guarded = bsf.build_split_manifest(
        objs, truth, md_path, seed=seed, association_map=assoc,
        seq_train_ids={victim})
    assert {victim, partner} <= set(guarded["train_ids"])
    assert not {victim, partner} & set(guarded["cal_ids"])
    assert guarded["n_seq_train_diverted_cal_to_train"] == 2

    # Symmetric case: the seq-train id is only the COUNTERPART (e.g. the
    # trained LSST id is absent from the rebuilt population) — the cal
    # member must still be detected and diverted.
    ghost_assoc = dict(assoc)
    ghost_assoc["GHOST_LSST"] = victim
    guarded2 = bsf.build_split_manifest(
        objs, truth, md_path, seed=seed, association_map=ghost_assoc,
        seq_train_ids={"GHOST_LSST"})
    assert {victim, partner} <= set(guarded2["train_ids"])
    assert not {victim, partner} & set(guarded2["cal_ids"])


def test_load_seq_train_ids_formats(tmp_path: Path, bsf):
    fm = tmp_path / "fold_map.json"
    fm.write_text(json.dumps({"n_folds": 5, "assignments": {"A": 0, "B": 3}}))
    assert bsf.load_seq_train_ids(fm) == {"A", "B"}

    sm = tmp_path / "split.json"
    sm.write_text(json.dumps({"train_ids": ["C", "D"], "cal_ids": ["E"]}))
    assert bsf.load_seq_train_ids(sm) == {"C", "D"}

    lst = tmp_path / "ids.json"
    lst.write_text(json.dumps(["F", "G"]))
    assert bsf.load_seq_train_ids(lst) == {"F", "G"}

    pq = tmp_path / "ids.parquet"
    pd.DataFrame({"object_id": ["H", "I"]}).to_parquet(pq, index=False)
    assert bsf.load_seq_train_ids(pq) == {"H", "I"}

    with pytest.raises(SystemExit):
        bsf.load_seq_train_ids(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"something": 1}))
    with pytest.raises(SystemExit):
        bsf.load_seq_train_ids(bad)
