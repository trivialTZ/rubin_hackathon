"""v13g ``--lsst-all-negative-fallback``: keep Rubin objects whose alert lightcurve
has no positive detection (SN light in the template), as ZTF already does.

Default off must be byte-identical to the previous builder; on, only the new
LSST fallback rows appear (flag 1, n_pos_det 0) and every other row is
unchanged; such a row is scored end to end by the head and the scorer.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_gold_positive_only as gpo  # noqa: E402
import test_snapshots_fusion_g5 as g5  # noqa: E402

from debass_meta.features.lightcurve import (  # noqa: E402
    extract_features_at_each_epoch,
    truncated_detection_windows,
)

snapmod = g5.snapmod
NEG_OID = gpo._LSST_ALLNEG
POS_OID = gpo._LSST_SHORT


def _negatives():
    return [gpo._lsst(100.0, -1000.0), gpo._lsst(101.0, -500.0), gpo._lsst(102.0, -800.0)]


def _build(tmp_path: Path, **overrides):
    paths = g5._write_repo(tmp_path)
    for oid, dets in ((NEG_OID, _negatives()),
                      (POS_OID, [gpo._lsst(100.0, 1000.0), gpo._lsst(101.0, -500.0), gpo._lsst(102.0, 1200.0)])):
        gpo._write(paths["lc_dir"], oid, dets)
    truth_path = paths["truth_dir"] / "object_truth.parquet"
    truth = pd.read_parquet(truth_path)
    extra = pd.DataFrame([
        {"object_id": oid, "final_class_ternary": "nonIa_snlike", "follow_proxy": 0,
         "label_source": "tns_spectroscopic", "label_quality": "spectroscopic",
         "tns_type": "SN II"} for oid in (NEG_OID, POS_OID)])
    pd.concat([truth, extra], ignore_index=True).to_parquet(truth_path, index=False)
    out = paths["gold_dir"] / "snapshots_fusion.parquet"
    manifest = paths["gold_dir"] / "split_fusion.json"
    kwargs = dict(
        lc_dir=paths["lc_dir"], silver_dir=paths["silver_dir"], truth_path=truth_path,
        bts_path=paths["truth_dir"] / "ztf_bts.parquet", labels_path=paths["labels"],
        trust_metadata_path=paths["trust_metadata"], output_path=out,
        split_manifest_path=manifest, n_jobs=1, seed=42)
    kwargs.update(overrides)
    snapmod.build_fusion_snapshots(**kwargs)
    return out, pd.read_parquet(out)


def test_helper_default_and_opt_in():
    dets = _negatives()
    assert truncated_detection_windows(dets, survey="LSST", max_n_det=20) == []
    assert extract_features_at_each_epoch(dets, survey="LSST") == []
    w = truncated_detection_windows(dets, survey="LSST", max_n_det=20, lsst_all_negative_fallback=True)
    assert [len(p) for p, _ in w] == [1, 2, 3]
    # opt-in changes nothing for an LSST object with a positive detection, nor for ZTF
    mixed = [gpo._lsst(100.0, 1000.0), gpo._lsst(101.0, -500.0), gpo._lsst(102.0, 1200.0)]
    a = truncated_detection_windows(mixed, survey="LSST", max_n_det=20)
    b = truncated_detection_windows(mixed, survey="LSST", max_n_det=20, lsst_all_negative_fallback=True)
    assert [[d["mjd"] for d in p] for p, _ in a] == [[d["mjd"] for d in p] for p, _ in b]
    ztf = g5._ztf_lc(3)
    assert len(truncated_detection_windows(ztf, survey="ZTF")) == len(
        truncated_detection_windows(ztf, survey="ZTF", lsst_all_negative_fallback=True)) == 3


def test_worker_rows_flagged(tmp_path):
    lc = tmp_path / "lc"
    lc.mkdir()
    gpo._write(lc, NEG_OID, _negatives())
    assert snapmod._extract_object_rows(NEG_OID, lc_dir_str=str(lc), max_n_det=20)[2] == []
    oid, src, rows = snapmod._extract_object_rows(
        NEG_OID, lc_dir_str=str(lc), max_n_det=20, lsst_all_negative_fallback=True)
    assert src["all_negative_fallback"] is True and len(rows) == 3
    for r in rows:
        assert r["lc_fallback_all_negative"] == 1.0 and r["n_pos_det"] == 0.0 and r["survey_is_lsst"] == 1.0
        assert r["n_det_neg"] == r["n_det"] and r["frac_neg"] == 1.0


def test_default_off_is_byte_identical(tmp_path):
    out_a, a = _build(tmp_path / "a")
    out_b, b = _build(tmp_path / "b", lsst_all_negative_fallback=False)
    assert NEG_OID not in set(a["object_id"].astype(str)) and POS_OID in set(a["object_id"].astype(str))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert out_a.read_bytes() == out_b.read_bytes()


def test_opt_in_adds_only_the_fallback_rows(tmp_path):
    _, off = _build(tmp_path / "off")
    _, on = _build(tmp_path / "on", lsst_all_negative_fallback=True)
    new = on[on["object_id"].astype(str) == NEG_OID]
    assert len(new) == 3 and (new["lc_fallback_all_negative"] == 1.0).all()
    assert (new["survey"] == "LSST").all() and (new["n_pos_det"] == 0).all()
    assert new["target_class"].eq("nonIa_snlike").all() and new["label_quality"].eq("spectroscopic").all()
    rest = on[on["object_id"].astype(str) != NEG_OID].reset_index(drop=True)
    pd.testing.assert_frame_equal(rest, off.reset_index(drop=True), check_exact=True)
    # every other LSST / ZTF row keeps flag 0 unless it is a ZTF all-negative object
    assert (on.loc[on["object_id"].astype(str) == POS_OID, "lc_fallback_all_negative"] == 0.0).all()


def test_cli_flag_default_off():
    src = Path(snapmod.__file__).read_text()
    assert '"--lsst-all-negative-fallback", action="store_true"' in src


# ---------------------------------------------- scored end to end

def test_fallback_row_scores_end_to_end(tmp_path, monkeypatch):
    """Head -> anchored blend -> conformal fallback -> scorer on a gold-shaped
    LSST fallback row (mags NaN, n_pos_det 0, frac_neg 1, no expert outputs)."""
    from test_hierarchical_followup import _build_dataset

    from debass_meta.models.anchor_blend import BlendSpec
    from debass_meta.models.hierarchical_followup import HierarchicalFollowup

    df, tr, ca, te = _build_dataset()
    head = HierarchicalFollowup(survey_cal_min=40, n_jobs=2, seed=42).fit(df, tr, ca, te)
    row = df[df["survey"] == "LSST"].iloc[[0]].copy()
    row["object_id"] = NEG_OID
    for c in row.columns:
        if c.startswith(("lcf_", "mag_", "proj__", "q__", "q_prior__")):
            row[c] = np.nan
        if c.startswith("avail__"):
            row[c] = 0.0
    row["lc_fallback_all_negative"] = 1.0
    p = head.predict_proba(row)
    assert p.shape == (1, 3) and np.isfinite(p).all() and np.isclose(p.sum(), 1.0)
    assert np.isfinite(head.predict_proba_raw(row)).all()

    score_v11 = importlib.import_module("scripts.score_fusion_v11")
    monkeypatch.setattr(score_v11, "load_followup_artifact", lambda d: head)
    gold_out, gold = _build(tmp_path / "g", lsst_all_negative_fallback=True, skip_experts=True)
    fb = gold[gold["object_id"].astype(str) == NEG_OID].copy()
    assert len(fb) == 3
    head_cols = set(head.head1_feature_cols) | set(head.head2_feature_cols)
    for c in head_cols - set(fb.columns):
        fb[c] = np.nan                                  # synthetic head columns absent from the real gold
    fb["n_det"] = fb["n_det"].astype(int)
    path = tmp_path / "fb.parquet"
    fb.to_parquet(path, index=False)
    blend = tmp_path / "blend"
    BlendSpec().save(blend)
    out = tmp_path / "pred.parquet"
    score_v11.main(["--snapshots", str(path), "--blend-dir", str(blend), "--followup-dir", str(tmp_path / "x"),
                    "--trust-dir", str(tmp_path / "notrust"), "--conformal", str(tmp_path / "noconf.pkl"),
                    "--out", str(out), "--no-priority"])
    pred = pd.read_parquet(out)
    assert len(pred) == 3 and pred["object_id"].astype(str).eq(NEG_OID).all()
    probs = pred[["p_snia", "p_nonia", "p_other"]].to_numpy(float)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1.0)
    assert set(pred["serving_regime"]) == {"none"}      # no expert outputs on this row: flagged, not hidden
