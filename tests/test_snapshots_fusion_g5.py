"""Tests for G5: scripts/build_snapshots_fusion.py + build_helpfulness_fusion.py.

All synthetic / self-contained except the parity test, which skips unless the
real data files exist.  Leakage tests: extending a lightcurve with future
detections must NOT change features at fixed n_det.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from debass_meta.features.lightcurve import FEATURE_NAMES, extract_features_at_each_epoch


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_g5_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


snapmod = _load_script("build_snapshots_fusion")
helpmod = _load_script("build_helpfulness_fusion")


# ------------------------------------------------------------------ #
# Synthetic fixtures                                                  #
# ------------------------------------------------------------------ #

def _ztf_detection(mjd: float, mag: float, fid: int = 1, rb: float = 0.9) -> dict:
    return {"mjd": mjd, "magpsf": mag, "sigmapsf": 0.1, "fid": fid,
            "isdiffpos": "t", "rb": rb}


def _ztf_lc(n: int, start: float = 60000.0) -> list[dict]:
    return [
        _ztf_detection(start + i, 20.0 - 0.2 * i, fid=1 + (i % 2))
        for i in range(n)
    ]


def _broker_event_row(object_id: str, field: str, score: float, jd: float) -> dict:
    return {
        "object_id": object_id, "broker": "fink", "classifier": "snn",
        "classifier_version": None, "expert_key": "fink/snn", "field": field,
        "class_name": None, "semantic_type": "probability",
        "event_scope": "alert", "event_time_jd": jd, "alert_id": None,
        "n_det": None, "raw_label_or_score": str(score),
        "canonical_projection": score, "ranking": 1, "availability": True,
        "fixture_used": False, "temporal_exactness": "exact_alert",
        "payload_hash": None, "provenance_json": "{}", "survey": "ZTF",
        "source_endpoint": None, "request_params_json": None,
        "status_code": None, "query_time_unix": None,
    }


def _write_repo(tmp_path: Path) -> dict[str, Path]:
    """Synthetic mini-repo: 4 LCs, truth, BTS, labels, silver, trust metadata."""
    lc_dir = tmp_path / "lightcurves"
    silver_dir = tmp_path / "silver"
    truth_dir = tmp_path / "truth"
    gold_dir = tmp_path / "gold"
    models_dir = tmp_path / "models"
    for d in (lc_dir, silver_dir, truth_dir, gold_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ZTFAAA: spec snia, locked test.  ZTFBBB: weak 'other' alerce_self_label,
    # locked train.  ZTFCCC: BTS-only (new object).  ZTFDDD: no truth → excluded.
    with open(lc_dir / "ZTFAAA.json", "w") as fh:
        json.dump(_ztf_lc(4), fh)
    with open(lc_dir / "ZTFBBB.json", "w") as fh:
        json.dump(_ztf_lc(3), fh)
    with open(lc_dir / "ZTFCCC.json", "w") as fh:
        json.dump(_ztf_lc(5), fh)
    with open(lc_dir / "ZTFDDD.json", "w") as fh:
        json.dump(_ztf_lc(2), fh)

    pd.DataFrame([
        {"object_id": "ZTFAAA", "final_class_ternary": "snia", "follow_proxy": 1,
         "label_source": "tns_spectroscopic", "label_quality": "spectroscopic"},
        {"object_id": "ZTFBBB", "final_class_ternary": "other", "follow_proxy": 0,
         "label_source": "alerce_self_label", "label_quality": "weak"},
    ]).to_parquet(truth_dir / "object_truth.parquet", index=False)

    pd.DataFrame([
        {"object_id": "ZTFCCC", "bts_type": "Ia", "ternary": "snia",
         "tns_name": "SN2026xx", "label_source": "ztf_bts",
         "label_quality": "spectroscopic"},
    ]).to_parquet(truth_dir / "ztf_bts.parquet", index=False)

    with open(tmp_path / "labels.csv", "w") as fh:
        fh.write("object_id,label,ra,dec,survey,n_det,discovery_broker\n")
        fh.write("ZTFAAA,snia,10.0,10.0,ZTF,4,alerce\n")
        fh.write("ZTFBBB,other,11.0,11.0,ZTF,3,alerce\n")
        fh.write("123456789,snia,12.0,12.0,LSST,3,alerce\n")  # LSST id: filtered out

    # fink/snn events at the JD of ZTFAAA's 2nd detection → available n_det>=2.
    jd2 = 60001.0 + 2400000.5
    pd.DataFrame([
        _broker_event_row("ZTFAAA", "snn_snia_vs_nonia", 0.9, jd2),
        _broker_event_row("ZTFAAA", "snn_sn_vs_all", 0.8, jd2),
    ]).to_parquet(silver_dir / "broker_events.parquet", index=False)

    with open(models_dir / "trust_metadata.json", "w") as fh:
        json.dump({"experts": {}, "train_ids": ["ZTFBBB"], "cal_ids": [],
                   "test_ids": ["ZTFAAA"]}, fh)

    return {
        "lc_dir": lc_dir, "silver_dir": silver_dir, "truth_dir": truth_dir,
        "gold_dir": gold_dir, "labels": tmp_path / "labels.csv",
        "trust_metadata": models_dir / "trust_metadata.json",
    }


def _build(tmp_path: Path, **overrides) -> tuple[pd.DataFrame, dict]:
    paths = _write_repo(tmp_path)
    out = paths["gold_dir"] / "snapshots_fusion.parquet"
    manifest = paths["gold_dir"] / "split_fusion.json"
    kwargs = dict(
        lc_dir=paths["lc_dir"], silver_dir=paths["silver_dir"],
        truth_path=paths["truth_dir"] / "object_truth.parquet",
        bts_path=paths["truth_dir"] / "ztf_bts.parquet",
        labels_path=paths["labels"],
        trust_metadata_path=paths["trust_metadata"],
        output_path=out, split_manifest_path=manifest,
        n_jobs=1, seed=42,
    )
    kwargs.update(overrides)
    snapmod.build_fusion_snapshots(**kwargs)
    with open(manifest) as fh:
        return pd.read_parquet(out), json.load(fh)


# ------------------------------------------------------------------ #
# BTS mapping parity                                                  #
# ------------------------------------------------------------------ #

def test_bts_mapping_known_types() -> None:
    assert snapmod.bts_type_to_ternary("Ia") == "snia"
    assert snapmod.bts_type_to_ternary("SN Ia") == "snia"
    assert snapmod.bts_type_to_ternary("Ia-91T") == "snia"
    assert snapmod.bts_type_to_ternary("II") == "nonIa_snlike"
    assert snapmod.bts_type_to_ternary("IIn") == "nonIa_snlike"
    assert snapmod.bts_type_to_ternary("SLSN-I") == "nonIa_snlike"
    # Pre-existing fetch_ztf_bts quirk (replicated, not fixed): non-SN types
    # fall through the "SN "-prefix fallback to nonIa_snlike.
    assert snapmod.bts_type_to_ternary("TDE") == "nonIa_snlike"
    assert snapmod.bts_type_to_ternary("-") == "nonIa_snlike"
    assert snapmod.bts_type_to_ternary(None) is None
    assert snapmod.bts_type_to_ternary("") is None
    assert snapmod.bts_type_to_ternary(float("nan")) is None


def test_bts_parity_check_passes_and_fails() -> None:
    good = pd.DataFrame([
        {"object_id": "a", "bts_type": "Ia", "ternary": "snia"},
        {"object_id": "b", "bts_type": "IIb", "ternary": "nonIa_snlike"},
    ])
    snapmod.check_bts_mapping_parity(good)  # must not raise

    bad = pd.DataFrame([{"object_id": "c", "bts_type": "Ia", "ternary": "other"}])
    with pytest.raises(AssertionError, match="parity FAILED"):
        snapmod.check_bts_mapping_parity(bad)


@pytest.mark.skipif(
    not (_ROOT / "data/truth/ztf_bts.parquet").exists(),
    reason="real ztf_bts.parquet not present",
)
def test_bts_parity_on_real_catalog() -> None:
    snapmod.check_bts_mapping_parity(
        pd.read_parquet(_ROOT / "data/truth/ztf_bts.parquet")
    )


# ------------------------------------------------------------------ #
# Truncation / no-leakage                                             #
# ------------------------------------------------------------------ #

def test_truncated_lists_match_base_extractor() -> None:
    lc = _ztf_lc(6)
    lc[2]["isdiffpos"] = "f"  # one negative detection — must be filtered
    epochs = extract_features_at_each_epoch(lc, max_n_det=20)
    truncs = snapmod._truncated_detection_lists(lc, max_n_det=20)
    assert len(epochs) == len(truncs) == 5
    for feats, trunc in zip(epochs, truncs):
        assert int(feats["n_det"]) == len(trunc)
        assert abs(float(feats["alert_mjd"]) - float(trunc[-1]["mjd"])) < 1e-9


def test_no_future_leakage_in_worker_rows(tmp_path: Path) -> None:
    """Appending future detections must not change rows at fixed n_det."""
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    with open(lc_dir / "ZTFSHORT.json", "w") as fh:
        json.dump(_ztf_lc(3), fh)
    with open(lc_dir / "ZTFLONG.json", "w") as fh:
        json.dump(_ztf_lc(8), fh)  # same first 3 detections + 5 future ones

    _, _, short_rows = snapmod._extract_object_rows(
        "ZTFSHORT", lc_dir_str=str(lc_dir), max_n_det=20)
    _, _, long_rows = snapmod._extract_object_rows(
        "ZTFLONG", lc_dir_str=str(lc_dir), max_n_det=20)
    assert len(short_rows) == 3 and len(long_rows) == 8
    for n in range(3):
        a, b = short_rows[n], long_rows[n]
        assert set(a) == set(b)
        for key in a:
            va, vb = a[key], b[key]
            if isinstance(va, float) and isinstance(vb, float):
                assert (math.isnan(va) and math.isnan(vb)) or va == pytest.approx(vb), key
            else:
                assert va == vb, key


# ------------------------------------------------------------------ #
# End-to-end snapshot build                                           #
# ------------------------------------------------------------------ #

def test_end_to_end_build_labels_experts_split(tmp_path: Path) -> None:
    df, manifest = _build(tmp_path)

    # Object list: labels ZTF ids + truthy stems; ZTFDDD (no truth) and the
    # LSST id (no lightcurve) excluded.
    assert set(df["object_id"]) == {"ZTFAAA", "ZTFBBB", "ZTFCCC"}
    assert len(df) == 4 + 3 + 5  # one row per detection epoch

    # Base features + schema columns present.
    for col in ["alert_jd", "survey", "lightcurve_source_object_id",
                "target_class", "label_source", "label_quality", *FEATURE_NAMES]:
        assert col in df.columns, col
    assert set(df["survey"]) == {"ZTF"}

    # Labels: truth-tier pass-through + BTS fallback.
    aaa = df[df["object_id"] == "ZTFAAA"]
    assert set(aaa["target_class"]) == {"snia"}
    assert set(aaa["label_quality"]) == {"spectroscopic"}
    bbb = df[df["object_id"] == "ZTFBBB"]
    assert set(bbb["target_class"]) == {"other"}
    assert set(bbb["label_source"]) == {"alerce_self_label"}  # provenance kept
    ccc = df[df["object_id"] == "ZTFCCC"]
    assert set(ccc["target_class"]) == {"snia"}
    assert set(ccc["label_source"]) == {"ztf_bts"}

    # Expert block: fink/snn event stamped at detection 2 → as-of availability.
    aaa = aaa.sort_values("n_det")
    assert aaa.iloc[0]["avail__fink__snn"] == 0.0
    assert aaa.iloc[1]["avail__fink__snn"] == 1.0
    assert aaa.iloc[3]["avail__fink__snn"] == 1.0
    # fink/snn projection: p_snia = 0.9*0.8, p_nonIa = 0.1*0.8, p_other = 0.2.
    assert aaa.iloc[1]["proj__fink__snn__p_snia"] == pytest.approx(0.72)
    assert aaa.iloc[1]["proj__fink__snn__p_other"] == pytest.approx(0.2)

    # Split manifest: locked assignments verbatim; new object never in test.
    assert manifest["test_ids"] == ["ZTFAAA"]
    assert "ZTFBBB" in manifest["train_ids"]
    assert "ZTFCCC" in manifest["train_ids"] + manifest["cal_ids"]
    assert "ZTFCCC" not in manifest["test_ids"]
    assert manifest["seed"] == 42
    assert manifest["n_new"] == 1

    # No trust-output columns at build time.
    assert not [c for c in df.columns if c.startswith("q__")]


def test_skip_experts_keeps_schema(tmp_path: Path) -> None:
    df, _ = _build(tmp_path, skip_experts=True, skip_traj=True)
    assert "avail__fink__snn" in df.columns
    assert set(df["avail__fink__snn"]) == {0.0}


def test_smoke_limit(tmp_path: Path) -> None:
    df, manifest = _build(tmp_path, limit=1, smoke=False, skip_experts=True)
    assert df["object_id"].nunique() == 1
    assert manifest["smoke"] is True  # limit runs are flagged as dev splits


# ------------------------------------------------------------------ #
# DP1 path                                                            #
# ------------------------------------------------------------------ #

def test_dp1_append_preserves_columns(tmp_path: Path) -> None:
    dp1_lc_dir = tmp_path / "dp1_lc"
    dp1_lc_dir.mkdir()
    mjds = [60632.1, 60633.2, 60634.3]
    pd.DataFrame({
        "diaObjectId": [111] * 3,
        "diaSourceId": [1, 2, 3],
        "band": ["g", "r", "i"],
        "midpointMjdTai": mjds,
        "psfFlux": [2000.0, 2500.0, 2200.0],
        "psfFluxErr": [100.0, 110.0, 105.0],
        "reliability": [0.9, 0.95, 0.92],
        "snr": [20.0, 22.7, 21.0],
        "ra": [10.0] * 3, "dec": [-5.0] * 3,
    }).to_parquet(dp1_lc_dir / "111.parquet", index=False)

    rows = []
    for n_det, mjd in enumerate(mjds, start=1):
        rows.append({"object_id": "111", "diaObjectId": 111, "n_det": n_det,
                     "alert_jd": mjd + 2400000.5, "survey": "LSST",
                     "is_published_sn": False, "simbad_main_type": None,
                     "is_gaia_known_star": False, "is_known_variable": False})
    rows.append({"object_id": "222", "diaObjectId": 222, "n_det": 1,
                 "alert_jd": 2460640.5, "survey": "LSST",
                 "is_published_sn": True, "simbad_main_type": "G",
                 "is_gaia_known_star": False, "is_known_variable": False})
    src = tmp_path / "dp1_snapshots.parquet"
    pd.DataFrame(rows).to_parquet(src, index=False)

    out = tmp_path / "dp1_fusion.parquet"
    snapmod.build_dp1_fusion(src, dp1_lc_dir, out)
    result = pd.read_parquet(out)

    assert len(result) == 4
    for col in pd.read_parquet(src).columns:  # all original cols preserved
        assert col in result.columns, col
    assert result[result["object_id"] == "222"]["is_published_sn"].iloc[0]

    if snapmod.EXT_FEATURE_NAMES:  # real EXT module present
        for name in snapmod.EXT_FEATURE_NAMES:
            assert name in result.columns, name
        no_lc = result[result["object_id"] == "222"]
        assert no_lc[list(snapmod.EXT_FEATURE_NAMES)].isna().all().all()


# ------------------------------------------------------------------ #
# Helpfulness builder                                                 #
# ------------------------------------------------------------------ #

def _synthetic_fusion_snapshot() -> pd.DataFrame:
    rows = []
    for oid, target in (("ZTFAAA", "snia"), ("ZTFBBB", "other")):
        for n_det in (1, 2):
            row = {
                "object_id": oid, "n_det": n_det,
                "alert_jd": 2460000.5 + n_det, "survey": "ZTF",
                "target_class": target,
                "target_follow_proxy": 1 if target == "snia" else 0,
                "label_source": "tns_spectroscopic",
                "label_quality": "spectroscopic",
            }
            for name in FEATURE_NAMES:
                row.setdefault(name, float(n_det))
            # fink/snn available everywhere; supernnova only on ZTFBBB.
            row.update({
                "avail__fink__snn": 1.0, "exact__fink__snn": 1.0,
                "temporal_exactness__fink__snn": "exact_alert",
                "prediction_type__fink__snn": "class_correctness",
                "mapped_pred_class__fink__snn": "snia",
                "proj__fink__snn__p_snia": 0.7,
                "proj__fink__snn__p_nonIa_snlike": 0.2,
                "proj__fink__snn__p_other": 0.1,
                "avail__supernnova": 1.0 if oid == "ZTFBBB" else 0.0,
                "exact__supernnova": 1.0 if oid == "ZTFBBB" else 0.0,
                "temporal_exactness__supernnova": "rerun_exact",
                "prediction_type__supernnova": "class_correctness",
                "mapped_pred_class__supernnova": "other",
                "proj__supernnova__p_snia": 0.1,
                "proj__supernnova__p_nonIa_snlike": 0.2,
                "proj__supernnova__p_other": 0.7,
                # per-expert trajectory block + cross-expert aggregate
                "traj__fink__snn__last": 0.6 + 0.1 * n_det,
                "traj__fink__snn__slope": 0.05,
                "traj_x__mean_last": 0.5,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_helpfulness_fusion_long_format(tmp_path: Path) -> None:
    snap_path = tmp_path / "snap.parquet"
    _synthetic_fusion_snapshot().to_parquet(snap_path, index=False)
    out_path = tmp_path / "help.parquet"
    helpmod.build_helpfulness_fusion(snap_path, out_path, chunk_objects=1)
    out = pd.read_parquet(out_path)

    # One row per (gold row × available expert): 4 fink/snn + 2 supernnova.
    assert len(out) == 6
    assert (out["expert_key"] == "fink/snn").sum() == 4
    assert (out["expert_key"] == "supernnova").sum() == 2

    # Targets: fink/snn predicts snia → correct on ZTFAAA only; supernnova
    # predicts other → correct on ZTFBBB.  is_sn = (target != other).
    snn = out[out["expert_key"] == "fink/snn"]
    assert set(snn[snn["object_id"] == "ZTFAAA"]["is_topclass_correct"]) == {1}
    assert set(snn[snn["object_id"] == "ZTFBBB"]["is_topclass_correct"]) == {0}
    assert set(snn[snn["object_id"] == "ZTFAAA"]["is_sn"]) == {1}
    assert set(snn[snn["object_id"] == "ZTFBBB"]["is_sn"]) == {0}
    local = out[out["expert_key"] == "supernnova"]
    assert set(local["is_topclass_correct"]) == {1}

    # Own-trajectory generic slots: filled for fink/snn, NaN for supernnova;
    # per-expert traj__ columns must NOT survive into the long table.
    assert snn["own_traj__last"].notna().all()
    assert snn["own_traj__slope"].notna().all()
    assert local["own_traj__last"].isna().all()
    assert not [c for c in out.columns if c.startswith("traj__")]
    assert "traj_x__mean_last" in out.columns

    # Context + provenance columns for downstream honesty filters.
    for col in ("survey", "label_source", "label_quality", "temporal_exactness"):
        assert col in out.columns, col


def test_helpfulness_fusion_empty_when_no_experts(tmp_path: Path) -> None:
    snap = _synthetic_fusion_snapshot()
    for col in snap.columns:
        if col.startswith("avail__"):
            snap[col] = 0.0
    snap_path = tmp_path / "snap.parquet"
    snap.to_parquet(snap_path, index=False)
    out_path = tmp_path / "help.parquet"
    helpmod.build_helpfulness_fusion(snap_path, out_path)
    out = pd.read_parquet(out_path)
    assert len(out) == 0


# ------------------------------------------------------------------ #
# Parity vs real v6e2 helpfulness (skips without local data)          #
# ------------------------------------------------------------------ #

@pytest.mark.skipif(
    not (_ROOT / "data/gold/expert_helpfulness.parquet").exists()
    or not (_ROOT / "data/gold/object_epoch_snapshots.parquet").exists(),
    reason="real v6e2 gold tables not present",
)
def test_trust_target_parity_against_v6e2(tmp_path: Path) -> None:
    """Rebuild helpfulness over a v6e2 snapshot sample via the fusion builder
    and check is_topclass_correct base-rate parity on shared rows."""
    snap = pd.read_parquet(_ROOT / "data/gold/object_epoch_snapshots.parquet")
    sample_ids = sorted(snap["object_id"].astype(str).unique())[:300]
    sample = snap[snap["object_id"].astype(str).isin(set(sample_ids))]
    snap_path = tmp_path / "v6e2_sample.parquet"
    sample.to_parquet(snap_path, index=False)
    out_path = tmp_path / "help_sample.parquet"
    helpmod.build_helpfulness_fusion(snap_path, out_path)
    helpmod.run_parity_check(
        out_path, _ROOT / "data/gold/expert_helpfulness.parquet"
    )
