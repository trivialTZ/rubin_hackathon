"""P5 sequence-arm (fusion_v11) unit tests — all offline, tiny synthetic LCs.

Covers: v9 9-dim regression (deployed artifacts stay byte-safe), the v11
negative-token schema (is_negative + signed_flux channels, full-window
truncation, LSST 0-positive → empty), dim-aware NormStats, per-artifact
``seq_schema`` routing in the seq_v9/seq_v11 experts, the registry + explicit
projector dispatch (B7/A1 export contract), and the fetch_elasticc2 helpers.
No torch, no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from debass_meta.features.sequence_dataset import (
    SEQ_CONTINUOUS_DIM,
    SEQ_CONTINUOUS_DIM_V11,
    NormStats,
    _cont_dim_for_schema,
    _infer_survey,
    sequence_arrays,
    truncated_positive_detections,
)


def _det(mjd, mag, *, band="g", pos=True, survey="ZTF", flux=None, magerr=0.1, snr=20.0):
    return {
        "mjd": mjd, "mag": mag, "band": band, "magerr": magerr, "snr": snr,
        "is_positive": pos, "survey": survey, "flux": flux, "fluxerr": None,
        "quality": 0.9,
    }


def _ztf_lc(n=6, seed=0):
    rng = np.random.default_rng(seed)
    mjd, mag, out = 60000.0, 19.5, []
    for k in range(n):
        mjd += float(rng.uniform(0.4, 3.0))
        mag += float(rng.normal(-0.1, 0.08))
        out.append(_det(mjd, mag, band="g" if k % 2 == 0 else "r"))
    return out


# ── v9 regression (deployed artifacts must stay byte-safe) ───────────────────

def test_v9_schema_is_default_and_nine_dim():
    c, b = sequence_arrays(_ztf_lc(8))
    assert c.shape == (8, SEQ_CONTINUOUS_DIM) == (8, 9)
    assert b.shape == (8,)
    assert c[0, 5] == 1.0 and np.all(c[1:, 5] == 0.0)   # is_first
    # explicit schema="v9" is identical to the default
    c2, _ = sequence_arrays(_ztf_lc(8), schema="v9")
    np.testing.assert_array_equal(c, c2)


def test_v9_positives_only_and_causal():
    lc = _ztf_lc(6)
    lc.insert(2, _det(60100.0, 18.0, pos=False))         # a negative — dropped by v9
    c, _ = sequence_arrays(lc, schema="v9")
    assert c.shape[0] == 6                                # negative not tokenized
    full, _ = sequence_arrays(_ztf_lc(10), schema="v9")
    pre, _ = sequence_arrays(_ztf_lc(10)[:4], schema="v9")
    np.testing.assert_array_equal(full[:4], pre)         # row-wise causal


# ── v11 negative-token schema ────────────────────────────────────────────────

def test_v11_dims_and_negative_channels():
    dets = [
        _det(60000, 19.0, survey="LSST", flux=100.0),
        _det(60001, None, survey="LSST", pos=False, flux=-50.0),   # in-window negative
        _det(60002, 18.5, survey="LSST", flux=200.0),
    ]
    c, _ = sequence_arrays(dets, schema="v11")
    assert c.shape == (3, SEQ_CONTINUOUS_DIM_V11) == (3, 11)
    np.testing.assert_array_equal(c[:, 9], [0.0, 1.0, 0.0])        # is_negative
    # signed_flux = asinh(±|flux|): fade token is negative, detections positive
    assert c[0, 10] > 0 and c[2, 10] > 0 and c[1, 10] < 0
    assert abs(c[1, 10] - float(np.arcsinh(-50.0))) < 1e-4


def test_v11_reduces_to_v9_on_first_nine_cols_without_negatives():
    lc = _ztf_lc(7)
    c9, _ = sequence_arrays(lc, schema="v9")
    c11, _ = sequence_arrays(lc, schema="v11")
    assert c11.shape == (7, 11)
    np.testing.assert_array_equal(c11[:, :9], c9)                  # appended cols only


def test_v11_lsst_all_negative_yields_empty():
    allneg = [_det(60000, None, survey="LSST", pos=False, flux=-10.0),
              _det(60001, None, survey="LSST", pos=False, flux=-12.0)]
    c, b = sequence_arrays(allneg, schema="v11")
    assert len(c) == 0 and len(b) == 0
    assert _infer_survey(allneg) == "lsst"


def test_v11_ztf_all_negative_keeps_fallback():
    # non-LSST all-negative LC keeps the ZTF fallback (11% of ZTF LCs depend on it)
    allneg = [_det(60000, 19.0, survey="ZTF", pos=False),
              _det(60001, 19.2, survey="ZTF", pos=False)]
    assert len(truncated_positive_detections(allneg)) == 2
    c, _ = sequence_arrays(allneg, schema="v11")
    assert len(c) == 2 and np.all(c[:, 9] == 1.0)                  # both flagged negative


def test_unknown_schema_raises():
    with pytest.raises(ValueError):
        _cont_dim_for_schema("v12")
    with pytest.raises(ValueError):
        sequence_arrays(_ztf_lc(3), schema="v12")


# ── dim-aware NormStats ──────────────────────────────────────────────────────

def test_normstats_v11_is_eleven_dim_and_zscores_signed_flux():
    arrays = [sequence_arrays(
        [_det(60000 + i, 19 - 0.1 * i, survey="LSST", flux=100.0 * (i + 1),
              pos=(i % 3 != 1)) for i in range(6)], schema="v11")[0]
        for _ in range(4)]
    stats = NormStats.fit(arrays)
    assert len(stats.mean) == 11 and len(stats.std) == 11
    out = stats.apply(arrays[0])
    assert not np.allclose(out[:, 10], arrays[0][:, 10])          # signed_flux z-scored
    np.testing.assert_array_equal(out[:, 9], arrays[0][:, 9])     # is_negative flag raw
    # roundtrip
    stats2 = NormStats.from_json(json.loads(json.dumps(stats.to_json())))
    np.testing.assert_allclose(stats.apply(arrays[0]), stats2.apply(arrays[0]))


def test_normstats_v9_unchanged_by_v11_additions():
    arrays = [sequence_arrays(_ztf_lc(8, seed=s))[0] for s in range(4)]
    stats = NormStats.fit(arrays)
    assert len(stats.mean) == 9
    out = stats.apply(arrays[0])
    assert not np.allclose(out[:, 0], arrays[0][:, 0])            # dim0 z-scored
    np.testing.assert_array_equal(out[:, 5], arrays[0][:, 5])    # is_first flag raw


# ── expert schema routing + registration (B7/A1/B8) ──────────────────────────

def test_seq_experts_schema_routing():
    from debass_meta.experts.local.seq_v9 import SeqV9Expert, SeqV11Expert

    v9 = SeqV9Expert()
    assert v9._seq_schema() == "v9"                               # no artifact loaded
    v9._artifact = SimpleNamespace(meta={"seq_schema": "v11"})
    assert v9._seq_schema() == "v11"                             # per-artifact override
    v9._artifact = SimpleNamespace(meta={})                      # legacy v9/v10 meta
    assert v9._seq_schema() == "v9"

    assert issubclass(SeqV11Expert, SeqV9Expert)
    assert SeqV11Expert.name == "seq_v11"
    assert SeqV11Expert.env_var == "DEBASS_SEQ_V11_MODEL"
    assert SeqV11Expert.model_dir_candidates == ("models/seq_classifier_v11",)


def test_seq_v11_env_override(monkeypatch, tmp_path):
    from debass_meta.experts.local.seq_v9 import SeqV11Expert

    monkeypatch.setenv("DEBASS_SEQ_V11_MODEL", str(tmp_path / "custom_v11"))
    assert SeqV11Expert()._model_dir == tmp_path / "custom_v11"


def test_seq_v11_registered_and_dispatched():
    from debass_meta.projectors.base import (
        ALL_EXPERT_KEYS,
        EXPERT_REGISTRY,
        _dispatch_projector,
        project_expert_events,
    )

    assert EXPERT_REGISTRY["seq_v11"] == ("any", "local_seq_v9")
    assert "seq_v11" in ALL_EXPERT_KEYS

    events = [
        {"class_name": "snia", "canonical_projection": 0.7},
        {"class_name": "nonIa_snlike", "canonical_projection": 0.2},
        {"class_name": "other", "canonical_projection": 0.1},
    ]
    proj = _dispatch_projector("seq_v11", events)
    assert proj["prediction_type"] == "class_correctness"
    assert abs(proj["p_snia"] - 0.7) < 1e-6
    # public entrypoint routes identically (never raises)
    assert project_expert_events("seq_v11", events)["mapped_pred_class"] == "snia"


def test_seq_v11_in_local_experts_and_not_sn_filter():
    from debass_meta.experts.local import ALL_LOCAL_EXPERTS
    from debass_meta.models.expert_trust import SN_FILTER_EXPERTS

    names = [c.name for c in ALL_LOCAL_EXPERTS]
    assert "seq_v11" in names and "seq_v9" in names
    assert "seq_v11" not in SN_FILTER_EXPERTS       # ternary trust target, not is_sn


# ── fetch_elasticc2 (offline helpers) ────────────────────────────────────────

def _load_fetch_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_elasticc2.py"
    spec = importlib.util.spec_from_file_location("fetch_elasticc2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_elasticc2_class_selection():
    fe = _load_fetch_module()
    assert fe.is_sn_model("SNIa-SALT3") and fe.is_sn_model("SLSN-I+host")
    assert not fe.is_sn_model("AGN") and not fe.is_sn_model("Cepheid")
    available = ["SNIa-SALT3", "SNII-Templates", "AGN", "Cepheid", "SLSN-I"]
    default = fe.select_models(None, available, full=False)
    assert set(default) == {"SNIa-SALT3", "SNII-Templates", "SLSN-I"}
    assert fe.select_models(None, available, full=True) == available
    assert fe.select_models("AGN,SNIa-SALT3", available, full=False) == ["AGN", "SNIa-SALT3"]
    with pytest.raises(SystemExit):
        fe.select_models("NoSuchClass", available, full=False)


def test_fetch_elasticc2_index_parsing():
    fe = _load_fetch_module()
    html = (
        '<a href="../">..</a>'
        '<a href="ELASTICC2_TRAIN_02_SNIa-SALT3/">dir</a>'
        '<a href="ELASTICC2_TRAIN_02_AGN/">dir</a>'
        '<a href="/absolute/skip/">skip</a>'
        '<a href="ELASTICC2_TRAIN_02_SNIa-SALT3_HEAD.FITS.gz">shard</a>'
    )
    links = fe._parse_index_links(html)
    assert "ELASTICC2_TRAIN_02_SNIa-SALT3/" in links
    assert "../" not in links and "/absolute/skip/" not in links
    assert fe.BASE_URL.endswith("ELASTICC2_TRAINING_SAMPLE_2/")
