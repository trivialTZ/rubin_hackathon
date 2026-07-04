"""v9 sequence encoder tests — causality (no-leakage), determinism, exports.

The load-bearing property: every export at prefix n_det=k must be IDENTICAL
whether or not future detections exist beyond k.  This holds by construction
(per-detection features look only backwards; the GRU is unidirectional) and is
asserted end-to-end here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from debass_meta.features.sequence_dataset import (  # noqa: E402
    BAND_TO_IDX,
    SEQ_CONTINUOUS_DIM,
    UNKNOWN_BAND_IDX,
    NormStats,
    sequence_arrays,
    truncated_positive_detections,
)
from debass_meta.models.seq_encoder import (  # noqa: E402
    SEQ_EMB_DIM,
    SeqEncoder,
    SeqEncoderConfig,
    encode_prefixes,
    gaussian_nll,
    load_encoder,
    save_encoder,
    ssl_next_dmag_loss,
)


def _det(mjd, mag, band="g", magerr=0.1, snr=20.0, positive=True, survey="ZTF"):
    return {
        "mjd": mjd, "mag": mag, "band": band, "magerr": magerr, "snr": snr,
        "is_positive": positive, "survey": survey,
        "flux": None, "fluxerr": None, "quality": 0.9,
    }


def _lightcurve(n=12, seed=0):
    rng = np.random.default_rng(seed)
    dets = []
    mjd, mag = 60000.0, 19.5
    for k in range(n):
        mjd += float(rng.uniform(0.4, 3.0))
        mag += float(rng.normal(-0.12, 0.08))
        dets.append(_det(mjd, mag, band="g" if k % 2 == 0 else "r"))
    return dets


def _encoder(seed=7):
    torch.manual_seed(seed)
    return SeqEncoder(SeqEncoderConfig()).eval()


# ── dataset-level ────────────────────────────────────────────────────────────

def test_sequence_arrays_shapes_and_bands():
    cont, bands = sequence_arrays(_lightcurve(8))
    assert cont.shape == (8, SEQ_CONTINUOUS_DIM)
    assert bands.shape == (8,)
    assert set(np.unique(bands)) <= {BAND_TO_IDX["g"], BAND_TO_IDX["r"]}
    assert cont[0, 5] == 1.0 and np.all(cont[1:, 5] == 0.0)        # is_first
    assert cont[0, 0] == 0.0 and np.all(cont[1:, 0] > 0.0)          # log_dt_prev


def test_sequence_features_are_causal_rowwise():
    full = _lightcurve(12)
    cont_full, bands_full = sequence_arrays(full)
    for k in (1, 3, 5, 8):
        cont_pre, bands_pre = sequence_arrays(full[:k])
        np.testing.assert_array_equal(cont_full[:k], cont_pre)
        np.testing.assert_array_equal(bands_full[:k], bands_pre)


def test_positive_filter_and_sorting_match_gold_recipe():
    dets = _lightcurve(6)
    dets.insert(2, _det(60100.0, 18.0, positive=False))            # negative det dropped
    shuffled = [dets[i] for i in (3, 0, 5, 1, 6, 2, 4)]
    trunc = truncated_positive_detections(shuffled)
    mjds = [d["mjd"] for d in trunc]
    assert mjds == sorted(mjds)
    assert all(d.get("is_positive", True) for d in trunc)
    assert len(trunc) == 6


def test_unknown_band_and_missing_err_flags():
    dets = [_det(60000.0, 19.0, band="w", magerr=None, snr=None)]
    cont, bands = sequence_arrays(dets)
    assert bands[0] == UNKNOWN_BAND_IDX
    assert cont[0, 6] == 1.0 and cont[0, 7] == 1.0                  # missing flags


def test_norm_stats_roundtrip(tmp_path: Path):
    arrays = [sequence_arrays(_lightcurve(10, seed=s))[0] for s in range(5)]
    stats = NormStats.fit(arrays)
    payload = json.loads(json.dumps(stats.to_json()))
    stats2 = NormStats.from_json(payload)
    np.testing.assert_allclose(stats.apply(arrays[0]), stats2.apply(arrays[0]))
    assert not np.allclose(stats.apply(arrays[0])[:, 0], arrays[0][:, 0])      # z-scored
    np.testing.assert_array_equal(stats.apply(arrays[0])[:, 5], arrays[0][:, 5])  # flags raw


# ── encoder-level ────────────────────────────────────────────────────────────

def test_encoder_prefix_causality_end_to_end():
    """THE no-leakage test: exports at prefix k identical with/without future dets."""
    enc = _encoder()
    stats = NormStats()  # identity normalization keeps the comparison exact
    full = _lightcurve(12)
    cont_f, bands_f = sequence_arrays(full)
    out_full = encode_prefixes(enc, stats.apply(cont_f), bands_f)
    for k in (1, 2, 5, 9):
        cont_p, bands_p = sequence_arrays(full[:k])
        out_pre = encode_prefixes(enc, stats.apply(cont_p), bands_p)
        np.testing.assert_allclose(out_full["emb"][:k], out_pre["emb"], rtol=0, atol=1e-5)
        np.testing.assert_allclose(
            out_full["surprisal"][:k], out_pre["surprisal"], rtol=0, atol=1e-5, equal_nan=True
        )


def test_surprisal_nan_at_first_then_finite():
    enc = _encoder()
    cont, bands = sequence_arrays(_lightcurve(6))
    out = encode_prefixes(enc, NormStats().apply(cont), bands)
    assert math.isnan(out["surprisal"][0]) and math.isnan(out["nll_mean"][0])
    assert np.all(np.isfinite(out["surprisal"][1:]))
    assert out["emb"].shape == (6, SEQ_EMB_DIM)


def test_encoder_deterministic_in_eval_mode():
    enc = _encoder()
    cont, bands = sequence_arrays(_lightcurve(7))
    a = encode_prefixes(enc, NormStats().apply(cont), bands)
    b = encode_prefixes(enc, NormStats().apply(cont), bands)
    np.testing.assert_array_equal(a["emb"], b["emb"])


def test_ssl_loss_masks_padding_and_decreases_with_training():
    torch.manual_seed(0)
    enc = SeqEncoder(SeqEncoderConfig())
    seqs = [sequence_arrays(_lightcurve(10, seed=s)) for s in range(24)]
    stats = NormStats.fit([c for c, _ in seqs])
    L = max(len(c) for c, _ in seqs)
    cont = np.zeros((len(seqs), L, SEQ_CONTINUOUS_DIM), dtype=np.float32)
    bands = np.zeros((len(seqs), L), dtype=np.int64)
    lengths = torch.tensor([len(c) for c, _ in seqs])
    for i, (c, b) in enumerate(seqs):
        cont[i, : len(c)] = stats.apply(c)
        bands[i, : len(b)] = b
    tc, tb = torch.from_numpy(cont), torch.from_numpy(bands)
    loss0, n0 = ssl_next_dmag_loss(enc, tc, tb, lengths)
    assert int(n0) == int((lengths - 1).sum())          # padding masked out
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    for _ in range(30):
        opt.zero_grad()
        loss, _ = ssl_next_dmag_loss(enc, tc, tb, lengths)
        loss.backward()
        opt.step()
    loss1, _ = ssl_next_dmag_loss(enc, tc, tb, lengths)
    assert float(loss1) < float(loss0)


def test_gaussian_nll_matches_closed_form():
    t = torch.tensor([0.5]); m = torch.tensor([0.0]); lv = torch.tensor([0.0])
    expected = 0.5 * (0.25 + 0.0 + math.log(2 * math.pi))
    assert abs(float(gaussian_nll(t, m, lv)) - expected) < 1e-6


def test_save_load_roundtrip(tmp_path: Path):
    enc = _encoder()
    stats = NormStats(mean=[0.1] * SEQ_CONTINUOUS_DIM, std=[2.0] * SEQ_CONTINUOUS_DIM)
    save_encoder(enc, stats, tmp_path / "art", extra_meta={"corpus_objects": 5})
    enc2, stats2 = load_encoder(tmp_path / "art")
    cont, bands = sequence_arrays(_lightcurve(5))
    a = encode_prefixes(enc, stats.apply(cont), bands)
    b = encode_prefixes(enc2, stats2.apply(cont), bands)
    np.testing.assert_allclose(a["emb"], b["emb"], atol=1e-6)
    meta = json.loads((tmp_path / "art" / "config.json").read_text())
    assert meta["corpus_objects"] == 5


# ── corpus exclusion (transduction hygiene) ──────────────────────────────────

def test_train_script_excludes_cal_test(tmp_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_seq_encoder", Path(__file__).resolve().parents[1] / "scripts" / "train_seq_encoder.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "train_ids": ["A", "B"], "cal_ids": ["C1", "C2"], "test_ids": ["T1"],
    }))
    excl = mod.load_split_exclusions(manifest)
    assert excl == {"C1", "C2", "T1"}
    with pytest.raises(SystemExit):
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"train_ids": ["A"]}))
        mod.load_split_exclusions(empty)
