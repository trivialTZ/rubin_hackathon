"""WP3 tests — self-supervised next-detection pretraining on unlabeled LSST.

No live network: the harvest is exercised with a fake Lasair adapter, and the
pretraining objective / artifact round-trip run on synthetic token streams.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from debass_meta.features.sequence_dataset import (
    N_BANDS,
    NormStats,
    _cont_dim_for_schema,
)
from debass_meta.models.seq_encoder import SeqEncoder, SeqEncoderConfig, load_encoder
from scripts.harvest_unlabeled_lsst import (
    harvest_object_ids,
    load_benchmark_exclusions,
    pick_detcount_column,
    record_ndet,
    record_object_id,
    select_ids,
    shard_of,
)
from scripts.pretrain_seq_ssl import (
    NextDetHeads,
    assert_encoder_roundtrip,
    build_corpus,
    continuous_target_dims,
    ssl_next_detection_loss,
)


# ------------------------------------------------------------------ #
# Pure harvest helpers                                                #
# ------------------------------------------------------------------ #


def test_pick_detcount_column():
    assert pick_detcount_column(["diaObjectId", "nDiaSources", "ra"]) == "nDiaSources"
    assert pick_detcount_column(["diaObjectId", "nDetections"]) == "nDetections"
    assert pick_detcount_column(["diaObjectId", "ra"]) is None


def test_record_helpers():
    rec = {"diaObjectId": 12345, "n_det": "7"}
    assert record_object_id(rec) == "12345"
    assert record_ndet(rec, "nDiaSources") == 7
    assert record_ndet({"nDiaSources": 4}, "nDiaSources") == 4
    assert record_ndet({}, "nDiaSources") is None


def test_shard_of_deterministic_and_partitions():
    # Stable across calls.
    assert shard_of("obj42", 4) == shard_of("obj42", 4)
    # n_shards<=1 always shard 0.
    assert shard_of("anything", 1) == 0
    # Roughly partitions the id space across shards.
    counts = [0, 0, 0]
    for i in range(300):
        counts[shard_of(f"obj{i}", 3)] += 1
    assert all(c > 0 for c in counts)


def test_select_ids_filters_and_dedups():
    records = [
        {"diaObjectId": "a", "n_det": 5},
        {"diaObjectId": "b", "n_det": 2},   # below min_ndet
        {"diaObjectId": "c", "n_det": 9},
        {"diaObjectId": "a", "n_det": 5},   # duplicate across pages
        {"diaObjectId": "bench", "n_det": 8},  # benchmark-excluded
    ]
    seen: set[str] = set()
    got = select_ids(records, detcol="nDiaSources", min_ndet=3,
                     exclude={"bench"}, seen=seen)
    assert got == ["a", "c"]
    # A second page reusing 'a' is deduped via the shared seen set.
    more = select_ids([{"diaObjectId": "a", "n_det": 5}, {"diaObjectId": "d", "n_det": 4}],
                      detcol="nDiaSources", min_ndet=3, exclude={"bench"}, seen=seen)
    assert more == ["d"]


def test_benchmark_exclusion_honored(tmp_path):
    manifest = tmp_path / "locked.json"
    manifest.write_text(json.dumps({
        "test_ids": ["100", "200", "300"],
        "excluded_ids": {"999": "counterpart reason"},
    }))
    excl = load_benchmark_exclusions(manifest)
    assert excl == {"100", "200", "300", "999"}

    records = [{"diaObjectId": o, "n_det": 5} for o in ["100", "200", "300", "999", "777"]]
    got = select_ids(records, detcol="nDiaSources", min_ndet=3, exclude=excl)
    assert got == ["777"]  # every benchmark id filtered out


def test_load_benchmark_exclusions_missing_file(tmp_path):
    assert load_benchmark_exclusions(tmp_path / "nope.json") == set()


# ------------------------------------------------------------------ #
# Harvest pagination / caching (fake Lasair adapter, no network)      #
# ------------------------------------------------------------------ #


class _FakeAdapter:
    """Fake LasairAdapter: serves a fixed objects table via offset paging."""

    def __init__(self, table: list[dict]):
        self.table = table
        self.calls = 0

    def _query_api(self, *, selected, tables, conditions, limit, offset, survey):
        self.calls += 1
        assert "ORDER BY objects.diaObjectId" in conditions
        assert tables == "objects"
        return self.table[offset : offset + limit]


class _ExplodingAdapter:
    def _query_api(self, **kwargs):
        raise AssertionError("network hit — cache was not used")


def test_harvest_pagination_and_caching(tmp_path):
    table = [{"diaObjectId": f"obj{i:03d}", "n_det": 5} for i in range(25)]
    cache = tmp_path / "cache"
    fake = _FakeAdapter(table)
    ids = harvest_object_ids(
        target=15, cache_dir=cache, exclude=set(), min_ndet=3,
        page_size=10, detcol="nDiaSources", adapter=fake)
    assert len(ids) == 15
    assert ids == [f"obj{i:03d}" for i in range(15)]  # ORDER BY preserved
    assert fake.calls == 2  # pages 0 and 1 only (stopped once target hit)
    assert (cache).exists() and list(cache.glob("objects_*.json"))

    # Re-run reads from disk cache — the exploding adapter must never be called.
    ids2 = harvest_object_ids(
        target=15, cache_dir=cache, exclude=set(), min_ndet=3,
        page_size=10, detcol="nDiaSources", adapter=_ExplodingAdapter())
    assert ids2 == ids


def test_harvest_excludes_benchmark_and_shards(tmp_path):
    table = [{"diaObjectId": f"obj{i:03d}", "n_det": 5} for i in range(40)]
    excl = {"obj000", "obj001"}
    fake = _FakeAdapter(table)
    ids = harvest_object_ids(
        target=40, cache_dir=tmp_path / "c", exclude=excl, min_ndet=3,
        shard_id=1, n_shards=3, page_size=1000, detcol="nDiaSources", adapter=fake)
    assert not (excl & set(ids))               # benchmark ids never harvested
    assert all(shard_of(o, 3) == 1 for o in ids)  # only this shard's slice


# ------------------------------------------------------------------ #
# SSL objective + artifact                                            #
# ------------------------------------------------------------------ #


def test_continuous_target_dims():
    assert continuous_target_dims("v9") == (0, 1)
    assert continuous_target_dims("v11") == (0, 1, 10)


def _synthetic_batch(schema: str, B: int = 8, L: int = 6, seed: int = 0):
    cont_dim = _cont_dim_for_schema(schema)
    rng = np.random.default_rng(seed)
    cont = rng.standard_normal((B, L, cont_dim)).astype(np.float32)
    # Make band predictable from a continuous channel so the CE head has signal.
    bands = (np.clip(cont[:, :, 0], -3, 3) + 3).astype(np.int64) % N_BANDS
    lengths = np.full(B, L, dtype=np.int64)
    return (torch.from_numpy(cont), torch.from_numpy(bands),
            torch.from_numpy(lengths))


@pytest.mark.parametrize("schema", ["v9", "v11"])
def test_ssl_loss_decreases(schema):
    torch.manual_seed(0)
    cont, bands, lengths = _synthetic_batch(schema)
    target_dims = continuous_target_dims(schema)
    encoder = SeqEncoder(SeqEncoderConfig(cont_dim=_cont_dim_for_schema(schema)))
    heads = NextDetHeads(encoder.config.hidden, len(target_dims))
    opt = torch.optim.Adam(list(encoder.parameters()) + list(heads.parameters()), lr=5e-3)

    init_loss, init_parts, n = ssl_next_detection_loss(
        encoder, heads, cont, bands, lengths, target_dims)
    assert int(n) == 8 * (6 - 1)  # every non-final step of every object is a target
    for _ in range(60):
        opt.zero_grad()
        loss, _, _ = ssl_next_detection_loss(encoder, heads, cont, bands, lengths, target_dims)
        loss.backward()
        opt.step()
    final_loss, final_parts, _ = ssl_next_detection_loss(
        encoder, heads, cont, bands, lengths, target_dims)
    assert float(final_loss) < float(init_loss)
    # Both heads improve on this overfittable batch.
    assert final_parts["reg"] < init_parts["reg"]
    assert final_parts["band"] < init_parts["band"]


def test_ssl_loss_masks_padding():
    """Padded steps beyond `length` must not contribute to the loss."""
    torch.manual_seed(1)
    schema = "v9"
    cont_dim = _cont_dim_for_schema(schema)
    cont = torch.randn(2, 5, cont_dim)
    bands = torch.randint(0, N_BANDS, (2, 5))
    lengths = torch.tensor([5, 2], dtype=torch.long)
    encoder = SeqEncoder(SeqEncoderConfig(cont_dim=cont_dim)).eval()  # no stochastic dropout
    heads = NextDetHeads(encoder.config.hidden, 2).eval()
    target_dims = (0, 1)
    _, _, n = ssl_next_detection_loss(encoder, heads, cont, bands, lengths, target_dims)
    # obj0 contributes 4 (steps 0..3 predict 1..4); obj1 contributes 1 (step0->1).
    assert int(n) == 5
    # Corrupting the padded tail of obj1 leaves the loss unchanged.
    _, parts_a, _ = ssl_next_detection_loss(encoder, heads, cont, bands, lengths, target_dims)
    cont2 = cont.clone()
    cont2[1, 2:] += 100.0
    bands2 = bands.clone()
    bands2[1, 2:] = 0
    _, parts_b, _ = ssl_next_detection_loss(encoder, heads, cont2, bands2, lengths, target_dims)
    assert parts_a["reg"] == pytest.approx(parts_b["reg"], abs=1e-5)
    assert parts_a["band"] == pytest.approx(parts_b["band"], abs=1e-5)


@pytest.mark.parametrize("schema", ["v9", "v11"])
def test_artifact_roundtrip(tmp_path, schema):
    cont_dim = _cont_dim_for_schema(schema)
    encoder = SeqEncoder(SeqEncoderConfig(cont_dim=cont_dim))
    stats = NormStats(mean=[0.0] * cont_dim, std=[1.0] * cont_dim)
    out = tmp_path / "enc"
    from debass_meta.models.seq_encoder import save_encoder

    save_encoder(encoder, stats, out, extra_meta={"seq_schema": schema, "cont_dim": cont_dim})
    # The deliverable: load_encoder + SeqClassifier accept it (asserts inside).
    assert_encoder_roundtrip(encoder, stats, out)

    # Independently confirm the classifier trainer's exact loading path works
    # and that meta carries the schema the classifier must be run with.
    enc2, _ = load_encoder(out)
    assert enc2.config.cont_dim == cont_dim
    meta = json.loads((out / "config.json").read_text())
    assert meta["seq_schema"] == schema
    assert (out / "encoder.pt").exists()
    assert (out / "norm_stats.json").exists()


# ------------------------------------------------------------------ #
# Corpus building                                                     #
# ------------------------------------------------------------------ #


def _write_lc(lc_dir, oid, n_det):
    dets = []
    for k in range(n_det):
        dets.append({
            "mjd": 60000.0 + k * 2.0,
            "mag": 20.0 - 0.1 * k,
            "magerr": 0.05,
            "snr": 15.0,
            "band": "r",
            "survey": "LSST",
            "is_positive": True,
            "flux": 100.0 + 10.0 * k,
        })
    (lc_dir / f"{oid}.json").write_text(json.dumps(dets))


def test_build_corpus_excludes_benchmark_and_short(tmp_path):
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    _write_lc(lc_dir, "keep_a", 4)
    _write_lc(lc_dir, "keep_b", 3)
    _write_lc(lc_dir, "bench_x", 5)   # excluded
    _write_lc(lc_dir, "too_short", 1)  # <2 dets -> no forecast target
    ids, seqs = build_corpus(
        lc_dir, exclude={"bench_x"}, max_len=60, schema="v9", surveys="both")
    assert set(ids) == {"keep_a", "keep_b"}
    assert all(len(c) >= 2 for c, _ in seqs)
