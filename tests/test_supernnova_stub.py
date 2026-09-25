"""v13b: SuperNNova stub outputs (uniform priors) must be unavailable, like ParSNIP's."""
from __future__ import annotations

from pathlib import Path

from debass_meta.experts.local.supernnova import SuperNNovaExpert

LC = [{"mjd": 60000.0 + i, "band": "g", "mag": 20.0 - 0.1 * i, "magerr": 0.05} for i in range(5)]
JD = 2400000.5 + 60010.0


def _expert(tmp_path: Path, *, model: bool, classify=None) -> SuperNNovaExpert:
    e = SuperNNovaExpert(model_dir=tmp_path)
    e._snn_available = True          # package "installed"
    e._model_file = (tmp_path / "model.pt") if model else None
    e._classify_lcs = classify
    return e


def test_stub_without_model_is_unavailable(tmp_path: Path) -> None:
    e = _expert(tmp_path, model=False)
    out = e.predict_epoch("obj", LC, JD)
    assert out.model_version == "stub" and out.available is False
    batch = e.predict_epoch_batch([("obj", LC, JD, 5)])
    assert all(o.model_version == "stub" and o.available is False for o in batch)


def test_batch_classify_failure_is_unavailable(tmp_path: Path) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("classify failed")

    e = _expert(tmp_path, model=True, classify=boom)
    batch = e.predict_epoch_batch([("a", LC, JD, 5), ("b", [], JD, 0)])
    assert [o.available for o in batch] == [False, False]
    assert all(o.model_version == "stub" for o in batch)


def test_single_epoch_without_detections_is_unavailable(tmp_path: Path) -> None:
    e = _expert(tmp_path, model=True, classify=lambda *a, **k: None)
    out = e.predict_epoch("obj", [], JD)
    assert out.model_version == "stub" and out.available is False
