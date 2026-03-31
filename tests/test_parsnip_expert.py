from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass.experts.local.parsnip import ParSNIPExpert


class FakeTable:
    def __init__(self, rows=None, names=None):
        self.colnames = list(names or [])
        self.rows = []
        if rows is not None and names is not None:
            for row in rows:
                self.rows.append({name: value for name, value in zip(self.colnames, row)})
        self.meta = {}

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        if isinstance(key, list):
            filtered = FakeTable(names=self.colnames)
            filtered.rows = [row for row, keep in zip(self.rows, key) if keep]
            filtered.meta = self.meta.copy()
            return filtered
        raise TypeError(key)


class FakeModel:
    def __init__(self, settings):
        self.settings = settings
        self.last_tables = None

    def predict(self, tables):
        self.last_tables = tables
        return FakeTable(
            rows=[("ZTF_TEST", 0.12, 0.03, 0.45)],
            names=("object_id", "color", "color_error", "s1"),
        )


class FakeClassifierInstance:
    def __init__(self, class_names=None):
        self.class_names = class_names or ["SN Ia", "SN II"]

    def classify(self, predictions):
        return FakeTable(
            rows=[("ZTF_TEST", 0.7, 0.3)],
            names=("object_id", "SN Ia", "SN II"),
        )


def _install_fake_astropy(monkeypatch):
    astropy_module = types.ModuleType("astropy")
    astropy_table_module = types.ModuleType("astropy.table")
    astropy_table_module.Table = FakeTable
    astropy_module.table = astropy_table_module
    monkeypatch.setitem(sys.modules, "astropy", astropy_module)
    monkeypatch.setitem(sys.modules, "astropy.table", astropy_table_module)


def _install_fake_parsnip(monkeypatch, *, predict_redshift=True):
    fake_model = FakeModel({"bands": ["ztfg", "ztfr"], "predict_redshift": predict_redshift})
    fake_classifier = FakeClassifierInstance()

    class FakeClassifierLoader:
        @staticmethod
        def load(path):
            return fake_classifier

    parsnip_module = types.ModuleType("parsnip")
    parsnip_module.load_model = lambda path, device="cpu", threads=8: fake_model
    parsnip_module.Classifier = FakeClassifierLoader
    monkeypatch.setitem(sys.modules, "parsnip", parsnip_module)
    return fake_model, fake_classifier


def test_parsnip_metadata_requires_classifier(monkeypatch, tmp_path: Path) -> None:
    _install_fake_parsnip(monkeypatch)

    model_dir = tmp_path / "parsnip"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("weights")

    expert = ParSNIPExpert(model_dir=model_dir, device="cpu")
    meta = expert.metadata()

    assert meta["available"] is True
    assert meta["model_component_loaded"] is True
    assert meta["classifier_loaded"] is False
    assert meta["model_loaded"] is False
    assert meta["inference_implemented"] is True


def test_parsnip_predict_epoch_returns_classifier_probabilities(
    monkeypatch, tmp_path: Path
) -> None:
    fake_model, _ = _install_fake_parsnip(monkeypatch, predict_redshift=True)
    _install_fake_astropy(monkeypatch)

    model_dir = tmp_path / "parsnip"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("weights")
    (model_dir / "classifier.pkl").write_text("classifier")

    expert = ParSNIPExpert(model_dir=model_dir, device="cpu")
    result = expert.predict_epoch(
        "ZTF_TEST",
        [
            {"mjd": 60000.0, "magpsf": 19.5, "sigmapsf": 0.1, "fid": 1, "ra": 10.0, "dec": -5.0},
            {"mjd": 60001.0, "magpsf": 19.1, "sigmapsf": 0.1, "fid": 2, "ra": 10.0, "dec": -5.0},
        ],
        2460002.0,
    )

    assert result.model_version == "loaded"
    assert result.class_probabilities == {"SN Ia": 0.7, "SN II": 0.3}
    assert result.raw_output["prediction"]["color"] == 0.12
    assert result.raw_output["bands"] == ["ztfg", "ztfr"]

    tables = fake_model.last_tables
    assert tables is not None and len(tables) == 1
    table = tables[0]
    assert table.meta["object_id"] == "ZTF_TEST"
    assert table["band"] == ["ztfg", "ztfr"]
    assert len(table["time"]) == 2
    assert all(value > 0 for value in table["flux"])


def test_parsnip_requires_redshift_when_model_does_not_predict_it(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_parsnip(monkeypatch, predict_redshift=False)
    _install_fake_astropy(monkeypatch)

    model_dir = tmp_path / "parsnip"
    model_dir.mkdir()
    (model_dir / "model.pt").write_text("weights")
    (model_dir / "classifier.pkl").write_text("classifier")

    expert = ParSNIPExpert(model_dir=model_dir, device="cpu")

    with pytest.raises(ValueError, match="requires redshift metadata"):
        expert.predict_epoch(
            "ZTF_TEST",
            [{"mjd": 60000.0, "magpsf": 19.5, "sigmapsf": 0.1, "fid": 1}],
            2460001.0,
        )
