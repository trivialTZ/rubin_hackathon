from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.crossmatch_lsst_to_ztf import crossmatch_lsst_to_ztf


def test_crossmatch_lsst_to_ztf_picks_closest_match(tmp_path: Path, monkeypatch) -> None:
    candidates = pd.DataFrame(
        [
            {
                "object_id": "170032882292621441",
                "ra": 10.0,
                "dec": -20.0,
                "n_det": 2,
                "last_mjd": 61091.0,
            }
        ]
    )
    candidates_path = tmp_path / "labels_lsst.csv"
    candidates.to_csv(candidates_path, index=False)

    class FakeClient:
        def query_objects(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "oid": "ZTF_FAR",
                        "meanra": 10.0003,
                        "meandec": -20.0003,
                        "lastmjd": 61090.0,
                        "ndet": 3,
                    },
                    {
                        "oid": "ZTF_NEAR",
                        "meanra": 10.00001,
                        "meandec": -20.00001,
                        "lastmjd": 61089.0,
                        "ndet": 4,
                    },
                ]
            )

    fake_core = types.SimpleNamespace(Alerce=lambda: FakeClient())
    monkeypatch.setitem(sys.modules, "alerce", types.SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "alerce.core", fake_core)

    output_path = tmp_path / "lsst_to_ztf.csv"
    crossmatch_lsst_to_ztf(
        candidates_path=candidates_path,
        output_path=output_path,
        radius_arcsec=2.0,
        page_size=10,
    )

    out = pd.read_csv(output_path, dtype={"lsst_object_id": "string", "ztf_object_id": "string"})
    assert list(out["lsst_object_id"]) == ["170032882292621441"]
    assert list(out["ztf_object_id"]) == ["ZTF_NEAR"]
    assert list(out["match_status"]) == ["matched"]
    matches = json.loads(out.iloc[0]["candidate_matches_json"])
    assert matches[0]["ztf_object_id"] == "ZTF_NEAR"


def test_crossmatch_lsst_to_ztf_records_no_match(tmp_path: Path, monkeypatch) -> None:
    candidates = pd.DataFrame([{"object_id": "1700", "ra": 1.0, "dec": 2.0}])
    candidates_path = tmp_path / "labels_lsst.csv"
    candidates.to_csv(candidates_path, index=False)

    class FakeClient:
        def query_objects(self, **kwargs):
            return pd.DataFrame()

    fake_core = types.SimpleNamespace(Alerce=lambda: FakeClient())
    monkeypatch.setitem(sys.modules, "alerce", types.SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "alerce.core", fake_core)

    output_path = tmp_path / "lsst_to_ztf.csv"
    crossmatch_lsst_to_ztf(
        candidates_path=candidates_path,
        output_path=output_path,
    )

    out = pd.read_csv(output_path, dtype={"lsst_object_id": "string"})
    assert list(out["lsst_object_id"]) == ["1700"]
    assert list(out["match_status"]) == ["no_match"]
    assert int(out.iloc[0]["match_count"]) == 0
