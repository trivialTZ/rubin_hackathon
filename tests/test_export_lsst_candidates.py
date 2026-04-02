from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.export_lsst_candidates as export_mod
from scripts.export_lsst_candidates import export_lsst_candidates


def test_export_lsst_candidates_writes_expected_schema(tmp_path: Path, monkeypatch) -> None:
    fake_discover = types.SimpleNamespace(
        query_lasair_recent=lambda since_mjd, max_candidates: pd.DataFrame(
            [
                {
                    "object_id": "170032882292621441",
                    "ra": 62.3,
                    "dec": -46.1,
                    "n_det": 1,
                    "last_mjd": 61091.03,
                    "discovery_broker": "lasair",
                }
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "scripts.discover_nightly", fake_discover)

    out_path = tmp_path / "labels_lsst.csv"
    export_lsst_candidates(
        output_path=out_path,
        lookback_days=2,
        max_candidates=10,
        require_sherlock=True,
    )

    df = pd.read_csv(out_path)
    assert list(df["object_id"]) == [170032882292621441]
    assert list(df["survey"]) == ["LSST"]
    assert list(df["identifier_kind"]) == ["lsst_dia_object_id"]


def test_export_lsst_candidates_can_export_recent_lsst_rows_without_sherlock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        export_mod,
        "_query_recent_lsst_candidates",
        lambda since_mjd, max_candidates: pd.DataFrame(
            [
                {
                    "object_id": "170112073844916258",
                    "ra": 187.7,
                    "dec": 9.3,
                    "n_det": 1,
                    "last_mjd": 61109.28,
                    "discovery_broker": "lasair",
                }
            ]
        ),
    )

    out_path = tmp_path / "labels_lsst.csv"
    export_lsst_candidates(output_path=out_path, lookback_days=1, max_candidates=1)

    df = pd.read_csv(out_path)
    assert list(df["object_id"]) == [170112073844916258]
    assert list(df["survey"]) == ["LSST"]
