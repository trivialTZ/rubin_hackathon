from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_lightcurves import fetch_lightcurves_for_objects


class _StubFetcher:
    def __init__(self, lc_dir: Path) -> None:
        self.lc_dir = lc_dir
        self.queries: list[str] = []

    def fetch_epoch_scores(self, object_id: str, max_epochs: int = 20):
        self.queries.append(object_id)
        self.lc_dir.mkdir(parents=True, exist_ok=True)
        with open(self.lc_dir / f"{object_id}.json", "w") as fh:
            json.dump([{"mjd": 60000.0, "magpsf": 20.0, "fid": 1, "isdiffpos": "t"}], fh)
        return [{"object_id": object_id, "n_det": 1}]


def test_fetch_lightcurves_for_objects_copies_associated_cache_for_lsst_primary(tmp_path: Path) -> None:
    fetcher = _StubFetcher(tmp_path)

    ok, skipped = fetch_lightcurves_for_objects(
        object_ids=["170032882292621441"],
        fetcher=fetcher,
        lc_dir=tmp_path,
        associations={
            "170032882292621441": {
                "ztf_object_id": "ZTF23abcegjv",
                "sep_arcsec": 0.21,
                "association_kind": "position_crossmatch",
                "association_source": "alerce_conesearch",
            }
        },
    )

    assert ok == 1
    assert skipped == 0
    assert fetcher.queries == ["ZTF23abcegjv"]
    assert (tmp_path / "170032882292621441.json").exists()
    assert (tmp_path / "ZTF23abcegjv.json").exists()
