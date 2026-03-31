from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.alerce_history import AlerceHistoryFetcher



def test_alerce_history_marks_latest_snapshot_as_unsafe_and_converts_mjd(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    (lc_dir / "ZTF1.json").write_text(json.dumps([
        {"mjd": 60000.0, "magpsf": 20.0, "fid": 1, "isdiffpos": "t"}
    ]))

    fetcher = AlerceHistoryFetcher(lc_dir=lc_dir)
    fetcher._fetch_probabilities = lambda object_id: [
        {
            "field": "lc_classifier_transient_SNIa",
            "raw_label_or_score": 0.9,
            "semantic_type": "probability",
            "canonical_projection": 0.9,
            "classifier": "lc_classifier_transient",
            "class_name": "SNIa",
        }
    ]

    epochs = fetcher.fetch_epoch_scores("ZTF1", max_epochs=1)

    assert len(epochs) == 1
    epoch = epochs[0]
    assert epoch.alert_jd == 2460000.5
    assert epoch.fields[0]["event_scope"] == "object_snapshot"
    assert epoch.fields[0]["temporal_exactness"] == "latest_object_unsafe"
    assert epoch.fields[0]["event_time_jd"] is None
