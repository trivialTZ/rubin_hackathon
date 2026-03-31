from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass.features.lightcurve import extract_features


def test_extract_features_skips_slope_when_times_are_identical() -> None:
    feats = extract_features(
        [
            {"mjd": 60000.0, "magpsf": 19.1, "fid": 1, "isdiffpos": "t"},
            {"mjd": 60000.0, "magpsf": 19.0, "fid": 1, "isdiffpos": "t"},
        ]
    )

    assert math.isnan(feats["dmag_dt"])
    assert feats["mag_std"] >= 0.0
