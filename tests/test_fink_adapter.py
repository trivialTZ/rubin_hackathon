from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.access.fink import FinkAdapter


def test_fink_extract_fields_preserves_alert_id_and_n_det() -> None:
    adapter = FinkAdapter()

    fields = adapter._extract_ztf_fields(
        {
            "alerts": [
                {
                    "i:jd": 2460001.5,
                    "i:candid": 123456789,
                    "i:ndethist": 4,
                    "d:snn_snia_vs_nonia": 0.8,
                    "d:snn_sn_vs_all": 0.9,
                    "d:rf_snia_vs_nonia": 0.2,
                }
            ]
        }
    )

    assert fields
    assert all(field["alert_id"] == 123456789 for field in fields)
    assert all(field["n_det"] == 4 for field in fields)
    assert all(field["event_time_jd"] == 2460001.5 for field in fields)
