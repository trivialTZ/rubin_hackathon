"""fusion v13j: guard G2 on P(Ia|SN) (--g2-metric p_ia_given_sn); the default stays the deployed p_snia."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.train_fusion_v11 import evaluate_g2


def _frame(p_sn: float, ia_share: float, n: int = 12) -> pd.DataFrame:
    return pd.DataFrame({
        "object_id": [f"o{i}" for i in range(n)], "n_det": 5, "survey": "lsst", "target_class": "snia",
        "label_quality": "spectroscopic", "label_source": "tns",
        "p_snia": p_sn * ia_share, "p_nonia": p_sn * (1 - ia_share), "p_other": 1 - p_sn,
    })


def test_low_p_sn_fails_p_snia_but_passes_on_the_ia_axis():
    df = _frame(p_sn=0.25, ia_share=0.55)               # p_snia 0.1375 < 0.15
    ids = set(df.object_id)
    old = evaluate_g2(df, ids, set(), acknowledge_unevaluable=False)
    new = evaluate_g2(df, ids, set(), acknowledge_unevaluable=False, metric="p_ia_given_sn")
    assert old["status"] == "FAIL" and "metric" not in old
    assert new["status"] == "PASS"
    assert new["median_p_snia"] == pytest.approx(0.1375)
    assert new["median_p_ia_given_sn"] == pytest.approx(0.55)


def test_ia_collapse_still_fails():
    df = _frame(p_sn=0.9, ia_share=0.05)               # SN-confident, Ia suppressed
    out = evaluate_g2(df, set(df.object_id), set(), acknowledge_unevaluable=False, metric="p_ia_given_sn")
    assert out["status"] == "FAIL"
    with pytest.raises(ValueError):
        evaluate_g2(df, set(df.object_id), set(), acknowledge_unevaluable=False, metric="bogus")
