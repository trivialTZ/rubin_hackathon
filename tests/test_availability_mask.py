"""Expert-group masking used by the input-availability ablation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from debass_meta.features.availability import (
    BROKER_EXPERTS,
    LOCAL_EXPERTS,
    expert_of_column,
    mask_experts,
    mask_group,
)


def _gold() -> pd.DataFrame:
    return pd.DataFrame({
        "object_id": ["a", "b"],
        "n_det": [3, 4],
        "mag_last": [19.0, 20.0],
        "survey_is_lsst": [1.0, 1.0],
        "proj__fink_lsst__snn__p_snia": [0.4, 0.2],
        "avail__fink_lsst__snn": [1.0, 1.0],
        "traj__fink_lsst__snn__last": [0.8, 0.3],
        "mapped_pred_class__fink_lsst__snn": ["nonIa_snlike", "other"],
        "proj__alerce__stamp_classifier__p_other": [0.1, 0.9],
        "proj__alerce__stamp_classifier_rubin_beta__p_other": [0.2, 0.7],
        "avail__alerce__stamp_classifier_rubin_beta": [1.0, 1.0],
        "proj__supernnova__p_snia": [0.6, 0.1],
        "avail__supernnova": [1.0, 1.0],
        "proj__ampel__snguess__p_nonIa_snlike": [0.5, 0.5],
        "avail__ampel__snguess": [1.0, 0.0],
        "q__seq_v11": [0.9, 0.2],
        "traj_x__mean_slope": [0.1, -0.1],
    })


def test_groups_partition_registry():
    assert "supernnova" in LOCAL_EXPERTS and "seq_v11" in LOCAL_EXPERTS and "ampel/snguess" in LOCAL_EXPERTS
    assert "fink_lsst/snn" in BROKER_EXPERTS and "babamul" in BROKER_EXPERTS
    assert not (LOCAL_EXPERTS & BROKER_EXPERTS)


def test_expert_of_column_prefers_longest_key():
    assert expert_of_column("proj__alerce__stamp_classifier__p_other") == "alerce/stamp_classifier"
    assert expert_of_column("proj__alerce__stamp_classifier_rubin_beta__p_other") == "alerce/stamp_classifier_rubin_beta"
    assert expert_of_column("avail__ampel__snguess") == "ampel/snguess"
    assert expert_of_column("proj__fink_lsst__snn__p_snia") == "fink_lsst/snn"
    assert expert_of_column("q__seq_v11") == "seq_v11"
    assert expert_of_column("mag_last") is None
    assert expert_of_column("traj_x__mean_slope") is None


def test_mask_brokers_keeps_local_and_lightcurve():
    g = _gold()
    m = mask_group(g, "brokers")
    assert m["proj__fink_lsst__snn__p_snia"].isna().all()
    assert (m["avail__fink_lsst__snn"] == 0).all()
    assert m["mapped_pred_class__fink_lsst__snn"].isna().all()
    assert m["traj_x__mean_slope"].isna().all()          # cross-expert trajectories go with the brokers
    assert m["proj__alerce__stamp_classifier_rubin_beta__p_other"].isna().all()
    np.testing.assert_array_equal(m["proj__supernnova__p_snia"], g["proj__supernnova__p_snia"])
    np.testing.assert_array_equal(m["q__seq_v11"], g["q__seq_v11"])
    np.testing.assert_array_equal(m["mag_last"], g["mag_last"])
    assert g["proj__fink_lsst__snn__p_snia"].notna().all()  # input untouched


def test_mask_local_and_rows():
    g = _gold()
    m = mask_group(g, "local")
    assert m["proj__supernnova__p_snia"].isna().all() and (m["avail__ampel__snguess"] == 0).all()
    assert m["q__seq_v11"].isna().all()
    assert m["traj_x__mean_slope"].notna().all()
    r = mask_experts(g, {"fink_lsst/snn"}, rows=np.array([True, False]))
    assert np.isnan(r.loc[0, "proj__fink_lsst__snn__p_snia"]) and r.loc[1, "proj__fink_lsst__snn__p_snia"] == 0.2
    assert r.loc[0, "avail__fink_lsst__snn"] == 0 and r.loc[1, "avail__fink_lsst__snn"] == 1
