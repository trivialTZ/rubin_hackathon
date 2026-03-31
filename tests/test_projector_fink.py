from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass.projectors.fink import project_events


def test_fink_snn_projector_computes_ternary_projection() -> None:
    result = project_events(
        "fink/snn",
        [
            {"field": "snn_snia_vs_nonia", "canonical_projection": 0.6},
            {"field": "snn_sn_vs_all", "canonical_projection": 0.8},
        ],
    )

    assert math.isclose(result["p_snia"], 0.48, rel_tol=1e-6)
    assert math.isclose(result["p_nonIa_snlike"], 0.32, rel_tol=1e-6)
    assert math.isclose(result["p_other"], 0.2, rel_tol=1e-6)
    assert result["mapped_pred_class"] == "snia"
