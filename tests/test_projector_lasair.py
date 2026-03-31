from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debass_meta.projectors.lasair import project_events


def test_lasair_projector_emits_context_one_hot() -> None:
    result = project_events("lasair/sherlock", [{"class_name": "SN"}])

    assert result["prediction_type"] == "context_only"
    assert result["context_tag"] == "SN"
    assert result["context_SN"] == 1.0
    assert result["context_AGN"] == 0.0
