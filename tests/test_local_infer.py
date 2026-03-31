from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_local_infer_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "local_infer.py"
    spec = importlib.util.spec_from_file_location("local_infer_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_merge_records_refreshes_existing_predictions() -> None:
    module = _load_local_infer_module()

    existing = [
        {"expert": "parsnip", "object_id": "ZTF1", "n_det": 1, "class_probabilities": {"SN Ia": 0.2}},
        {"expert": "parsnip", "object_id": "ZTF1", "n_det": 2, "class_probabilities": {"SN Ia": 0.3}},
    ]
    new_records = [
        {"expert": "parsnip", "object_id": "ZTF1", "n_det": 1, "class_probabilities": {"SN Ia": 0.9}},
        {"expert": "parsnip", "object_id": "ZTF2", "n_det": 1, "class_probabilities": {"SN Ia": 0.7}},
    ]

    combined, appended, refreshed = module._merge_records(existing, new_records)
    lookup = {(rec["expert"], rec["object_id"], rec["n_det"]): rec for rec in combined}

    assert appended == 1
    assert refreshed == 1
    assert len(combined) == 3
    assert lookup[("parsnip", "ZTF1", 1)]["class_probabilities"]["SN Ia"] == 0.9
    assert lookup[("parsnip", "ZTF2", 1)]["class_probabilities"]["SN Ia"] == 0.7
