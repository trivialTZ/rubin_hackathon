from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.watch_cpu_prep import _has_failure, _log_phase_hint


def test_build_log_failure_is_detected(tmp_path: Path) -> None:
    log_path = tmp_path / "build.log"
    log_path.write_text(
        "Bronze → Silver...\n"
        "Traceback (most recent call last):\n"
        "pyarrow.lib.ArrowInvalid: bad column\n"
    )

    assert _has_failure(log_path) is True
    assert _log_phase_hint(log_path, "build") == "failed"


def test_build_log_phase_hint_tracks_backfill_and_normalize(tmp_path: Path) -> None:
    backfill_log = tmp_path / "backfill.log"
    backfill_log.write_text("[lasair] wrote 739 records → data/bronze/lasair.parquet\n")
    assert _log_phase_hint(backfill_log, "build") == "backfill"

    normalize_log = tmp_path / "normalize.log"
    normalize_log.write_text("Bronze → Silver...\n")
    assert _log_phase_hint(normalize_log, "build") == "normalize"
