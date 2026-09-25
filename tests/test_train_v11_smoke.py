"""End-to-end smoke test for the fusion_v11 orchestrator (P6).

Runs ``scripts/train_fusion_v11.main`` on a self-contained synthetic gold
snapshot (no ``data/`` access) via the ``--skip-stage-a`` path (q__ trust
columns pre-seeded), and asserts that the whole pipeline — hierarchical heads
(G7) → anchored blend (α, G3) → post-blend conformal → guards G2/G6 → gates →
reports — produces artifacts and a report JSON carrying every guard status.

Also covers the two orchestrator-owned refusal paths:
  * B0 ordering: Head-2 refuses to fit when ``--truth`` is absent.
  * G6-at-train: locked-benchmark ids leaking into train∪cal hard-fail.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_fusion_v11 import main

SEED = 7
_SANS = ("fink__snn", "alerce__stamp_classifier", "alerce_lc",
         "lasair__sherlock", "babamul")
_SN_CENTER = np.array([2.0, -1.0, 0.5, 1.0])
_OTHER_CENTER = np.array([-2.0, 1.5, -1.0, -0.5])
_IA_SHIFT = np.array([1.5, -1.0, 0.0, 0.8])


def _class_for(is_sn: int, is_ia: int) -> str:
    if not is_sn:
        return "other"
    return "snia" if is_ia else "nonIa_snlike"


def _expert_block(row: dict, rng: np.random.Generator, p_snia: float) -> None:
    """Pre-seed proj__/avail__/exact__/q__ so --skip-stage-a needs no Stage-A
    and the anchor has fired experts (compute_anchor keys off proj__+q__)."""
    for san in _SANS:
        if rng.random() < 0.75:
            row[f"proj__{san}__p_snia"] = float(np.clip(p_snia + rng.normal(0, 0.2), 0, 1))
            row[f"avail__{san}"] = 1.0
            row[f"exact__{san}"] = 1.0
            row[f"q__{san}"] = float(rng.uniform(0.4, 0.9))
        else:
            row[f"proj__{san}__p_snia"] = np.nan
            row[f"avail__{san}"] = 0.0
            row[f"exact__{san}"] = 0.0
            row[f"q__{san}"] = np.nan


def _build_gold(seed: int = SEED):
    """Synthetic v11 trust snapshot + object-level split.

    ZTF: multi-epoch, mostly spectroscopic (feeds head-1 isotonic + head-2 ZTF
    + conformal + α).  LSST: single-epoch, tiny spec cal (Platt fallback; keeps
    the LSST spec-Ia union < 10 so G2 is deterministically G2_UNEVALUABLE, the
    honest current-regime status — B1).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    split: dict[str, str] = {}

    def emit(oid, survey, is_sn, is_ia, quality, n_epochs, phase):
        cls = _class_for(is_sn, is_ia)
        subtype = ("SN Ia" if is_ia else "SN II")
        spec = quality == "spectroscopic"
        for n_det in range(1, n_epochs + 1):
            feats = (_SN_CENTER if is_sn else _OTHER_CENTER) \
                + (_IA_SHIFT if is_ia else 0.0) + rng.normal(0, 1.0, 4)
            row = {
                "object_id": oid, "n_det": n_det, "survey": survey,
                "target_class": cls, "label_quality": quality,
                "label_source": "tns" if spec else "weak_stamp",
                "tns_type": subtype if spec else None,
                "bts_type": subtype if spec else "-",
                "alert_jd": 2460000.0 + n_det,
                "survey_is_lsst": 1.0 if survey == "LSST" else 0.0,
                "traj_x__mean_slope": float(rng.normal()),
                "lc_all_nan": 0,
            }
            for j, v in enumerate(feats):
                row[f"lcf_{j}"] = float(v)
            # base-51-ish coverage column so the anchor's lc_cov bucket is real
            row["mag_mean"] = float(feats[0])
            _expert_block(row, rng, 0.8 if is_sn else 0.15)
            rows.append(row)
        split[oid] = phase

    zi = 0
    for phase, count in (("train", 130), ("cal", 70), ("test", 24)):
        for _ in range(count):
            is_sn = int(rng.random() < 0.6)
            is_ia = int(is_sn and rng.random() < 0.55)
            quality = "spectroscopic"
            if is_sn and phase == "train" and rng.random() < 0.15:
                quality = "weak"
            emit(f"ZTF{zi:05d}", "ZTF", is_sn, is_ia, quality,
                 int(rng.integers(4, 9)), phase)
            zi += 1

    li = 0
    for phase, count in (("train", 40), ("cal", 16), ("test", 12)):
        for k in range(count):
            is_sn = int(k % 2 == 0)
            is_ia = int(is_sn and rng.random() < 0.5)
            quality = "spectroscopic" if (is_sn and phase != "train") else (
                "weak" if is_sn else "context")
            emit(f"LSST{li:05d}", "LSST", is_sn, is_ia, quality, 1, phase)
            li += 1

    df = pd.DataFrame(rows)
    train_ids = {k for k, v in split.items() if v == "train"}
    cal_ids = {k for k, v in split.items() if v == "cal"}
    test_ids = {k for k, v in split.items() if v == "test"}
    return df, train_ids, cal_ids, test_ids


def _write_inputs(tmp_path: Path, *, locked_overlap: bool = False, with_truth: bool = True):
    df, tr, ca, te = _build_gold()
    snap = tmp_path / "snap_v11_trust.parquet"
    df.to_parquet(snap, index=False)

    split_path = tmp_path / "split_v11.json"
    # A properly-armed build stamps lsst_live_locked_armed=true (the split was
    # built WITH --lsst-live-locked). The orchestrator refuses an unarmed split
    # when a locked manifest exists (G6 counterpart quarantine would no-op).
    split_path.write_text(json.dumps(
        {"train_ids": sorted(tr), "cal_ids": sorted(ca), "test_ids": sorted(te),
         "lsst_live_locked_armed": True}))

    # locked benchmark: disjoint test ids (PASS) or a deliberate train leak (FAIL)
    locked_ids = (sorted(te)[:3] if not locked_overlap else sorted(tr)[:2])
    locked_path = tmp_path / "lsst_live_locked_test.json"
    locked_path.write_text(json.dumps(
        {"test_ids": locked_ids, "frozen_utc": "2026-07-04T00:00:00Z",
         "policy": "hash", "source": "smoke"}))

    truth_path = tmp_path / "object_truth_v11.parquet"
    if with_truth:
        pd.DataFrame({"object_id": sorted(tr | ca | te)}).to_parquet(truth_path, index=False)

    return snap, split_path, locked_path, truth_path


def _argv(tmp_path: Path, snap, split_path, locked_path, truth_path, *, extra=()):
    return [
        "--skip-stage-a", "--smoke", "--acknowledge-g2-unevaluable",
        "--output-snapshots", str(snap),
        "--split", str(split_path),
        "--lsst-locked", str(locked_path),
        "--truth", str(truth_path),
        "--followup-dir", str(tmp_path / "followup_v11"),
        "--blend-dir", str(tmp_path / "blend_v11"),
        "--conformal-dir", str(tmp_path / "conformal_v11"),
        "--trust-dir", str(tmp_path / "trust_v11"),
        "--metrics-out", str(tmp_path / "fusion_v11_train.json"),
        "--build-report", str(tmp_path / "no_such_build_report.json"),
        "--n-jobs", "2",
        *extra,
    ]


def test_smoke_end_to_end(tmp_path):
    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path)
    rc = main(_argv(tmp_path, snap, split_path, locked_path, truth_path))
    assert rc == 0, "acknowledged G2_UNEVALUABLE must exit 0"

    report = json.loads((tmp_path / "fusion_v11_train.json").read_text())

    # every guard the orchestrator owns is present with a status
    guards = report["guards"]
    for g in ("G2", "G3", "G6", "G7"):
        assert g in guards, f"missing guard {g}"
        assert "status" in guards[g], f"guard {g} has no status"
    statuses = report["guard_statuses"]
    assert statuses["G6"] == "PASS"
    assert statuses["G7"] == "PASS"
    assert statuses["G3"] in ("PASS", "FAIL")
    assert statuses["G2"] in ("PASS", "FAIL", "G2_UNEVALUABLE")

    # G1/G4 are eval-time-only — recorded, never computed in training
    assert set(report["eval_only_guards"]) == {"G1", "G4"}

    # gates surfaced from the head + the seq_v11 in/out verdict
    assert isinstance(report["gates"], list)
    assert any(
        isinstance(e, dict) and e.get("gate") == "seq_v11_in_out"
        for e in report["gates"])

    # reports block: availability audit, α ledger, per-cell n, coverage read
    assert "availability_audit" in report
    assert "cal_alpha_fallback_ledger" in report["blend"]
    assert "per_cell_n_after_honesty" in report["blend"]
    assert "empirical_coverage_cal" in report["conformal"]

    # artifacts on disk
    assert (tmp_path / "followup_v11" / "metadata.json").exists()
    assert (tmp_path / "blend_v11" / "blend.json").exists()
    assert (tmp_path / "conformal_v11" / "mondrian_aps.pkl").exists()


def test_b0_truth_guard_refuses_head2(tmp_path):
    """B0: Head-2 refuses to fit when object_truth_v11.parquet is absent."""
    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path, with_truth=False)
    assert not truth_path.exists()
    with pytest.raises(SystemExit):
        main(_argv(tmp_path, snap, split_path, locked_path, truth_path))


def test_g6_train_leak_hard_fails(tmp_path):
    """G6-at-train: a locked-benchmark id in train∪cal is a hard assert."""
    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path, locked_overlap=True)
    with pytest.raises(AssertionError):
        main(_argv(tmp_path, snap, split_path, locked_path, truth_path))


def test_g6_split_predates_manifest_refused(tmp_path):
    """A manifest exists but the split-in-force was built WITHOUT it
    (lsst_live_locked_armed absent) — the orchestrator refuses (SystemExit)."""
    snap, split_path, locked_path, truth_path = _write_inputs(tmp_path)
    # Strip the armed flag to simulate a pre-manifest split.
    raw = json.loads(split_path.read_text())
    raw.pop("lsst_live_locked_armed", None)
    split_path.write_text(json.dumps(raw))
    with pytest.raises(SystemExit):
        main(_argv(tmp_path, snap, split_path, locked_path, truth_path))
    # …but the acknowledge flag lets it continue past the setup refusal.
    rc = main(_argv(tmp_path, snap, split_path, locked_path, truth_path,
                    extra=("--acknowledge-split-predates-manifest",)))
    assert rc == 0
