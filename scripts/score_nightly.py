"""Emit the trust-aware nightly scoring payload."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from debass_meta.models.expert_trust import ExpertTrustArtifact
from debass_meta.models.followup import FollowupArtifact
from debass_meta.projectors import ALL_EXPERT_KEYS, PHASE1_EXPERT_KEYS, sanitize_expert_key


def _clean_scalar(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return value


def _load_object_ids(path: Path | None) -> list[str] | None:
    if path is None or not path.exists():
        return None
    with open(path) as fh:
        return [row["object_id"] for row in csv.DictReader(fh)]


def _load_trust_models(models_dir: Path) -> dict[str, ExpertTrustArtifact]:
    models: dict[str, ExpertTrustArtifact] = {}
    if not models_dir.exists():
        return models
    for expert_dir in models_dir.iterdir():
        if not expert_dir.is_dir() or not (expert_dir / "metadata.json").exists():
            continue
        artifact = ExpertTrustArtifact.load(expert_dir)
        models[artifact.expert_key] = artifact
    return models


def _resolve_snapshot_path(snapshot_arg: str | None) -> Path:
    if snapshot_arg:
        return Path(snapshot_arg).resolve()
    candidates = [
        Path("data/gold/object_epoch_snapshots_scoring.parquet"),
        Path("data/gold/object_epoch_snapshots_trust.parquet"),
        Path("data/gold/object_epoch_snapshots.parquet"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "No snapshot parquet found. Looked for scoring, trust, and canonical gold snapshots."
    )


def build_score_payload(row, trust_models: dict[str, ExpertTrustArtifact], followup_model: FollowupArtifact | None, *, recommend_threshold: float = 0.5) -> dict:
    payload = {
        "object_id": row["object_id"],
        "n_det": int(row["n_det"]),
        "alert_jd": float(row["alert_jd"]),
        "expert_confidence": {},
    }
    row_df = pd.DataFrame([row.to_dict()])

    for expert_key in ALL_EXPERT_KEYS:
        san = sanitize_expert_key(expert_key)
        available = bool(row.get(f"avail__{san}", 0))
        block = {
            "available": available,
            "trust": None,
            "prediction_type": _clean_scalar(row.get(f"prediction_type__{san}")),
            "exactness": _clean_scalar(row.get(f"temporal_exactness__{san}")),
        }
        if not available:
            block["reason"] = row.get(f"reason__{san}") or "expert unavailable at this epoch"
            payload["expert_confidence"][expert_key] = block
            continue

        if row.get(f"prediction_type__{san}") == "context_only":
            block["context_tag"] = _clean_scalar(row.get(f"context_tag__{san}"))
            block["reason"] = row.get(f"reason__{san}") or "phase-1 context expert"
            payload["expert_confidence"][expert_key] = block
            continue

        trust_col = f"q__{san}"
        if row.get(trust_col) is not None and not pd.isna(row.get(trust_col)):
            block["trust"] = float(row[trust_col])
        elif expert_key in trust_models:
            block["trust"] = float(trust_models[expert_key].predict_trust(row_df)[0])
        else:
            block["reason"] = "trust head not trained"

        mapped_pred_class = _clean_scalar(row.get(f"mapped_pred_class__{san}"))
        if mapped_pred_class is not None:
            block["mapped_pred_class"] = mapped_pred_class
        projected = {}
        for key in ("p_snia", "p_nonIa_snlike", "p_other", "top1_prob", "margin", "entropy", "p_snia_scalar"):
            column = f"proj__{san}__{key}"
            if column in row and row.get(column) is not None and not pd.isna(row.get(column)):
                projected[key] = float(row[column])
        if projected:
            block["projected"] = projected
        payload["expert_confidence"][expert_key] = block

    # --- Explicit trust-weighted ensemble (interpretable) ---
    tw_weights: list[float] = []
    tw_p_snia: list[float] = []
    tw_p_nonia: list[float] = []
    tw_p_other: list[float] = []
    for expert_key in ALL_EXPERT_KEYS:
        block = payload["expert_confidence"].get(expert_key, {})
        if not block.get("available"):
            continue
        trust_val = block.get("trust")
        projected = block.get("projected", {})
        if trust_val is None or not projected:
            continue
        w = float(trust_val)
        if w <= 0:
            continue
        p_s = projected.get("p_snia")
        p_n = projected.get("p_nonIa_snlike")
        p_o = projected.get("p_other")
        # Only include experts with full ternary output — zeroing missing
        # channels biases the ensemble toward p_snia for scalar-only experts.
        if p_s is not None and p_n is not None and p_o is not None:
            tw_weights.append(w)
            tw_p_snia.append(w * float(p_s))
            tw_p_nonia.append(w * float(p_n))
            tw_p_other.append(w * float(p_o))

    ensemble: dict[str, Any] = {"p_follow_proxy": None, "recommended": None}
    if tw_weights:
        total_w = sum(tw_weights)
        ensemble["p_snia_weighted"] = sum(tw_p_snia) / total_w
        ensemble["p_nonIa_snlike_weighted"] = sum(tw_p_nonia) / total_w
        ensemble["p_other_weighted"] = sum(tw_p_other) / total_w
        ensemble["n_trusted_experts"] = len(tw_weights)

    if followup_model is not None:
        p_follow = float(followup_model.predict_followup(row_df)[0])
        ensemble["p_follow_proxy"] = p_follow
        ensemble["recommended"] = bool(p_follow >= recommend_threshold)
    payload["ensemble"] = ensemble
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Score trust-aware nightly payloads")
    parser.add_argument("--snapshots", default=None,
                        help="Snapshot parquet. If omitted, prefer scoring snapshots, then trust, then canonical gold.")
    parser.add_argument("--trust-models-dir", default="models/trust")
    parser.add_argument("--followup-model-dir", default="models/followup")
    parser.add_argument("--object", default=None)
    parser.add_argument("--from-labels", default=None)
    parser.add_argument("--n-det", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    parser.add_argument("--recommend-threshold", type=float, default=0.5)
    args = parser.parse_args()

    snapshot_path = _resolve_snapshot_path(args.snapshots)
    snapshot_df = pd.read_parquet(snapshot_path)
    trust_models = _load_trust_models(Path(args.trust_models_dir))
    followup_model = FollowupArtifact.load(Path(args.followup_model_dir)) if (Path(args.followup_model_dir) / "metadata.json").exists() else None

    object_ids = None
    if args.object:
        object_ids = [args.object]
    elif args.from_labels:
        object_ids = _load_object_ids(Path(args.from_labels))

    if object_ids is not None:
        snapshot_df = snapshot_df[snapshot_df["object_id"].isin(object_ids)]
    if args.n_det is not None:
        snapshot_df = snapshot_df[snapshot_df["n_det"] == args.n_det]

    payloads = [
        build_score_payload(row, trust_models, followup_model, recommend_threshold=args.recommend_threshold)
        for _, row in snapshot_df.iterrows()
    ]

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            for payload in payloads:
                fh.write(json.dumps(payload) + "\n")
        print(f"Wrote scores → {out_path}")
    else:
        for payload in payloads:
            print(json.dumps(payload))


if __name__ == "__main__":
    main()
