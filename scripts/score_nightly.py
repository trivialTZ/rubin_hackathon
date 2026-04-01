"""Emit the trust-aware nightly scoring payload."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from debass_meta.models.expert_trust import ExpertTrustArtifact
from debass_meta.models.followup import FollowupArtifact
from debass_meta.projectors import PHASE1_EXPERT_KEYS, sanitize_expert_key


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


def build_score_payload(row, trust_models: dict[str, ExpertTrustArtifact], followup_model: FollowupArtifact | None, *, recommend_threshold: float = 0.5) -> dict:
    payload = {
        "object_id": row["object_id"],
        "n_det": int(row["n_det"]),
        "alert_jd": float(row["alert_jd"]),
        "expert_confidence": {},
    }
    row_df = pd.DataFrame([row.to_dict()])

    for expert_key in PHASE1_EXPERT_KEYS:
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

    ensemble = {"p_follow_proxy": None, "recommended": None}
    if followup_model is not None:
        p_follow = float(followup_model.predict_followup(row_df)[0])
        ensemble["p_follow_proxy"] = p_follow
        ensemble["recommended"] = bool(p_follow >= recommend_threshold)
    payload["ensemble"] = ensemble
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Score trust-aware nightly payloads")
    parser.add_argument("--snapshots", default="data/gold/object_epoch_snapshots_trust.parquet")
    parser.add_argument("--trust-models-dir", default="models/trust")
    parser.add_argument("--followup-model-dir", default="models/followup")
    parser.add_argument("--object", default=None)
    parser.add_argument("--from-labels", default=None)
    parser.add_argument("--n-det", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    parser.add_argument("--recommend-threshold", type=float, default=0.5)
    args = parser.parse_args()

    snapshot_df = pd.read_parquet(args.snapshots)
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
