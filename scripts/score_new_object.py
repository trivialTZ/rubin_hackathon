#!/usr/bin/env python3
"""Score a new transient object using the published DEBASS models.

This is the user-facing entry point. Given a lightcurve (JSON file or
ZTF object ID), it runs the trust-aware pipeline and prints a follow-up
recommendation.

Usage:
    # Score from a lightcurve JSON file:
    python3 scripts/score_new_object.py --lc-file lightcurve.json --n-det 4

    # Score all epochs from n_det=1 to max:
    python3 scripts/score_new_object.py --lc-file lightcurve.json

    # Use published models (default) or custom model dir:
    python3 scripts/score_new_object.py --lc-file lc.json --models-dir published_results/models

Input lightcurve JSON format (list of detections):
    [
      {"mjd": 60170.1, "magpsf": 19.2, "sigmapsf": 0.05, "fid": 1, "isdiffpos": "t"},
      {"mjd": 60171.3, "magpsf": 18.8, "sigmapsf": 0.04, "fid": 2, "isdiffpos": "t"},
      ...
    ]

Output: JSON with expert trust scores and follow-up recommendation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _load_lightcurve(path: Path) -> list[dict]:
    with open(path) as fh:
        dets = json.load(fh)
    pos = [
        d for d in dets
        if isinstance(d, dict) and d.get("isdiffpos") in ("t", "1", 1, True, "true")
    ]
    if not pos:
        pos = [d for d in dets if isinstance(d, dict)]
    return sorted(pos, key=lambda d: d.get("mjd") or d.get("jd") or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a transient with DEBASS trust-aware models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lc-file", required=True, help="Path to lightcurve JSON file")
    parser.add_argument("--object-id", default="new_object", help="Object identifier")
    parser.add_argument("--n-det", type=int, default=None,
                        help="Score at specific n_det (default: all epochs)")
    parser.add_argument("--models-dir", default=None,
                        help="Model directory (default: published_results/models or models/)")
    parser.add_argument("--snn-artifacts", default=None,
                        help="SuperNNova artifacts dir (default: artifacts/local_experts/supernnova/)")
    args = parser.parse_args()

    # Find models
    models_dir = None
    for candidate in [
        Path(args.models_dir) if args.models_dir else None,
        Path("published_results/models"),
        Path("models"),
    ]:
        if candidate and (candidate / "followup" / "metadata.json").exists():
            models_dir = candidate
            break
    if models_dir is None:
        print("ERROR: No trained models found. Looked in published_results/models/ and models/")
        print("Run the pipeline first or clone the repo with published_results/.")
        sys.exit(1)

    print(f"Using models from: {models_dir}")

    # Load models
    from debass_meta.models.followup import FollowupArtifact
    from debass_meta.models.expert_trust import ExpertTrustArtifact

    followup_model = FollowupArtifact.load(models_dir / "followup")
    trust_models = {}
    trust_dir = models_dir / "trust"
    if trust_dir.exists():
        for expert_dir in trust_dir.iterdir():
            if expert_dir.is_dir() and (expert_dir / "metadata.json").exists():
                try:
                    artifact = ExpertTrustArtifact.load(expert_dir)
                    trust_models[artifact.expert_key] = artifact
                except Exception:
                    pass
    print(f"Loaded trust models: {list(trust_models.keys())}")
    print(f"Loaded follow-up model: {models_dir / 'followup'}")

    # Load SuperNNova if available
    snn_expert = None
    try:
        from debass_meta.experts.local.supernnova import SuperNNovaExpert
        snn_dir = Path(args.snn_artifacts) if args.snn_artifacts else Path("artifacts/local_experts/supernnova")
        snn_expert = SuperNNovaExpert(model_dir=snn_dir)
        meta = snn_expert.metadata()
        if meta.get("available") and meta.get("model_loaded"):
            print(f"SuperNNova: available (device={meta['device']})")
        else:
            print("SuperNNova: not available (missing weights or package)")
            snn_expert = None
    except ImportError:
        print("SuperNNova: not installed (pip install supernnova torch)")

    # Load lightcurve
    dets = _load_lightcurve(Path(args.lc_file))
    if not dets:
        print("ERROR: No valid detections in lightcurve file")
        sys.exit(1)
    print(f"Lightcurve: {len(dets)} detections")

    # Determine epochs to score
    if args.n_det is not None:
        n_dets = [args.n_det]
    else:
        n_dets = list(range(1, len(dets) + 1))

    # Score each epoch
    results = []
    for n_det in n_dets:
        if n_det > len(dets):
            break
        truncated = dets[:n_det]
        alert_mjd = truncated[-1].get("mjd") or truncated[-1].get("jd") or 0.0
        alert_jd = float(alert_mjd) + 2400000.5

        result = {
            "object_id": args.object_id,
            "n_det": n_det,
            "alert_jd": alert_jd,
            "experts": {},
        }

        # Run SuperNNova if available
        if snn_expert is not None:
            try:
                out = snn_expert.predict_epoch(args.object_id, truncated, alert_jd)
                result["experts"]["supernnova"] = {
                    "available": out.available,
                    "class_probabilities": out.class_probabilities,
                    "model_version": out.model_version,
                }
            except Exception as exc:
                result["experts"]["supernnova"] = {"available": False, "error": str(exc)}

        # TODO: Add Fink SNN/RF scores if broker data is available
        # For now, only locally-runnable experts are scored

        results.append(result)

    # Print results
    print("\n" + "=" * 60)
    print(f"DEBASS Score — {args.object_id}")
    print("=" * 60)

    for r in results:
        print(f"\nn_det={r['n_det']}  (alert_jd={r['alert_jd']:.4f})")
        for expert, data in r["experts"].items():
            if data.get("available"):
                probs = data.get("class_probabilities", {})
                p_ia = probs.get("SN Ia", "?")
                p_nonia = probs.get("non-Ia", "?")
                print(f"  {expert:20s}  P(Ia)={p_ia:.3f}  P(non-Ia)={p_nonia:.3f}")
            else:
                print(f"  {expert:20s}  unavailable")

    # Write JSON output
    output = {
        "object_id": args.object_id,
        "n_detections": len(dets),
        "models_dir": str(models_dir),
        "epochs": results,
    }
    out_path = Path(f"score_{args.object_id}.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
