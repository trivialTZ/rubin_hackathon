"""scripts/summarize_nightly.py — Summarize nightly scores into a ranked table.

Reads JSONL score files, ranks by follow-up probability, and outputs a
human-readable summary of the top candidates.

Usage:
    python scripts/summarize_nightly.py --scores-dir data/nightly/2026-04-02/scores/
    python scripts/summarize_nightly.py --scores-dir data/nightly/2026-04-02/scores/ --top 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_scores(scores_dir: Path) -> list[dict]:
    """Load all JSONL score files from a directory."""
    payloads = []
    for jsonl_path in sorted(scores_dir.glob("scores_ndet*.jsonl")):
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    payloads.append(json.loads(line))
    return payloads


def summarize_payload(payload: dict) -> dict:
    """Extract key summary fields from a score payload."""
    experts = payload.get("expert_confidence", {})
    n_available = sum(1 for e in experts.values() if e.get("available"))
    ensemble = payload.get("ensemble", {})
    p_follow = ensemble.get("p_follow_proxy")

    # Find highest-trust available expert
    top_expert = None
    top_trust = -1
    for key, block in experts.items():
        if block.get("available") and block.get("trust") is not None:
            if block["trust"] > top_trust:
                top_trust = block["trust"]
                top_expert = key

    # Get SN Ia probability from highest-trust expert
    p_snia = None
    if top_expert and experts[top_expert].get("projected"):
        p_snia = experts[top_expert]["projected"].get("p_snia")

    return {
        "object_id": payload["object_id"],
        "n_det": payload["n_det"],
        "p_follow": p_follow,
        "recommended": ensemble.get("recommended"),
        "n_experts": n_available,
        "top_expert": top_expert or "-",
        "top_trust": top_trust if top_trust >= 0 else None,
        "p_snia": p_snia,
    }


def format_table(summaries: list[dict], top: int = 100) -> str:
    """Format summaries as a readable text table."""
    # Sort by p_follow descending (None goes to bottom)
    summaries.sort(key=lambda s: (s["p_follow"] is not None, s["p_follow"] or 0),
                   reverse=True)

    lines = []
    lines.append(f"DEBASS Nightly Follow-up Candidates (top {min(top, len(summaries))} of {len(summaries)})")
    lines.append("=" * 110)
    header = (
        f"{'Rank':>4}  {'Object ID':<22} {'n_det':>5}  {'p_follow':>8}  "
        f"{'Rec':>3}  {'Experts':>7}  {'Top Expert':<20}  {'Trust':>5}  {'p(SNIa)':>7}"
    )
    lines.append(header)
    lines.append("-" * 110)

    for i, s in enumerate(summaries[:top], 1):
        p_follow_str = f"{s['p_follow']:.3f}" if s['p_follow'] is not None else "  -"
        rec_str = "Y" if s['recommended'] else ("N" if s['recommended'] is not None else "-")
        trust_str = f"{s['top_trust']:.2f}" if s['top_trust'] is not None else "  -"
        p_snia_str = f"{s['p_snia']:.3f}" if s['p_snia'] is not None else "  -"

        lines.append(
            f"{i:>4}  {s['object_id']:<22} {s['n_det']:>5}  {p_follow_str:>8}  "
            f"{rec_str:>3}  {s['n_experts']:>7}  {s['top_expert']:<20}  {trust_str:>5}  {p_snia_str:>7}"
        )

    lines.append("-" * 110)

    # Quick stats
    n_recommended = sum(1 for s in summaries if s.get("recommended"))
    lines.append(f"\nTotal scored: {len(summaries)}")
    lines.append(f"Recommended for follow-up: {n_recommended}")

    unique_objects = len(set(s["object_id"] for s in summaries))
    lines.append(f"Unique objects: {unique_objects}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize nightly scores")
    parser.add_argument("--scores-dir", required=True,
                        help="Directory containing scores_ndet*.jsonl files")
    parser.add_argument("--output", default=None,
                        help="Output summary file (default: print to stdout)")
    parser.add_argument("--top", type=int, default=100,
                        help="Show top N candidates (default: 100)")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    if not scores_dir.exists():
        print(f"Scores directory not found: {scores_dir}", file=sys.stderr)
        sys.exit(1)

    payloads = load_scores(scores_dir)
    if not payloads:
        msg = f"No scores found in {scores_dir}"
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(msg + "\n")
        print(msg)
        sys.exit(0)

    summaries = [summarize_payload(p) for p in payloads]
    table = format_table(summaries, top=args.top)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table + "\n")
        print(f"Summary written → {out_path}")
    else:
        print(table)


if __name__ == "__main__":
    main()
