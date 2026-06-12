"""Phase 2 — bulk pull Babamul per-object alerts for the existing ZTF cohort.

Uses BabamulAdapter.bulk_fetch_objects (threaded per-object GET) — this is
the right endpoint for the 6 alert features (properties.star, survey_matches,
etc). The cross-matches endpoint returns catalog enrichment (Gaia/TNS/NED),
NOT alert features — they're separate datasets.

Writes a single bronze parquet via the standard write_bronze plumbing so
silver/gold pick it up automatically.

Usage:
    python scripts/_exp_babamul_cohort_pull.py
    python scripts/_exp_babamul_cohort_pull.py --limit 100        # quick smoke
    python scripts/_exp_babamul_cohort_pull.py --bronze-dir data/bronze
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from debass_meta.access.babamul import BabamulAdapter
from debass_meta.access.base import BrokerOutput
from debass_meta.ingest.bronze import write_bronze


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="data/labels.csv")
    p.add_argument("--bronze-dir", default="data/bronze")
    p.add_argument("--limit", type=int, default=0,
                   help="0 = full cohort; >0 caps for a smoke run")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--n-threads", type=int, default=16)
    args = p.parse_args()

    df = pd.read_csv(REPO_ROOT / args.labels)
    ztf = df[df["object_id"].astype(str).str.startswith("ZTF")].copy()
    ids = ztf["object_id"].astype(str).tolist()
    if args.limit:
        ids = ids[: args.limit]
    print(f"cohort size: {len(ids)} ZTF labels")

    adapter = BabamulAdapter()
    probe = adapter.probe()
    if probe.get("status") != "ok":
        print(f"FAIL probe: {probe}")
        return 1
    print(f"probe ok: {probe}")

    t0 = time.time()
    payloads = adapter.bulk_fetch_objects(ids, survey="ZTF", n_threads=args.n_threads)
    dt = time.time() - t0
    n_hit = sum(1 for v in payloads.values() if v)
    pct = 100.0 * n_hit / max(len(ids), 1)
    print(
        f"per-object fetch: {n_hit}/{len(ids)} ({pct:.1f}%) hits in {dt:.1f}s "
        f"({len(ids) / max(dt, 1e-3):.0f} obj/s)"
    )

    if not n_hit:
        print("no hits; nothing to write")
        return 1

    # Build BrokerOutput records — both hits and 404-style misses, so the
    # silver/gold pipeline sees the full coverage truth instead of pretending
    # missing rows = unprobed.
    outputs: list[BrokerOutput] = []
    now = time.time()
    for oid, payload in payloads.items():
        if payload:
            events = adapter._extract_event(payload, oid, "ZTF")
            outputs.append(BrokerOutput(
                broker=adapter.name,
                object_id=oid,
                query_time=now,
                raw_payload=payload,
                semantic_type=adapter.semantic_type,
                survey="ZTF",
                source_endpoint=f"{adapter._base}/surveys/ZTF/objects/cross-matches",
                request_params={"bulk": True},
                status_code=200,
                fields=events,
                events=events,
                availability=bool(events),
                fixture_used=False,
            ))
        else:
            outputs.append(adapter.unavailable_output(
                oid,
                source_endpoint=f"{adapter._base}/surveys/ZTF/objects/cross-matches",
                survey="ZTF",
                identifier_kind="ztf_object_id",
                reason="not_indexed_by_babamul",
            ))

    out_path = write_bronze(outputs, bronze_dir=REPO_ROOT / args.bronze_dir)
    print(f"wrote {len(outputs)} records → {out_path}")

    # Summary stats on the 6 features for hits
    print("\n=== feature value distribution (hits only) ===")
    rows = []
    for o in outputs:
        if not o.availability:
            continue
        rows.append({e["field"]: e["raw_label_or_score"] for e in o.events})
    if rows:
        feat_df = pd.DataFrame(rows)
        for col in feat_df.columns:
            vc = feat_df[col].value_counts(dropna=False)
            print(f"  {col}: {dict(vc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
