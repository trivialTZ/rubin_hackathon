"""Backfill ALeRCE LSST Rubin stamp classifier probabilities for LSST objects.

Endpoint: GET https://api-lsst.alerce.online/probability_api/probability?oid=<id>
No auth required. Parallel-safe. Retry on 502/503/504 with backoff.

Output: data/truth_candidates/alerce_stamp_lsst.parquet
  columns: object_id, stamp_class, stamp_prob, stamp_classifier_version, query_time, all_probs_json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from debass_meta.access.identifiers import infer_identifier_kind

_BASE = "https://api-lsst.alerce.online/probability_api/probability"
_RETRY_DELAYS = (2.0, 5.0, 12.0)


_RETRY_STATUS = {429, 403, 500, 502, 503, 504}


def fetch_one(oid: str, *, timeout: int = 20) -> dict:
    url = f"{_BASE}?oid={oid}"
    for attempt, delay in enumerate([0.0, *_RETRY_DELAYS]):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                rows = json.loads(r.read())
            if not isinstance(rows, list) or not rows:
                return {"object_id": oid, "stamp_class": None, "stamp_prob": None,
                        "stamp_classifier_version": None, "query_time": time.time(),
                        "all_probs_json": json.dumps([]), "error": "empty"}
            rows.sort(key=lambda x: -float(x.get("probability", 0) or 0))
            top = rows[0]
            return {
                "object_id": oid,
                "stamp_class": top.get("class_name"),
                "stamp_prob": float(top.get("probability", 0) or 0),
                "stamp_classifier_version": top.get("classifier_name"),
                "query_time": time.time(),
                "all_probs_json": json.dumps(rows),
                "error": None,
            }
        except urllib.error.HTTPError as e:
            if e.code in _RETRY_STATUS and attempt < len(_RETRY_DELAYS):
                continue
            return {"object_id": oid, "stamp_class": None, "stamp_prob": None,
                    "stamp_classifier_version": None, "query_time": time.time(),
                    "all_probs_json": "[]", "error": f"HTTP {e.code}"}
        except Exception as e:
            if attempt < len(_RETRY_DELAYS):
                continue
            return {"object_id": oid, "stamp_class": None, "stamp_prob": None,
                    "stamp_classifier_version": None, "query_time": time.time(),
                    "all_probs_json": "[]", "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--output", default="data/truth_candidates/alerce_stamp_lsst.parquet")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="If output exists, only fetch oids that previously errored / missing stamp_class.")
    args = ap.parse_args()

    d = pd.read_csv(args.labels)
    d["id_kind"] = d["object_id"].astype(str).apply(infer_identifier_kind)
    all_lsst_ids = d[d["id_kind"] == "lsst_dia_object_id"]["object_id"].astype(str).tolist()
    print(f"Total LSST objects: {len(all_lsst_ids)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = None
    if args.resume and out_path.exists():
        existing_df = pd.read_parquet(out_path)
        existing_df["object_id"] = existing_df["object_id"].astype(str)
        successful = set(existing_df[existing_df["stamp_class"].notna()]["object_id"])
        print(f"Resume mode: {len(successful)} oids already have stamp_class; fetching the rest.")
        lsst_ids = [o for o in all_lsst_ids if o not in successful]
    else:
        lsst_ids = all_lsst_ids
    print(f"Objects to query this run: {len(lsst_ids)}")
    if not lsst_ids:
        print("nothing to do")
        return

    rows = []
    done = 0
    errors = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for r in ex.map(fetch_one, lsst_ids):
            rows.append(r)
            done += 1
            if r.get("error"):
                errors += 1
            if done % 200 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                eta_min = (len(lsst_ids) - done) / max(rate, 1e-6) / 60
                print(f"  progress: {done}/{len(lsst_ids)}  errors={errors}  rate={rate:.1f}/s  ETA={eta_min:.1f} min",
                      flush=True)

    new_df = pd.DataFrame(rows)
    if existing_df is not None:
        # Keep previously successful rows; replace failed/missing ones with the new fetch
        already_ok = existing_df[existing_df["stamp_class"].notna()]
        df = pd.concat([already_ok, new_df], ignore_index=True, sort=False)
    else:
        df = new_df
    df.to_parquet(out_path, index=False)
    print(f"\nwrote {len(df)} rows -> {out_path}")
    print(f"success: {df['stamp_class'].notna().sum()} / {len(df)}")
    print(f"class dist:")
    print(df[df['stamp_class'].notna()]['stamp_class'].value_counts().to_string())
    print(f"conf >= 0.85: {(df['stamp_prob'] >= 0.85).sum()}")
    print(f"conf >= 0.70: {(df['stamp_prob'] >= 0.70).sum()}")
    if errors:
        print(f"\nERRORS: {errors}")


if __name__ == "__main__":
    main()
