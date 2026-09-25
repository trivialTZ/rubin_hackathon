#!/usr/bin/env bash
# refresh_lsst_live.sh — weekly LSST-live truth refresh + benchmark re-score.
#
# fusion_v11 P1 ops tool. Makes metaDEBASS a CONTINUOUSLY-evaluated system:
#   1. TNS bulk refresh  (full master or daily-diff upsert)
#   2. epoch-aware truth rebuild (live seed + cohort re-clean)
#   3. route post-freeze spec arrivals by sha1(object_id) hash (B1):
#        even hex -> TEST (appended to the frozen manifest), odd -> train/cal.
#        Frozen ids NEVER move.
#   4. re-score the locked benchmark (RUNBOOK SCORING flow, spec §7):
#        empty locked-split JSON as --trust-metadata, cohort-only --labels CSV,
#        --bts "", --no-lsst-weak, --skip-traj, NEVER --smoke.
#   5. append one benchmark row.
#
# This is a SCORING flow, not a training build — it never touches locked
# artifacts and writes only *_v11 / dated-report paths.
#
# Usage:
#   bash jobs/refresh_lsst_live.sh [--mode auto|full|diff] [--diff-date YYYYMMDD]
set -euo pipefail

REPO="${DEBASS_ROOT:-/Users/tz/Documents/GitHub/rubin_hackathon}"
cd "$REPO"

# venv (RUNBOOK §0): nothing works outside it.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${DEBASS_VENV:-$HOME/.venvs/debass_py313}/bin/activate"
fi

TNS_MODE="auto"
DIFF_DATE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) TNS_MODE="$2"; shift 2 ;;
    --diff-date) DIFF_DATE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TNS_PARQUET="data/truth/tns_public.parquet"
LIVE_TRUTH="data/truth/lsst_live_truth.parquet"
CLEANED_TRUTH="data/truth/object_truth_20260704_cleaned.parquet"
MANIFEST="data/gold/lsst_live_locked_test.json"
TRAIN_SPLIT="${TRAIN_SPLIT:-data/gold/split_fusion_v11_scc.json}"   # train/cal ids of the deployed stack
COHORT_CSV="data/live_eval_20260704/cohort/merged.csv"
COHORT_LABELS="data/live_eval_20260704/cohort/labels.csv"
LC_DIR="data/live_eval_20260704/lightcurves"
SILVER_DIR="data/live_eval_20260704/silver"
BENCH_DIR="reports/lsst_live_bench"
SCORE_SCRIPT="${SCORE_SCRIPT:-scripts/score_fusion_v11.py}"
BLEND_DIR="${BLEND_DIR:-models/anchor_blend_v11}"
FOLLOWUP_DIR="${FOLLOWUP_DIR:-models/followup_fusion_v11}"
TRUST_DIR="${TRUST_DIR:-models/trust_fusion_v11}"
CONFORMAL="${CONFORMAL:-models/conformal_fusion_v11/mondrian_aps.pkl}"

mkdir -p "$BENCH_DIR" "$(dirname "$MANIFEST")"

echo "=== [1/5] TNS bulk refresh (mode=$TNS_MODE) ==="
if [[ -n "$DIFF_DATE" ]]; then
  python scripts/download_tns_bulk.py --out "$TNS_PARQUET" --mode "$TNS_MODE" --diff-date "$DIFF_DATE"
else
  python scripts/download_tns_bulk.py --out "$TNS_PARQUET" --mode "$TNS_MODE"
fi

echo "=== [2/5] epoch-aware truth rebuild ==="
# Live seed (new spec arrivals + catalog negatives).
python scripts/build_truth_lsst_live.py --mode live \
  --tns-bulk "$TNS_PARQUET" --live-out "$LIVE_TRUTH" || {
    echo "WARN: live seed skipped (no Lasair creds / network) — continuing with cohort re-clean" >&2
  }
# Cohort re-clean (APPEND-ONLY: merges cleaned survivors with the existing
# frozen manifest — previously-appended even-hash TEST ids are never dropped
# and the original frozen_utc is preserved; B1/§6 "frozen ids never move").
python scripts/build_truth_lsst_live.py --mode cohort-clean \
  --tns-bulk "$TNS_PARQUET" --cohort "$COHORT_CSV" \
  --cleaned-out "$CLEANED_TRUTH" --manifest-out "$MANIFEST"

echo "=== [3/5] hash-route post-freeze spec arrivals ==="
# New spec ids (from the live seed) route even->TEST (append manifest), odd->train/cal.
# Frozen ids in the manifest are never moved (build_manifest is order-stable + dedup).
python - "$LIVE_TRUTH" "$MANIFEST" "$TRAIN_SPLIT" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from scripts.build_truth_lsst_live import hash_route

live_path, man_path, split_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if not live_path.exists():
    print("  no live truth — nothing to route"); raise SystemExit(0)
man = json.loads(man_path.read_text())
frozen = set(man["test_ids"])
# excluded ids (G6 counterpart integrity, e.g. locked-ZTF-twin objects) must
# never be re-frozen by a refresh
excluded = set(man.get("excluded_ids", {}))
# ids the deployed fusion model trained or calibrated on must never enter the
# benchmark (the 2026-07-07 refresh appended 10 such weak-labelled candidates)
split = json.loads(split_path.read_text())
trained = set(map(str, split["train_ids"])) | set(map(str, split["cal_ids"]))
live = pd.read_parquet(live_path)
spec = live[live["label_quality"] == "spectroscopic"]["object_id"].astype(str)
added = [o for o in spec
         if o not in frozen and o not in excluded and o not in trained and hash_route(o) == "test"]
print(f"  skipped {sum(o in trained for o in spec)} spec ids already in the deployed train/cal split")
for o in added:
    if o not in frozen:
        man["test_ids"].append(o); frozen.add(o)
man_path.write_text(json.dumps(man, indent=2))
print(f"  routed {len(added)} new spec ids -> TEST (manifest now {len(man['test_ids'])})")
PY

echo "=== [4/5] re-score locked benchmark (RUNBOOK scoring flow) ==="
EMPTY_SPLIT="$BENCH_DIR/empty_trust_metadata.json"
printf '{"train_ids": [], "cal_ids": [], "test_ids": []}\n' > "$EMPTY_SPLIT"
SNAP="$BENCH_DIR/snapshots_${STAMP}.parquet"
python scripts/build_snapshots_fusion.py \
  --lc-dir "$LC_DIR" \
  --silver-dir "$SILVER_DIR" \
  --truth "$CLEANED_TRUTH" \
  --bts "" \
  --labels "$COHORT_LABELS" \
  --trust-metadata "$EMPTY_SPLIT" \
  --no-lsst-weak \
  --lsst-live-locked "$MANIFEST" \
  --output "$SNAP" \
  --split-manifest "$BENCH_DIR/split_${STAMP}.json" \
  --skip-traj --n-jobs 8

if [[ -f "$SCORE_SCRIPT" ]]; then
  python "$SCORE_SCRIPT" \
    --snapshots "$SNAP" \
    --followup-dir "$FOLLOWUP_DIR" \
    --trust-dir "$TRUST_DIR" \
    --conformal "$CONFORMAL" \
    --blend-dir "$BLEND_DIR" \
    --scores-dir "$BENCH_DIR/scores" --tag "lsst_live_${STAMP}" --no-priority
else
  echo "WARN: $SCORE_SCRIPT not present (P4 pending) — snapshot built, scoring skipped" >&2
fi

echo "=== [5/5] append benchmark row ==="
python - "$BENCH_DIR" "$MANIFEST" "$STAMP" <<'PY'
import json, sys
from pathlib import Path
bench_dir, man_path, stamp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
man = json.loads(man_path.read_text())
ledger = bench_dir / "benchmark_ledger.jsonl"
row = {"stamp": stamp, "n_test_ids": len(man["test_ids"]),
       "frozen_utc": man.get("frozen_utc"), "source": man.get("source")}
with ledger.open("a") as fh:
    fh.write(json.dumps(row) + "\n")
print(f"  appended benchmark row -> {ledger}: {row}")
PY

echo "OK refresh_lsst_live complete ($STAMP)"
