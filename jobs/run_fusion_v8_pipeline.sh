#!/bin/bash
# metaDEBASS fusion_v8 — full SCC chain: build → helpfulness → train → score (+DP1) → eval.
#   qsub -N debass_fusion_v8 -cwd -V -P pi-brout -l h_rt=08:00:00 -l mem_per_core=8G -pe omp 16 \
#        -o logs/fusion_v8.qsub.out -e logs/fusion_v8.qsub.err jobs/run_fusion_v8_pipeline.sh
#
# Smoke variant (sanity, ~minutes):
#   FUSION_V8_SMOKE=1 bash jobs/run_fusion_v8_pipeline.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

NSLOTS=${NSLOTS:-16}
SMOKE_FLAG=""
if [[ "${FUSION_V8_SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
fi
mkdir -p logs data/scores reports/fusion_v8 reports/metrics

echo "$(ts) fusion_v8 — START (NSLOTS=${NSLOTS}${SMOKE_FLAG:+, smoke})"

# Smoke runs write/read .smoke-suffixed artifacts so they can never clobber
# the real fusion_v8 outputs; explicit paths keep the chain consistent.
SNAP="data/gold/object_epoch_snapshots_fusion_v8.parquet"
SPLIT="data/gold/split_fusion_v8.json"
HELP="data/gold/expert_helpfulness_fusion_v8.parquet"
if [[ -n "${SMOKE_FLAG}" ]]; then
    SNAP="data/gold/object_epoch_snapshots_fusion_v8.smoke.parquet"
    SPLIT="data/gold/split_fusion_v8.smoke.json"
    HELP="data/gold/expert_helpfulness_fusion_v8.smoke.parquet"
fi
SNAP_TRUST="${SNAP%.parquet}_trust.parquet"

# 1. gold snapshots (all LCs + BTS/object_truth labels + EXT/traj features)
python3 -u scripts/build_snapshots_fusion.py ${SMOKE_FLAG}

# 1b. DP1 eval-only snapshot → data/gold/dp1_snapshots_fusion_v8.parquet
#     (--dp1-only: step 1 already built the gold table)
python3 -u scripts/build_snapshots_fusion.py --dp1-only ${SMOKE_FLAG}

# 2. long-format Stage-A helpfulness table with EXT/traj context
python3 -u scripts/build_helpfulness_fusion.py --snapshots "${SNAP}" --output "${HELP}"

# 3. Stage A pooled trust → Stage B multiclass → Dirichlet → conformal (+ gates)
python3 -u scripts/train_fusion_v8.py --n-jobs "${NSLOTS}" ${SMOKE_FLAG} \
    --snapshots "${SNAP}" --helpfulness "${HELP}" --split "${SPLIT}" \
    --output-snapshots "${SNAP_TRUST}"

# 4. score gold test snapshot + DP1 (DP1 carries truth-catalog cols, spec #2)
python3 -u scripts/score_fusion_v8.py ${SMOKE_FLAG} --snapshots "${SNAP_TRUST}"
python3 -u scripts/score_fusion_v8.py --dp1 ${SMOKE_FLAG}

# 5. honest evaluation — Tables 1-7 + pre-registered headline & guards
python3 -u scripts/eval_fusion_v8.py ${SMOKE_FLAG} --snapshots "${SNAP_TRUST}" --split "${SPLIT}"

echo "$(ts) fusion_v8 DONE"
echo "  artifacts:"
ls -1 models/trust_fusion_v8 models/followup_fusion_v8 models/conformal_fusion_v8 2>/dev/null || true
echo "  reports:"
ls -1 reports/fusion_v8 2>/dev/null || true
python3 - <<'EOF'
import json, pathlib
p = pathlib.Path("reports/fusion_v8/headline_guards.json")
if p.exists():
    hg = json.loads(p.read_text())
    print("HEADLINE:", json.dumps(hg.get("headline", {}), indent=2)[:1200])
    for g in hg.get("guards", []):
        print("GUARD:", g.get("guard"), "→", g.get("status") or ("PASS" if g.get("pass") else "FAIL"))
else:
    print("headline_guards.json not found")
EOF
