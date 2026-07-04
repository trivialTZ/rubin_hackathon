#!/bin/bash
#$ -N debass_v9_build
#$ -cwd -V
#$ -l h_rt=06:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v9_build.qsub.out
#$ -e logs/fusion_v9_build.qsub.err
# metaDEBASS fusion_v9 — stage 1 on SCC: gold snapshot build (full 12,773-label
# set incl. LSST, BTS-unclassified demotion auto-applied), DP1 snapshot, and
# the long-format helpfulness table.  Produces the split manifest the GPU
# pretrain job needs.  Submit via jobs/submit_fusion_v9_chain.sh.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
mkdir -p logs data/gold data/scores reports/metrics

echo "$(ts) v9 build — START (NSLOTS=${NSLOTS})"
python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" --dp1
python3 -u scripts/build_helpfulness_fusion.py \
    --snapshots data/gold/object_epoch_snapshots_fusion_v8.parquet \
    --output data/gold/expert_helpfulness_fusion_v8.parquet --parity-check
echo "$(ts) v9 build — DONE"
