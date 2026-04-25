#!/bin/bash
#$ -N debass_v6e_arr
#$ -l h_rt=04:00:00
#$ -l mem_per_core=16G
#$ -pe omp 1
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v6e_dp1.array.$TASK_ID.out
#
# v6e DP1 build — sharded SGE array job.
# Submit with:
#   qsub -t 1-8 jobs/run_v6e_dp1_array.sh
# Each task processes (N_total / 8) LCs determined by stride.
# After all 8 tasks complete, run:
#   python scripts/merge_dp1_snapshot_shards.py \
#     --shard-glob 'data/gold/dp1_snapshots_v6e.shard-*-of-8.parquet' \
#     --out data/gold/dp1_snapshots_50k_v6e.parquet
#   python scripts/score_dp1_v5d.py --snapshots data/gold/dp1_snapshots_50k_v6e.parquet \
#     --trust-dir models/trust_safe_v5d --followup-dir models/followup_safe_v5d \
#     --out-dir reports/v6e_dp1_50k

set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python

N_SHARDS=8
SHARD_ID=$((SGE_TASK_ID - 1))

OUT="data/gold/dp1_snapshots_v6e.shard-${SHARD_ID}-of-${N_SHARDS}.parquet"

echo "=== v6e shard ${SHARD_ID}/${N_SHARDS} @ $(date) on $(hostname) ==="
$PY scripts/build_dp1_snapshots.py \
    --truth data/truth/dp1_truth_50k.parquet \
    --out   "$OUT" \
    --shard-id "$SHARD_ID" \
    --n-shards "$N_SHARDS"
echo "=== shard ${SHARD_ID} done @ $(date) ==="
