#!/bin/bash
#$ -N debass_v6e2_arr
#$ -l h_rt=06:00:00
#$ -l mem_per_core=16G
#$ -pe omp 1
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v6e2_gold.array.$TASK_ID.out
#
# v6e.2 gold rebuild — sharded SGE array.
# Mirrors jobs/run_v6e_dp1_array.sh but for the training-set gold builder.
#
# Submit with:
#   qsub -t 1-8 jobs/run_v6e2_gold_array.sh
#
# After all 8 tasks complete:
#   python scripts/merge_dp1_snapshot_shards.py \
#     --shard-glob 'data/gold/object_epoch_snapshots_v6e2.shard-*-of-8.parquet' \
#     --out data/gold/object_epoch_snapshots_v6e2_raw.parquet
#   bash jobs/run_v6e2_pipeline.sh
#
# h_rt=6h is conservative; v6e DP1 array did 41K LSST LCs in 17 min wall.
# Training set is 12,772 objects (~3K LSST + 9K ZTF) — expected ≪ 1 hr/shard.

set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python

N_SHARDS=8
SHARD_ID=$((SGE_TASK_ID - 1))

OUT="data/gold/object_epoch_snapshots_v6e2.shard-${SHARD_ID}-of-${N_SHARDS}.parquet"

echo "=== v6e2 gold shard ${SHARD_ID}/${N_SHARDS} @ $(date) on $(hostname) ==="
$PY scripts/build_object_epoch_snapshots.py \
    --lc-dir      data/lightcurves \
    --silver-dir  data/silver \
    --gold-dir    data/gold \
    --truth       data/truth/object_truth.parquet \
    --objects-csv data/labels.csv \
    --max-n-det   20 \
    --output      "$OUT" \
    --shard-id    "$SHARD_ID" \
    --n-shards    "$N_SHARDS"
echo "=== shard ${SHARD_ID} done @ $(date) ==="
