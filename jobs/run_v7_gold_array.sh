#!/bin/bash
#$ -N debass_v7_gold
#$ -l h_rt=06:00:00
#$ -l mem_per_core=16G
#$ -pe omp 1
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v7_gold.array.$TASK_ID.out
#
# v7 gold rebuild — sharded SGE array.
# Picks up new fink/slsn (broker_events.parquet) + ampel/snguess
# (silver/local_expert_outputs/ampel/snguess/part-latest.parquet).
#
# Submit:
#   qsub -t 1-8 jobs/run_v7_gold_array.sh
# After all 8 done:
#   python scripts/merge_dp1_snapshot_shards.py \
#     --shard-glob 'data/gold/object_epoch_snapshots_v7.shard-*-of-8.parquet' \
#     --out data/gold/object_epoch_snapshots_v7_raw.parquet

set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python

N_SHARDS=8
SHARD_ID=$((SGE_TASK_ID - 1))

OUT="data/gold/object_epoch_snapshots_v7.shard-${SHARD_ID}-of-${N_SHARDS}.parquet"

echo "=== v7 gold shard ${SHARD_ID}/${N_SHARDS} @ $(date) on $(hostname) ==="
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
