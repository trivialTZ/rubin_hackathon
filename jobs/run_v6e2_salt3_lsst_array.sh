#!/bin/bash
#$ -N debass_v6e2_salt3
#$ -l h_rt=02:00:00
#$ -l mem_per_core=16G
#$ -pe omp 1
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v6e2_salt3.array.$TASK_ID.out
#
# v6e.2 — sharded salt3_chi2 inference on LSST training labels.
# Each shard writes to its own silver dir to avoid races on the JSON cache.
#
# Submit:
#   qsub -t 1-8 jobs/run_v6e2_salt3_lsst_array.sh
#
# Then run jobs/run_v6e2_post_array_merge.sh (interactive or qsub) to
# combine shards into data/silver and continue with lc_features_bv.

set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python

N_SHARDS=8
SHARD_ID=$((SGE_TASK_ID - 1))
SHARD_SILVER="data/silver_v6e2_salt3_shard_${SHARD_ID}"

echo "=== v6e2 salt3 shard ${SHARD_ID}/${N_SHARDS} @ $(date) on $(hostname) ==="
$PY -u scripts/local_infer.py \
    --expert salt3_chi2 \
    --from-labels data/labels_lsst.csv \
    --lc-dir data/lightcurves \
    --silver-dir "$SHARD_SILVER" \
    --max-n-det 20 \
    --shard-id "$SHARD_ID" \
    --n-shards "$N_SHARDS"
echo "=== shard ${SHARD_ID} done @ $(date) ==="
