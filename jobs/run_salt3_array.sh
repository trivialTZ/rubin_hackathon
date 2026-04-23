#!/bin/bash
# DEBASS Light-up — F4b: SALT3 χ² sharded SGE array job.
# Expects N_SHARDS task slots. Each slot processes objects[i] where (i % N_SHARDS == SGE_TASK_ID-1).
# Submit as:
#   qsub -t 1-30 -N debass_salt3 -cwd -V -l h_rt=03:00:00 -l mem_per_core=8G -pe omp 1 \
#        -o logs/salt3.\$TASK_ID.out -e logs/salt3.\$TASK_ID.err jobs/run_salt3_array.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true

N_SHARDS=${N_SHARDS:-30}
SHARD_ID=$((SGE_TASK_ID - 1))

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) SALT3 shard $SHARD_ID / $N_SHARDS start"
python3 -u scripts/run_salt3_shard.py \
    --from-labels data/labels.csv \
    --lc-dir data/lightcurves \
    --shard-id "$SHARD_ID" \
    --n-shards "$N_SHARDS" \
    --max-n-det 20 \
    --output-root data/silver/local_expert_outputs/salt3_chi2_shards
echo "$(ts) SALT3 shard $SHARD_ID done"
