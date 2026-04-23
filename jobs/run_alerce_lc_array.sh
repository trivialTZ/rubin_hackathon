#!/bin/bash
# DEBASS — alerce_lc inference SGE array.
#   N_SHARDS=30 qsub -t 1-30 -N debass_alerce -cwd -V -l h_rt=02:00:00 -l mem_per_core=8G -pe omp 1 \
#        -o logs/alerce.\$TASK_ID.out -e logs/alerce.\$TASK_ID.err -v N_SHARDS=30 jobs/run_alerce_lc_array.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
N_SHARDS=${N_SHARDS:-30}
SHARD_ID=$((SGE_TASK_ID - 1))
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) alerce_lc shard $SHARD_ID / $N_SHARDS start"
python3 -u scripts/run_local_expert_shard.py \
    --expert alerce_lc \
    --from-labels data/labels.csv \
    --lc-dir data/lightcurves \
    --shard-id "$SHARD_ID" \
    --n-shards "$N_SHARDS" \
    --max-n-det 20
echo "$(ts) alerce_lc shard $SHARD_ID done"
