#!/bin/bash
# Train the local ALeRCE-style LightGBM classifier.
# qsub -N debass_alerce_lc -cwd -V -l h_rt=02:00:00 -l mem_per_core=8G -pe omp 4 \
#      -o logs/alerce_lc_train.out -e logs/alerce_lc_train.err jobs/run_alerce_lc_train.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) alerce_lc train start"
python3 -u scripts/train_alerce_lc.py \
    --from-labels data/labels.csv \
    --n-estimators 300 \
    --max-n-det 20 \
    --n-jobs 4
echo "$(ts) alerce_lc train done"
ls -la artifacts/local_experts/alerce_lc/ 2>/dev/null || true
