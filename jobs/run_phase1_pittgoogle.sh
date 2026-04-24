#!/bin/bash -l
#$ -N debass_pitt_ph1
#$ -l h_rt=01:00:00
#$ -l mem_per_core=8G
#$ -pe omp 2
#$ -j y
#$ -o logs/phase1_pittgoogle.log

set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
set -a && source .env && set +a
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) Phase 1 — Pitt-Google BQ batch (cost-safe)"
python scripts/backfill_pittgoogle_batch.py --from-labels data/labels.csv --save-fixtures --max-gb 15
echo "$(ts) pitt_google DONE"
ls -la data/bronze/ | grep pitt_google_ | tail -5
