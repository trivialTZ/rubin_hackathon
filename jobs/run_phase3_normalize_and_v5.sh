#!/bin/bash -l
#$ -N debass_phase3_v5
#$ -l h_rt=06:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -j y
#$ -o logs/phase3_v5.log

# Runs sequentially AFTER Phase 1 backfills complete:
#   1. Incremental normalize for fink/lasair/pitt_google bronze → silver
#   2. F3-SAFE-v5 pipeline: build gold → helpfulness → trust → followup
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
set -a && source .env && set +a
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

for broker in fink lasair pitt_google; do
  echo "$(ts) normalize_${broker}"
  python3 -u scripts/normalize_antares_only.py --broker "$broker"
done

echo "$(ts) F3-SAFE-v5 pipeline"
bash jobs/run_f3_safe_v5_pipeline.sh

echo "$(ts) Phase 3 DONE"
