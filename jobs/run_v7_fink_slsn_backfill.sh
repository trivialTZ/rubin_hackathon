#!/bin/bash -l
#$ -N v7_fink_slsn
#$ -l h_rt=04:00:00
#$ -l mem_per_core=12G
#$ -pe omp 4
#$ -j y
#$ -o logs/v7_fink_slsn.$JOB_ID.log

set -euo pipefail

cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true

python -u scripts/backfill.py --broker fink --from-labels data/labels.csv --parallel 8
