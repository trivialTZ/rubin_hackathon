#!/bin/bash -l
#$ -N debass_lasair_ph1
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o logs/phase1_lasair.log

set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
set -a && source .env && set +a
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) Phase 1 — Lasair backfill (ZTF + LSST dual-mode, Power Users tier)"
python -u scripts/backfill.py --broker lasair --from-labels data/labels.csv --parallel 16
echo "$(ts) lasair DONE"
ls -la data/bronze/ | grep lasair_ | tail -5
