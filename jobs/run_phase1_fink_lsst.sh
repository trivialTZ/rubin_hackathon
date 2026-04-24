#!/bin/bash -l
#$ -N debass_fink_ph1
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o logs/phase1_fink.log

set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
set -a && source .env && set +a
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) Phase 1 — Fink backfill (ZTF + LSST via dual-survey)"
python scripts/backfill.py --broker fink --from-labels data/labels.csv --parallel 8
echo "$(ts) fink DONE"
ls -la data/bronze/ | grep fink_ | tail -5
