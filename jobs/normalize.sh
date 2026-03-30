#!/bin/bash -l
#$ -N debass_normalize
#$ -l h_rt=01:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -j y
#$ -o logs/normalize.$JOB_ID.log
#$ -hold_jid debass_backfill
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

set -euo pipefail

module load python3/3.13.8
source "$HOME/debass_env/bin/activate"

cd "$DEBASS_ROOT"

python scripts/normalize.py \
    --bronze-dir data/bronze \
    --silver-dir data/silver \
    --gold-dir   data/gold \
    ${DEBASS_LABELS:+--labels "$DEBASS_LABELS"}
