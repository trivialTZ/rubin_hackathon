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

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.11.4}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

python scripts/normalize.py \
    --bronze-dir data/bronze \
    --silver-dir data/silver \
    --gold-dir   data/gold \
    ${DEBASS_LABELS:+--labels "$DEBASS_LABELS"}
