#!/bin/bash -l
#$ -N debass_backfill
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o logs/backfill.$JOB_ID.log
#$ -t 1-3
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# Job array: one task per phase-1 broker (alerce, fink, lasair)

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.11.4}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

BROKERS=(alerce fink lasair)
BROKER=${BROKERS[$((SGE_TASK_ID - 1))]}

echo "Backfilling broker: $BROKER"
python scripts/backfill.py --broker "$BROKER" --limit "${DEBASS_LIMIT:-500}"
