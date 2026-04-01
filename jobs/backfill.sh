#!/bin/bash -l
#$ -N debass_meta_backfill
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o logs/backfill.$JOB_ID.log
#$ -t 1-3
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# Job array: one task per phase-1 broker (alerce, fink, lasair)
# Uses data/labels.csv as the source object list so SCC backfill matches the
# weak-label seed set prepared by download_training.sh.

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

BROKERS=(alerce fink lasair)
BROKER=${BROKERS[$((SGE_TASK_ID - 1))]}

echo "Backfilling broker: $BROKER"
python scripts/backfill.py --broker "$BROKER" --from-labels data/labels.csv
