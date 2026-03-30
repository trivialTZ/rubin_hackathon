#!/bin/bash -l
#$ -N debass_local_infer
#$ -l h_rt=06:00:00
#$ -l mem_per_core=16G
#$ -pe omp 8
#$ -j y
#$ -o logs/local_infer.$JOB_ID.log
#$ -hold_jid debass_normalize
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

python scripts/local_infer.py \
    --expert all \
    --limit  "${DEBASS_LIMIT:-1000}" \
    --silver-dir data/silver
