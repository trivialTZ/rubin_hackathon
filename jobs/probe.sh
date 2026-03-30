#!/bin/bash -l
#$ -N debass_probe
#$ -l h_rt=00:30:00
#$ -l mem_per_core=4G
#$ -pe omp 1
#$ -j y
#$ -o logs/probe.$JOB_ID.log
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.11.4}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

python scripts/probe.py --object ZTF21abbzjeq
