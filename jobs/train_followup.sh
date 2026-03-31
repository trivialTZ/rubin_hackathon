#!/bin/bash -l
#$ -N debass_train_followup
#$ -l h_rt=02:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -j y
#$ -o logs/train_followup.$JOB_ID.log

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

export OMP_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"

cd "$DEBASS_ROOT"

python scripts/train_followup.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --trust-models-dir models/trust \
    --model-dir models/followup \
    --n-estimators "${DEBASS_N_EST:-200}" \
    --n-jobs "${NSLOTS:-1}"
