#!/bin/bash -l
#$ -N debass_meta_train_expert_trust
#$ -l h_rt=04:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -j y
#$ -o logs/train_expert_trust.$JOB_ID.log

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

export OMP_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"

cd "$DEBASS_ROOT"

python scripts/train_expert_trust.py \
    --snapshots data/gold/object_epoch_snapshots.parquet \
    --helpfulness data/gold/expert_helpfulness.parquet \
    --models-dir models/trust \
    --output-snapshots data/gold/object_epoch_snapshots_trust.parquet
