#!/bin/bash -l
#$ -N debass_meta_local_infer_snn
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 2
#$ -j y
#$ -o logs/local_infer_snn.$JOB_ID.log

# SuperNNova local inference on CPU.
# Uses Fink-trained SNN model via classify_lcs (no GPU needed).
#
# Usage:
#   qsub -V -P pi-brout jobs/local_infer_snn.sh

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

echo "=== DEBASS: SuperNNova local inference (CPU) ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "Slots: ${NSLOTS:-1}"

python3 scripts/local_infer.py \
    --expert supernnova \
    --cpu-only \
    --from-labels data/labels.csv \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --max-n-det "${DEBASS_MAX_N_DET:-20}"

echo "Done: $(date)"
