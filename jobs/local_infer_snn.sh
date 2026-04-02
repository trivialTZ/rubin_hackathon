#!/bin/bash -l
#$ -N debass_meta_local_infer_snn
#$ -l h_rt=02:00:00
#$ -l mem_per_core=16G
#$ -pe omp 2
#$ -j y
#$ -o logs/local_infer_snn.$JOB_ID.log

# SuperNNova local inference on CPU (batch mode).
# Uses Fink-trained SNN model via classify_lcs.
# Batch mode: model loaded once per chunk (~500 objects), not per epoch.
# Expected runtime: ~5–15 min for 7K objects (was ~6h without batching).
#
# Usage:
#   qsub -V -P pi-brout jobs/local_infer_snn.sh

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

# Flush Python output immediately so logs are readable in real time
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"

cd "$DEBASS_ROOT"

echo "=== DEBASS: SuperNNova local inference (CPU, batch mode) ==="
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
