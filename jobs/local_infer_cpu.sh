#!/bin/bash -l
#$ -N debass_meta_local_infer_cpu
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o logs/local_infer_cpu.$JOB_ID.log

# Run CPU-only local expert inference (ALeRCE LC classifier).
# Produces per-(object_id, n_det) classification scores that replace
# the ALeRCE API broker scores in the epoch table.

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

echo "=== DEBASS: CPU local expert inference ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"

export OMP_NUM_THREADS="${NSLOTS:-1}"

python scripts/local_infer.py \
    --expert     all \
    --cpu-only \
    --from-labels data/labels.csv \
    --lc-dir     data/lightcurves \
    --max-n-det  "${DEBASS_MAX_N_DET:-20}" \
    --require-live-model

echo "Done: $(date)"
