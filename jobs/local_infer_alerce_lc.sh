#!/bin/bash -l
#$ -N debass_meta_local_infer_alerce_lc
#$ -l h_rt=02:00:00
#$ -l mem_per_core=8G
#$ -pe omp 1
#$ -j y
#$ -o logs/local_infer_alerce_lc.$JOB_ID.log

# Run ALeRCE LC local expert inference on CPU.
# Random forest is fast — no batch mode needed (~15 min for 7K objects × 20 epochs).
# Requires: artifacts/local_experts/alerce_lc/model.pkl (from train_alerce_lc.sh)
#
# Usage:
#   qsub -V -P pi-brout jobs/local_infer_alerce_lc.sh

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

export PYTHONUNBUFFERED=1

cd "$DEBASS_ROOT"

echo "=== DEBASS: ALeRCE LC local inference (CPU) ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"

python3 scripts/local_infer.py \
    --expert alerce_lc \
    --cpu-only \
    --from-labels data/labels.csv \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --max-n-det "${DEBASS_MAX_N_DET:-20}"

echo "Done: $(date)"
