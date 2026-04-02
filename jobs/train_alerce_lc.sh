#!/bin/bash -l
#$ -N debass_meta_train_alerce_lc
#$ -l h_rt=02:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o logs/train_alerce_lc.$JOB_ID.log

# Train the local ALeRCE-style LC classifier (balanced random forest).
# CPU only — no GPU needed.

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${NSLOTS:-4}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-4}"

cd "$DEBASS_ROOT"

echo "=== DEBASS: training local ALeRCE LC classifier ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "Slots: ${NSLOTS:-4}"

python3 scripts/train_alerce_lc.py \
    --from-labels data/labels.csv \
    --lc-dir     data/lightcurves \
    --n-estimators 300 \
    --max-n-det  "${DEBASS_MAX_N_DET:-20}" \
    --sample-n-dets 3 5 10 15 20 \
    --n-jobs "${NSLOTS:-4}"

echo ""
echo "Next: run ALeRCE LC inference"
echo "  qsub -V -P pi-brout jobs/local_infer_alerce_lc.sh"
echo ""
echo "Done: $(date)"
