#!/bin/bash -l
#$ -N debass_meta_meta_build_epochs
#$ -l h_rt=02:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o logs/build_epochs.$JOB_ID.log
#$ -hold_jid debass_meta_meta_backfill
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# Build trust-aware CPU artifacts from cached lightcurves + broker events.
# Assumes backfill.sh has already populated data/bronze via the job array.

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_meta_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
if [ ! -f "$DEBASS_VENV/bin/activate" ]; then
    echo "ERROR: virtualenv not found at $DEBASS_VENV"
    exit 1
fi
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

echo "=== DEBASS: building trust-aware CPU artifacts ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"

export OMP_NUM_THREADS="${NSLOTS:-1}"
export OPENBLAS_NUM_THREADS="${NSLOTS:-1}"
export MKL_NUM_THREADS="${NSLOTS:-1}"
export NUMEXPR_NUM_THREADS="${NSLOTS:-1}"

# Normalise bronze → silver (event-level only; no legacy gold)
python scripts/normalize.py \
    --bronze-dir data/bronze \
    --silver-dir data/silver

# Build truth table from weak labels
python scripts/build_truth_table.py \
    --labels data/labels.csv \
    --output data/truth/object_truth.parquet

python scripts/build_object_epoch_snapshots.py \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --gold-dir data/gold \
    --truth data/truth/object_truth.parquet \
    --objects-csv data/labels.csv \
    --max-n-det "${DEBASS_MAX_N_DET:-20}"

python scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots.parquet \
    --output data/gold/expert_helpfulness.parquet

# Keep the baseline epoch table available as an explicit benchmark artifact.
python scripts/build_epoch_table_from_lc.py \
    --lc-dir     data/lightcurves \
    --silver-dir data/silver \
    --epochs-dir data/epochs \
    --labels     data/labels.csv \
    --max-n-det  "${DEBASS_MAX_N_DET:-20}"

echo "Done: $(date)"
