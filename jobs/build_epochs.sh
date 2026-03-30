#!/bin/bash -l
#$ -N debass_build_epochs
#$ -l h_rt=02:00:00
#$ -l mem_per_core=8G
#$ -pe omp 4
#$ -j y
#$ -o logs/build_epochs.$JOB_ID.log
#$ -hold_jid debass_download
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# Build (object_id, n_det) epoch feature table from cached lightcurves.
# Also fetches ALeRCE + Fink broker scores for all objects.

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
if [ ! -f "$DEBASS_VENV/bin/activate" ]; then
    echo "ERROR: virtualenv not found at $DEBASS_VENV"
    exit 1
fi
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

echo "=== DEBASS: building epoch feature table ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"

# Fetch broker scores for all labelled objects (ALeRCE + Fink)
# Use --from-labels to avoid ARG_MAX shell limits with large object lists
python scripts/backfill.py \
    --broker      all \
    --from-labels data/labels.csv \
    --bronze-dir  data/bronze

# Normalise bronze → silver
python scripts/normalize.py \
    --bronze-dir data/bronze \
    --silver-dir data/silver \
    --skip-gold

# Build no-leakage epoch table from lightcurves
python scripts/build_epoch_table_from_lc.py \
    --lc-dir     data/lightcurves \
    --silver-dir data/silver \
    --epochs-dir data/epochs \
    --labels     data/labels.csv \
    --max-n-det  "${DEBASS_MAX_N_DET:-20}"

echo "Done: $(date)"
