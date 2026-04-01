#!/bin/bash -l
#$ -N debass_meta_download
#$ -l h_rt=12:00:00
#$ -l mem_per_core=8G
#$ -pe omp 1
#$ -j y
#$ -o logs/download.$JOB_ID.log
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# Download weakly labelled ALeRCE seed objects + lightcurves.
# Network-bound job; no GPU needed.
# For 2000 objects at ~0.15s/LC expect ~10-15 min.
#
# Environment variables:
#   DEBASS_LIMIT   (default 2000) — total weakly labelled seed objects to download
#   DEBASS_ROOT    — path to rubin_hackathon repo

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
if [ ! -f "$DEBASS_VENV/bin/activate" ]; then
    echo "ERROR: virtualenv not found at $DEBASS_VENV"
    exit 1
fi
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"
mkdir -p logs data/lightcurves

echo "=== DEBASS: downloading weakly labelled ALeRCE seed data ==="
echo "Limit: ${DEBASS_LIMIT:-2000} objects"
echo "Node:  $(hostname)"
echo "Start: $(date)"

python scripts/download_alerce_training.py \
    --limit      "${DEBASS_LIMIT:-2000}" \
    --lc-dir     data/lightcurves \
    --labels-out data/labels.csv \
    --resume \
    --rate-limit 0.15

echo "Done: $(date)"
