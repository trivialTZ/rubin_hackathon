#!/bin/bash -l
#$ -N debass_score_all
#$ -l h_rt=01:00:00
#$ -l mem_per_core=8G
#$ -pe omp 1
#$ -j y
#$ -o logs/score_all.$JOB_ID.log
#$ -hold_jid debass_train_early
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# Score all objects in the epoch table and write follow-up priority reports.

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

echo "=== DEBASS: scoring all objects ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"

mkdir -p reports/scores

# Score at n_det = 3, 5, 10 (early, medium, late)
for N in 3 5 10; do
    echo "  Scoring at n_det=${N}..."
    python scripts/score_early.py \
        --from-labels data/labels.csv \
        --n-det       $N \
        --epochs-dir  data/epochs \
        --model-dir   models/early_meta \
        > reports/scores/scores_ndet${N}.txt 2>&1
done

# Full trajectory for all objects (all n_det)
python scripts/score_early.py \
    --from-labels data/labels.csv \
    --epochs-dir  data/epochs \
    --model-dir   models/early_meta \
    --max-n-det   20

echo "Done: $(date)"
