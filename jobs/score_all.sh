#!/bin/bash -l
#$ -N debass_meta_score_all
#$ -l h_rt=01:00:00
#$ -l mem_per_core=8G
#$ -pe omp 1
#$ -j y
#$ -o logs/score_all.$JOB_ID.log
#$ -hold_jid debass_meta_train_followup
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# Score all objects and write trust-aware expert_confidence payloads.
# Baseline scoring is available only when explicitly requested.

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
DEBASS_SCORE_MODE=${DEBASS_SCORE_MODE:-trust}
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
echo "Mode:  $DEBASS_SCORE_MODE"

mkdir -p reports/scores

if [ "$DEBASS_SCORE_MODE" = "baseline" ]; then
    for N in 3 5 10; do
        echo "  Baseline scoring at n_det=${N}..."
        python scripts/score_early.py \
            --from-labels data/labels.csv \
            --n-det       "$N" \
            --epochs-dir  data/epochs \
            --model-dir   models/early_meta \
            > reports/scores/scores_ndet${N}.txt 2>&1
    done
else
    if [ ! -f data/gold/object_epoch_snapshots_trust.parquet ]; then
        echo "ERROR: trust-aware snapshots not found at data/gold/object_epoch_snapshots_trust.parquet"
        echo "Run trust training first or set DEBASS_SCORE_MODE=baseline for benchmark scoring."
        exit 1
    fi
    if [ ! -f models/trust/metadata.json ]; then
        echo "ERROR: trust model metadata not found at models/trust/metadata.json"
        echo "Run jobs/train_expert_trust.sh before scoring."
        exit 1
    fi
    if [ ! -f models/followup/metadata.json ]; then
        echo "ERROR: follow-up model metadata not found at models/followup/metadata.json"
        echo "Run jobs/train_followup.sh before scoring."
        exit 1
    fi

    for N in 3 5 10; do
        echo "  Trust-aware scoring at n_det=${N}..."
        python scripts/score_nightly.py \
            --from-labels data/labels.csv \
            --n-det "$N" \
            --snapshots data/gold/object_epoch_snapshots_trust.parquet \
            --trust-models-dir models/trust \
            --followup-model-dir models/followup \
            --output "reports/scores/scores_ndet${N}.jsonl"
    done
fi

echo "Done: $(date)"
