#!/bin/bash -l
#$ -N debass_meta_analyze
#$ -l h_rt=00:30:00
#$ -l mem_per_core=8G
#$ -pe omp 1
#$ -j y
#$ -o logs/analyze_results.$JOB_ID.log

# Run results analysis after the full pipeline completes.
#
# Usage:
#   qsub -V -P pi-brout -hold_jid <score_job_id> jobs/analyze_results.sh

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

echo "=== DEBASS: Results Analysis ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"

# Find the best scores file (prefer n_det=3 for early classification)
SCORES_FILE=""
for N in 3 5 10; do
    if [ -f "reports/scores/scores_ndet${N}.jsonl" ]; then
        SCORES_FILE="reports/scores/scores_ndet${N}.jsonl"
        break
    fi
done

python3 scripts/analyze_results.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --snapshots-raw data/gold/object_epoch_snapshots.parquet \
    --helpfulness data/gold/expert_helpfulness.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics.json \
    --followup-metrics reports/metrics/followup_metrics.json \
    ${SCORES_FILE:+--scores "$SCORES_FILE"} \
    --output-dir reports

echo ""
echo "=== RESULTS SUMMARY ==="
cat reports/results_summary.txt

echo ""
echo "Done: $(date)"
