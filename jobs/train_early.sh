#!/bin/bash -l
#$ -N debass_train_early
#$ -l h_rt=04:00:00
#$ -l mem_per_core=16G
#$ -pe omp 8
#$ -j y
#$ -o logs/train_early.$JOB_ID.log
#$ -hold_jid debass_build_epochs
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# Train the early-epoch meta-classifier (LightGBM) on epoch feature table.
# hold_jid is set to debass_build_epochs; if running after GPU job,
# submit_all.sh overrides this to hold on debass_local_infer_gpu instead.

set -euo pipefail

module load python3/3.13.8
source "$HOME/debass_env/bin/activate"

cd "$DEBASS_ROOT"

echo "=== DEBASS: training early-epoch meta-classifier ==="
echo "Node:  $(hostname)"
echo "NSLOTS: $NSLOTS"
echo "Start: $(date)"

python scripts/train_early.py \
    --epochs-dir   data/epochs \
    --model-dir    models/early_meta \
    --max-n-det    "${DEBASS_MAX_N_DET:-20}" \
    --val-frac     0.2 \
    --n-estimators "${DEBASS_N_EST:-500}"

echo "Done: $(date)"
