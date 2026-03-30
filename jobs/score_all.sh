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

module load python3/3.13.8
source "$HOME/debass_env/bin/activate"

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
