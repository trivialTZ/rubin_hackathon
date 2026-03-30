#!/bin/bash -l
#$ -N debass_train
#$ -l h_rt=08:00:00
#$ -l mem_per_core=28G
#$ -pe omp 8
#$ -j y
#$ -o logs/train.$JOB_ID.log
#$ -hold_jid debass_local_infer
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

set -euo pipefail

module load python3/3.13.8
module load pytorch
source "$HOME/debass_env/bin/activate"

cd "$DEBASS_ROOT"

python scripts/train.py \
    --method  "${DEBASS_METHOD:-lgbm}" \
    --val-frac 0.2 \
    --gold-dir  data/gold \
    --model-dir models
