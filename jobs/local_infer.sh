#!/bin/bash -l
#$ -N debass_local_infer
#$ -l h_rt=06:00:00
#$ -l mem_per_core=16G
#$ -pe omp 8
#$ -j y
#$ -o logs/local_infer.$JOB_ID.log
#$ -hold_jid debass_normalize
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

set -euo pipefail

module load python3/3.13.8
module load pytorch
source "$HOME/debass_env/bin/activate"

cd "$DEBASS_ROOT"

python scripts/local_infer.py \
    --expert all \
    --limit  "${DEBASS_LIMIT:-1000}" \
    --silver-dir data/silver
