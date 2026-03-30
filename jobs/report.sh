#!/bin/bash -l
#$ -N debass_report
#$ -l h_rt=00:30:00
#$ -l mem_per_core=8G
#$ -pe omp 1
#$ -j y
#$ -o logs/report.$JOB_ID.log
#$ -hold_jid debass_train
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

set -euo pipefail

module load python3/3.13.8
source "$HOME/debass_env/bin/activate"

cd "$DEBASS_ROOT"

python scripts/report.py \
    --model-dir models \
    --gold-dir  data/gold

echo "Report written to reports/metrics/evaluation.json"
