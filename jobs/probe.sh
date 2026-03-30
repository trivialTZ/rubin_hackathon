#!/bin/bash -l
#$ -N debass_probe
#$ -l h_rt=00:30:00
#$ -l mem_per_core=4G
#$ -pe omp 1
#$ -j y
#$ -o logs/probe.$JOB_ID.log
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

set -euo pipefail

module load python3/3.13.8
source "$HOME/debass_env/bin/activate"

cd "$DEBASS_ROOT"

python scripts/probe.py --object ZTF21abbzjeq
