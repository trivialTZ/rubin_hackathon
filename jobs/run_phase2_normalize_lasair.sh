#!/bin/bash -l
#$ -N debass_norm_las
#$ -l h_rt=02:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -j y
#$ -cwd
#$ -o /project/pi-brout/rubin_hackathon/logs/phase2_normalize_lasair.log

# Normalize lasair bronze → silver, including the big new power-user run (1.08 GB).
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
set -a && source .env && set +a
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) normalize broker=lasair"
python3 -u scripts/normalize_antares_only.py --broker lasair

echo "$(ts) silver counts after all 3 broker normalizes"
python3 -u -c "
import pandas as pd
d = pd.read_parquet('data/silver/broker_events.parquet', columns=['broker','expert_key'])
print('total:', len(d))
print(d['broker'].value_counts().to_string())
print()
print('-- all expert_keys (trust-candidates in bold would be ≥5 events) --')
print(d['expert_key'].value_counts().to_string())
"
echo "$(ts) lasair normalize DONE"
