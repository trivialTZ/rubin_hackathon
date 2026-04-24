#!/bin/bash -l
#$ -N debass_norm_fp
#$ -l h_rt=03:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -j y
#$ -cwd
#$ -o /project/pi-brout/rubin_hackathon/logs/phase2_normalize_fp.log

# Incremental normalize for the two brokers that already finished Phase 1:
#   fink (1776909829.parquet, 950 MB) + pitt_google (1776910601.parquet, 313 MB)
# Appends to data/silver/broker_events.parquet without touching alerce/antares/ampel.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
set -a && source .env && set +a
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

for broker in fink pitt_google; do
  echo "$(ts) normalize broker=$broker"
  python3 -u scripts/normalize_antares_only.py --broker "$broker"
done

echo "$(ts) silver event counts (by broker)"
python3 -u -c "
import pandas as pd
d = pd.read_parquet('data/silver/broker_events.parquet', columns=['broker','expert_key'])
print('total:', len(d))
print(d['broker'].value_counts().to_string())
print('\n-- top 30 expert_keys --')
print(d['expert_key'].value_counts().head(30).to_string())
"

echo "$(ts) Phase 2 (fink+pitt) normalize DONE"
