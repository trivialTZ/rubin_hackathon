#!/bin/bash
#$ -N debass_f3safev5b
#$ -l h_rt=01:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/f3safev5b.qsub.out

# F3-SAFE-v5b — LSST truth filled from ALeRCE stamp + Fink xm + consensus.
# Rebuilds helpfulness + retrains trust + retrains followup on patched v5b snapshot.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) F3-SAFE-v5b — helpfulness + trust + followup on LSST-truth-filled snapshot"

python3 -u scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v5b.parquet \
    --output    data/gold/expert_helpfulness_safe_v5b.parquet

python3 -u scripts/train_expert_trust.py \
    --snapshots        data/gold/object_epoch_snapshots_safe_v5b.parquet \
    --helpfulness      data/gold/expert_helpfulness_safe_v5b.parquet \
    --models-dir       models/trust_safe_v5b \
    --output-snapshots data/gold/object_epoch_snapshots_trust_safe_v5b.parquet \
    --metrics-out      reports/metrics/expert_trust_metrics_safe_v5b.json \
    --n-jobs 4

python3 -u scripts/train_followup.py \
    --snapshots        data/gold/object_epoch_snapshots_trust_safe_v5b.parquet \
    --trust-models-dir models/trust_safe_v5b \
    --model-dir        models/followup_safe_v5b \
    --metrics-out      reports/metrics/followup_metrics_safe_v5b.json

echo "$(ts) F3-SAFE-v5b DONE"
echo "  trust heads (safe-v5b): $(ls -1 models/trust_safe_v5b/ | grep -v 'metadata.json$' | wc -l)"
ls -1 models/trust_safe_v5b/ | grep -v 'metadata.json$' | sort
python3 -c 'import json; d=json.load(open("reports/metrics/followup_metrics_safe_v5b.json")); print("followup cal AUC:", d["test_calibrated"]["roc_auc"])'
