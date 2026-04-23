#!/bin/bash
# DEBASS Light-up — F3-SAFE-v3: SAFE + SALT3 + alerce_lc (7 clean experts).
#   qsub -N debass_f3safev3 -cwd -V -l h_rt=04:00:00 -l mem_per_core=16G -pe omp 4 \
#        -o logs/f3safev3.qsub.out -e logs/f3safev3.qsub.err jobs/run_f3_safe_v4_pipeline.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) F3-SAFE-v3 — SAFE + SALT3 + alerce_lc (7 clean experts)"

python3 -u scripts/build_object_epoch_snapshots.py \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --gold-dir data/gold \
    --truth data/truth/object_truth.parquet \
    --objects-csv data/labels.csv \
    --output data/gold/object_epoch_snapshots_safe_v4.parquet \
    --max-n-det 20

python3 -u scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v4.parquet \
    --output data/gold/expert_helpfulness_safe_v4.parquet

python3 -u scripts/train_expert_trust.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v4.parquet \
    --helpfulness data/gold/expert_helpfulness_safe_v4.parquet \
    --models-dir models/trust_safe_v4 \
    --output-snapshots data/gold/object_epoch_snapshots_trust_safe_v4.parquet \
    --metrics-out reports/metrics/expert_trust_metrics_safe_v4.json \
    --n-jobs 4

python3 -u scripts/train_followup.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v4.parquet \
    --trust-models-dir models/trust_safe_v4 \
    --model-dir models/followup_safe_v4 \
    --metrics-out reports/metrics/followup_metrics_safe_v4.json

echo "$(ts) F3-SAFE-v3 DONE"
echo "  trust heads (safe-v3): $(ls -1 models/trust_safe_v4/ | grep -v 'metadata.json$' | wc -l)"
ls -1 models/trust_safe_v4/ | grep -v 'metadata.json$' | sort
python3 -c 'import json; d=json.load(open("reports/metrics/followup_metrics_safe_v4.json")); print("followup cal AUC:", d["test_calibrated"]["roc_auc"])'
