#!/bin/bash
# DEBASS Light-up — F3-SAFE-v2: no-leakage rebuild with SALT3 χ² as 6th clean expert.
# Outputs write to *_safe_v2 paths; preserves the 5-expert SAFE baseline.
# Submit as:  qsub -N debass_f3safev2 -cwd -V -l h_rt=04:00:00 -l mem_per_core=16G -pe omp 4 \
#                  -o logs/f3safev2.qsub.out -e logs/f3safev2.qsub.err jobs/run_f3_safe_v2_pipeline.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) ============================================================"
echo "$(ts) F3-SAFE-v2 pipeline — SAFE + SALT3 χ²"
echo "$(ts) ============================================================"

echo "$(ts) Step 1/4: build_object_epoch_snapshots (safe v2)"
python3 -u scripts/build_object_epoch_snapshots.py \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --gold-dir data/gold \
    --truth data/truth/object_truth.parquet \
    --objects-csv data/labels.csv \
    --output data/gold/object_epoch_snapshots_safe_v2.parquet \
    --max-n-det 20
echo "$(ts) gold-safe-v2 rebuilt"

echo "$(ts) Step 2/4: build_expert_helpfulness (safe v2)"
python3 -u scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v2.parquet \
    --output data/gold/expert_helpfulness_safe_v2.parquet
echo "$(ts) helpfulness-safe-v2 rebuilt"

echo "$(ts) Step 3/4: train_expert_trust (safe v2)"
python3 -u scripts/train_expert_trust.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v2.parquet \
    --helpfulness data/gold/expert_helpfulness_safe_v2.parquet \
    --models-dir models/trust_safe_v2 \
    --output-snapshots data/gold/object_epoch_snapshots_trust_safe_v2.parquet \
    --metrics-out reports/metrics/expert_trust_metrics_safe_v2.json \
    --n-jobs 4
echo "$(ts) trust heads trained (safe v2)"

echo "$(ts) Step 4/4: train_followup (safe v2)"
python3 -u scripts/train_followup.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v2.parquet \
    --trust-models-dir models/trust_safe_v2 \
    --model-dir models/followup_safe_v2 \
    --metrics-out reports/metrics/followup_metrics_safe_v2.json
echo "$(ts) followup trained (safe v2)"

echo "$(ts) Summary — live experts (safe v2):"
ls -1 models/trust_safe_v2/ | grep -v 'metadata\.json$' | sort
echo "$(ts) Trust heads count (safe v2): $(ls -1 models/trust_safe_v2/ | grep -v 'metadata.json$' | wc -l)"
echo "$(ts) Compare: SAFE vs SAFE-v2"
echo "$(ts)   SAFE    trust heads:  $(ls -1 models/trust_safe/ 2>/dev/null | grep -v 'metadata.json$' | wc -l)"
echo "$(ts)   SAFE-v2 trust heads:  $(ls -1 models/trust_safe_v2/ | grep -v 'metadata.json$' | wc -l)"
echo "$(ts) F3-SAFE-v2 DONE"
