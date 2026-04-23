#!/bin/bash
# DEBASS Light-up — F3 pipeline: gold snapshots → helpfulness → trust heads → followup → analyze
# Run from /project/pi-brout/rubin_hackathon after normalize (F2) finishes.
# Submit as:  qsub -N debass_f3 -cwd -V -l h_rt=04:00:00 -l mem_per_core=16G -pe omp 4 \
#                  -o logs/f3.qsub.out -e logs/f3.qsub.err jobs/run_f3_pipeline.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) ============================================================"
echo "$(ts) F3 pipeline — build gold + train all trust heads"
echo "$(ts) ============================================================"

echo "$(ts) Step 1/4: build_object_epoch_snapshots"
python3 -u scripts/build_object_epoch_snapshots.py \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --gold-dir data/gold \
    --truth data/truth/object_truth.parquet \
    --objects-csv data/labels.csv \
    --allow-unsafe-latest-snapshot \
    --max-n-det 20
echo "$(ts) gold rebuilt"

echo "$(ts) Step 2/4: build_expert_helpfulness"
python3 -u scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots.parquet \
    --output data/gold/expert_helpfulness.parquet
echo "$(ts) helpfulness rebuilt"

echo "$(ts) Step 3/4: train_expert_trust (all experts auto-discovered from helpfulness)"
python3 -u scripts/train_expert_trust.py \
    --snapshots data/gold/object_epoch_snapshots.parquet \
    --helpfulness data/gold/expert_helpfulness.parquet \
    --models-dir models/trust \
    --output-snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --metrics-out reports/metrics/expert_trust_metrics.json \
    --allow-unsafe-alerce \
    --n-jobs 4
echo "$(ts) trust heads trained"

echo "$(ts) Step 4/4: train_followup"
python3 -u scripts/train_followup.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --trust-models-dir models/trust \
    --model-dir models/followup \
    --metrics-out reports/metrics/followup_metrics.json
echo "$(ts) followup trained"

echo "$(ts) Summary — live experts:"
ls -1 models/trust/ | grep -v 'metadata\.json$' | sort
echo "$(ts) Trust heads count: $(ls -1 models/trust/ | grep -v 'metadata.json$' | wc -l)"
echo "$(ts) F3 pipeline DONE"
