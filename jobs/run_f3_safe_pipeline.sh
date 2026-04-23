#!/bin/bash
# DEBASS Light-up — F3-SAFE: no-leakage rebuild for apples-to-apples comparison vs F3e.
# Drops `--allow-unsafe-latest-snapshot` on gold build AND `--allow-unsafe-alerce` on trust.
# All outputs write to *_safe paths so F3e artifacts are preserved.
# Submit as:  qsub -N debass_f3safe -cwd -V -l h_rt=04:00:00 -l mem_per_core=16G -pe omp 4 \
#                  -o logs/f3safe.qsub.out -e logs/f3safe.qsub.err jobs/run_f3_safe_pipeline.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true

ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) ============================================================"
echo "$(ts) F3-SAFE pipeline — no-leakage baseline"
echo "$(ts) ============================================================"

echo "$(ts) Step 1/4: build_object_epoch_snapshots (safe)"
python3 -u scripts/build_object_epoch_snapshots.py \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --gold-dir data/gold \
    --truth data/truth/object_truth.parquet \
    --objects-csv data/labels.csv \
    --output data/gold/object_epoch_snapshots_safe.parquet \
    --max-n-det 20
echo "$(ts) gold-safe rebuilt"

echo "$(ts) Step 2/4: build_expert_helpfulness (safe)"
python3 -u scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_safe.parquet \
    --output data/gold/expert_helpfulness_safe.parquet
echo "$(ts) helpfulness-safe rebuilt"

echo "$(ts) Step 3/4: train_expert_trust (safe — no unsafe-alerce)"
python3 -u scripts/train_expert_trust.py \
    --snapshots data/gold/object_epoch_snapshots_safe.parquet \
    --helpfulness data/gold/expert_helpfulness_safe.parquet \
    --models-dir models/trust_safe \
    --output-snapshots data/gold/object_epoch_snapshots_trust_safe.parquet \
    --metrics-out reports/metrics/expert_trust_metrics_safe.json \
    --n-jobs 4
echo "$(ts) trust heads trained (safe)"

echo "$(ts) Step 4/4: train_followup (safe)"
python3 -u scripts/train_followup.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe.parquet \
    --trust-models-dir models/trust_safe \
    --model-dir models/followup_safe \
    --metrics-out reports/metrics/followup_metrics_safe.json
echo "$(ts) followup trained (safe)"

echo "$(ts) Summary — live experts (safe):"
ls -1 models/trust_safe/ | grep -v 'metadata\.json$' | sort
echo "$(ts) Trust heads count (safe): $(ls -1 models/trust_safe/ | grep -v 'metadata.json$' | wc -l)"

echo "$(ts) Compare: F3e (unsafe) vs F3-SAFE"
echo "$(ts)   F3e   trust heads:  $(ls -1 models/trust/ 2>/dev/null | grep -v 'metadata.json$' | wc -l)"
echo "$(ts)   SAFE  trust heads:  $(ls -1 models/trust_safe/ | grep -v 'metadata.json$' | wc -l)"
if [ -f reports/metrics/followup_metrics.json ]; then
    echo "$(ts)   F3e   followup AUC: $(python3 -c 'import json; print(json.load(open("reports/metrics/followup_metrics.json")).get("auc", "?"))')"
fi
echo "$(ts)   SAFE  followup AUC: $(python3 -c 'import json; print(json.load(open("reports/metrics/followup_metrics_safe.json")).get("auc", "?"))')"
echo "$(ts) F3-SAFE pipeline DONE"
