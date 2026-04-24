#!/bin/bash
#$ -N debass_f3safev5c
#$ -l h_rt=01:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/f3safev5c.qsub.out

# F3-SAFE-v5c — circularity fix.
# Tier-2 LSST truth now stamp-only (no broker-consensus feedback).
# Followup trainer downweights weak/context labels so spectroscopic truth drives
# the Ia/non-Ia decision boundary.
# Preserves v5 and v5b artifacts — all outputs go to _v5c suffixed files.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) F3-SAFE-v5c — stamp-only Tier 2 + weak-weight 0.25"

# --- Stage 1: rebuild LSST truth with stamp-only Tier 2 ---
echo "$(ts) [1/5] build_lsst_truth (stamp_only) -> object_truth_lsst_v5c.parquet"
python3 -u scripts/build_lsst_truth.py \
    --snapshot data/gold/object_epoch_snapshots_safe_v5.parquet \
    --stamp    data/truth_candidates/alerce_stamp_lsst.parquet \
    --tier2-mode stamp_only \
    --output   data/truth_candidates/object_truth_lsst_v5c.parquet

echo "$(ts) [2/5] merge_lsst_truth -> object_truth_v5c.parquet"
python3 -u scripts/merge_lsst_truth.py \
    --lsst    data/truth_candidates/object_truth_lsst_v5c.parquet \
    --output  data/truth/object_truth_v5c.parquet

echo "$(ts) [3/5] patch_snapshot_truth_lsst -> object_epoch_snapshots_safe_v5c.parquet"
python3 -u scripts/patch_snapshot_truth_lsst.py \
    --snapshot-in  data/gold/object_epoch_snapshots_safe_v5.parquet \
    --lsst-truth   data/truth_candidates/object_truth_lsst_v5c.parquet \
    --snapshot-out data/gold/object_epoch_snapshots_safe_v5c.parquet

# --- Stage 2: rebuild helpfulness, retrain trust + followup ---
echo "$(ts) [4/5] build_expert_helpfulness"
python3 -u scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v5c.parquet \
    --output    data/gold/expert_helpfulness_safe_v5c.parquet

echo "$(ts) [5/5] train_expert_trust + train_followup"
python3 -u scripts/train_expert_trust.py \
    --snapshots        data/gold/object_epoch_snapshots_safe_v5c.parquet \
    --helpfulness      data/gold/expert_helpfulness_safe_v5c.parquet \
    --models-dir       models/trust_safe_v5c \
    --output-snapshots data/gold/object_epoch_snapshots_trust_safe_v5c.parquet \
    --metrics-out      reports/metrics/expert_trust_metrics_safe_v5c.json \
    --n-jobs 4

python3 -u scripts/train_followup.py \
    --snapshots        data/gold/object_epoch_snapshots_trust_safe_v5c.parquet \
    --trust-models-dir models/trust_safe_v5c \
    --model-dir        models/followup_safe_v5c \
    --metrics-out      reports/metrics/followup_metrics_safe_v5c.json \
    --weak-weight 0.25

echo "$(ts) F3-SAFE-v5c DONE"
echo "  trust heads (safe-v5c): $(ls -1 models/trust_safe_v5c/ | grep -v 'metadata.json$' | wc -l)"
ls -1 models/trust_safe_v5c/ | grep -v 'metadata.json$' | sort
python3 -c 'import json; d=json.load(open("reports/metrics/followup_metrics_safe_v5c.json")); print("followup cal AUC:", d["test_calibrated"]["roc_auc"])'
