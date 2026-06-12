#!/bin/bash
#$ -N debass_v7_pipe
#$ -l h_rt=02:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v7_pipeline.qsub.out
#
# v7 post-gold pipeline:
#   1) (gold shards already merged to v7_raw)
#   2) Apply LSST truth patches → safe_v7
#   3) Build expert helpfulness (auto-discovers fink/slsn + ampel/snguess)
#   4) Train trust heads (13 = 11 v6e.2 + fink/slsn + ampel/snguess)
#   5) Train followup head
#   6) Rescore DP1 v6e snapshots
set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) v7 pipeline start"

echo "$(ts) [1/5] build_lsst_truth (stamp_only)"
$PY scripts/build_lsst_truth.py \
    --snapshot   data/gold/object_epoch_snapshots_v7_raw.parquet \
    --stamp      data/truth_candidates/alerce_stamp_lsst.parquet \
    --tier2-mode stamp_only \
    --output     data/truth_candidates/object_truth_lsst_v7.parquet

echo "$(ts) [2/5] patch_snapshot_truth_lsst -> object_epoch_snapshots_safe_v7.parquet"
$PY scripts/patch_snapshot_truth_lsst.py \
    --snapshot-in  data/gold/object_epoch_snapshots_v7_raw.parquet \
    --lsst-truth   data/truth_candidates/object_truth_lsst_v7.parquet \
    --snapshot-out data/gold/object_epoch_snapshots_safe_v7.parquet

echo "$(ts) [3/5] build_expert_helpfulness"
$PY scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v7.parquet \
    --output    data/gold/expert_helpfulness_safe_v7.parquet

echo "$(ts) [4/5] train_expert_trust + train_followup"
$PY scripts/train_expert_trust.py \
    --snapshots        data/gold/object_epoch_snapshots_safe_v7.parquet \
    --helpfulness      data/gold/expert_helpfulness_safe_v7.parquet \
    --models-dir       models/trust_safe_v7 \
    --output-snapshots data/gold/object_epoch_snapshots_trust_safe_v7.parquet \
    --metrics-out      reports/metrics/expert_trust_metrics_safe_v7.json \
    --n-jobs 4

$PY scripts/train_followup.py \
    --snapshots        data/gold/object_epoch_snapshots_trust_safe_v7.parquet \
    --trust-models-dir models/trust_safe_v7 \
    --model-dir        models/followup_safe_v7 \
    --metrics-out      reports/metrics/followup_metrics_safe_v7.json \
    --weak-weight 0.25

echo "$(ts) [5/5] rescore DP1 v6e snapshots with v7 heads"
$PY scripts/score_dp1_v5d.py \
    --snapshots    data/gold/dp1_snapshots_50k_v6e.parquet \
    --trust-dir    models/trust_safe_v7 \
    --followup-dir models/followup_safe_v7 \
    --out-dir      reports/v7_dp1_50k

echo "$(ts) v7 DONE"
echo
echo "  trust heads: $(ls -1 models/trust_safe_v7/ | grep -v 'metadata.json$' | wc -l)"
ls -1 models/trust_safe_v7/ | grep -v 'metadata.json$' | sort
echo
$PY - <<'PY'
import json
import pandas as pd
m = json.load(open("reports/metrics/expert_trust_metrics_safe_v7.json"))
print("v7 trust-head AUCs (raw / calibrated):")
for k in sorted(m.keys()):
    row = m[k]
    raw = row.get("raw", {}).get("roc_auc", row.get("roc_auc"))
    cal = row.get("calibrated", {}).get("roc_auc")
    print(f"  {k:40s}  raw={raw}  cal={cal}")
f = json.load(open("reports/metrics/followup_metrics_safe_v7.json"))
print("v7 followup cal AUC:", f["test_calibrated"]["roc_auc"])
print()
p = pd.read_parquet("reports/v7_dp1_50k/predictions.parquet")
print("v7 ensemble_n_trusted_experts on DP1:")
print(p["ensemble_n_trusted_experts"].value_counts().sort_index())
PY
