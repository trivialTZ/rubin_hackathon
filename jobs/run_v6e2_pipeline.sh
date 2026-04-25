#!/bin/bash
#$ -N debass_v6e2_pipe
#$ -l h_rt=02:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v6e2_pipeline.qsub.out
#
# v6e.2 post-gold pipeline:
#   1) Merge gold shards
#   2) Apply v5b/v5c LSST truth patches (re-uses v5c truth pipeline)
#   3) Build expert helpfulness
#   4) Train trust heads (auto-discovers all 11 experts)
#   5) Train followup head (--weak-weight 0.25 to match v5d)
#   6) Rescore DP1 v6e snapshots with new heads
#
# Pre-reqs:
#   - Gold shards data/gold/object_epoch_snapshots_v6e2.shard-{0..7}-of-8.parquet exist
#   - All v5b/v5c truth-input files exist on SCC (alerce_stamp_lsst.parquet etc.)
#   - data/gold/dp1_snapshots_50k_v6e.parquet exists (from v6e Tier A run)
set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) v6e.2 pipeline start"

echo "$(ts) [1/6] merge gold shards"
$PY scripts/merge_dp1_snapshot_shards.py \
    --shard-glob 'data/gold/object_epoch_snapshots_v6e2.shard-*-of-8.parquet' \
    --out        data/gold/object_epoch_snapshots_v6e2_raw.parquet

echo "$(ts) [2/6] build_lsst_truth (stamp_only — same as v5c)"
$PY scripts/build_lsst_truth.py \
    --snapshot   data/gold/object_epoch_snapshots_v6e2_raw.parquet \
    --stamp      data/truth_candidates/alerce_stamp_lsst.parquet \
    --tier2-mode stamp_only \
    --output     data/truth_candidates/object_truth_lsst_v6e2.parquet

echo "$(ts) [3/6] patch_snapshot_truth_lsst -> object_epoch_snapshots_safe_v6e2.parquet"
$PY scripts/patch_snapshot_truth_lsst.py \
    --snapshot-in  data/gold/object_epoch_snapshots_v6e2_raw.parquet \
    --lsst-truth   data/truth_candidates/object_truth_lsst_v6e2.parquet \
    --snapshot-out data/gold/object_epoch_snapshots_safe_v6e2.parquet

echo "$(ts) [4/6] build_expert_helpfulness"
$PY scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v6e2.parquet \
    --output    data/gold/expert_helpfulness_safe_v6e2.parquet

echo "$(ts) [5/6] train_expert_trust + train_followup"
$PY scripts/train_expert_trust.py \
    --snapshots        data/gold/object_epoch_snapshots_safe_v6e2.parquet \
    --helpfulness      data/gold/expert_helpfulness_safe_v6e2.parquet \
    --models-dir       models/trust_safe_v6e2 \
    --output-snapshots data/gold/object_epoch_snapshots_trust_safe_v6e2.parquet \
    --metrics-out      reports/metrics/expert_trust_metrics_safe_v6e2.json \
    --n-jobs 4

$PY scripts/train_followup.py \
    --snapshots        data/gold/object_epoch_snapshots_trust_safe_v6e2.parquet \
    --trust-models-dir models/trust_safe_v6e2 \
    --model-dir        models/followup_safe_v6e2 \
    --metrics-out      reports/metrics/followup_metrics_safe_v6e2.json \
    --weak-weight 0.25

echo "$(ts) [6/6] rescore DP1 v6e snapshots with v6e2 heads"
$PY scripts/score_dp1_v5d.py \
    --snapshots    data/gold/dp1_snapshots_50k_v6e.parquet \
    --trust-dir    models/trust_safe_v6e2 \
    --followup-dir models/followup_safe_v6e2 \
    --out-dir      reports/v6e2_dp1_50k

echo "$(ts) v6e.2 DONE"
echo
echo "  trust heads: $(ls -1 models/trust_safe_v6e2/ | grep -v 'metadata.json$' | wc -l)"
ls -1 models/trust_safe_v6e2/ | grep -v 'metadata.json$' | sort
echo
$PY - <<'PY'
import json
import pandas as pd
m = json.load(open("reports/metrics/expert_trust_metrics_safe_v6e2.json"))
print("v6e.2 trust-head AUCs (raw / calibrated):")
for k in sorted(m.keys()):
    row = m[k]
    raw = row.get("raw", {}).get("roc_auc", row.get("roc_auc"))
    cal = row.get("calibrated", {}).get("roc_auc")
    print(f"  {k:40s}  raw={raw}  cal={cal}")
f = json.load(open("reports/metrics/followup_metrics_safe_v6e2.json"))
print("v6e.2 followup cal AUC:", f["test_calibrated"]["roc_auc"])
print()
p = pd.read_parquet("reports/v6e2_dp1_50k/predictions.parquet")
print("v6e.2 ensemble_n_trusted_experts on DP1:")
print(p["ensemble_n_trusted_experts"].value_counts().sort_index())
PY
