#!/bin/bash
#$ -N debass_f3safev5d
#$ -l h_rt=01:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/f3safev5d.qsub.out

# F3-SAFE-v5d — SN-filter fix for fink_lsst/cats + fink_lsst/snn.
#
# Background: v5c's fink_lsst/cats and fink_lsst/snn trust heads have
# inflated aggregate AUC (0.996, 0.982) dominated by easy-negative rejection
# in the AGN-heavy LSST sample, with inverted AUC on spec-Ia slices. Root
# cause: these experts' ternary projection caps p_snia at 0.5 × score,
# making them SN-vs-other filters, not Ia-vs-non-Ia classifiers. Training
# their trust heads with target=is_topclass_correct is a category error.
#
# Fix (Option A, verified in scripts/_exp_sn_filter_retrain.py):
#   SN_FILTER_EXPERTS = {fink_lsst/snn, fink_lsst/cats}
#   trust target = is_sn (int(target_class != 'other'))
#
# v5d reuses v5c truth + gold snapshots. Only helpfulness + trust + followup
# are rebuilt. Full pipeline ~10 min.
#
# Preserves v5, v5b, v5c artifacts; outputs suffixed _v5d.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) F3-SAFE-v5d — SN-filter trust target for fink_lsst/cats + fink_lsst/snn"

echo "$(ts) [1/3] build_expert_helpfulness (now emits is_sn column)"
python3 -u scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v5c.parquet \
    --output    data/gold/expert_helpfulness_safe_v5d.parquet

echo "$(ts) [2/3] train_expert_trust (cats/snn → is_sn; others → is_topclass_correct)"
python3 -u scripts/train_expert_trust.py \
    --snapshots        data/gold/object_epoch_snapshots_safe_v5c.parquet \
    --helpfulness      data/gold/expert_helpfulness_safe_v5d.parquet \
    --models-dir       models/trust_safe_v5d \
    --output-snapshots data/gold/object_epoch_snapshots_trust_safe_v5d.parquet \
    --metrics-out      reports/metrics/expert_trust_metrics_safe_v5d.json \
    --n-jobs 4

echo "$(ts) [3/3] train_followup"
python3 -u scripts/train_followup.py \
    --snapshots        data/gold/object_epoch_snapshots_trust_safe_v5d.parquet \
    --trust-models-dir models/trust_safe_v5d \
    --model-dir        models/followup_safe_v5d \
    --metrics-out      reports/metrics/followup_metrics_safe_v5d.json \
    --weak-weight 0.25

echo "$(ts) F3-SAFE-v5d DONE"
echo "  trust heads (safe-v5d):"
ls -1 models/trust_safe_v5d/ | grep -v 'metadata.json$' | sort
echo
python3 - <<PY
import json
m = json.load(open("reports/metrics/expert_trust_metrics_safe_v5d.json"))
print("v5d trust-head AUCs (raw / calibrated):")
for k in sorted(m.keys()):
    row = m[k]
    raw = row.get("raw", {}).get("roc_auc", row.get("roc_auc"))
    cal = row.get("calibrated", {}).get("roc_auc")
    print(f"  {k:40s}  raw={raw}  cal={cal}")
f = json.load(open("reports/metrics/followup_metrics_safe_v5d.json"))
print("v5d followup cal AUC:", f["test_calibrated"]["roc_auc"])
PY
