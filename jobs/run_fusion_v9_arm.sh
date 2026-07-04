#!/bin/bash
#$ -N debass_v9_arm
#$ -cwd -V
#$ -l h_rt=06:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v9_arm.qsub.out
#$ -e logs/fusion_v9_arm.qsub.err
# metaDEBASS fusion_v9 — sequence arm on SCC.  Runs AFTER the v8 pipeline
# (needs its trust snapshot + Stage-A artifacts) and the GPU pretrain job
# (needs models/seq_encoder_v9).  Submit via jobs/submit_fusion_v9_chain.sh.
#
# Steps: frozen seq-feature export + join → v9 Stage-B train (Stage A reused,
# component gates incl. seq_features) → score (gold + DP1, tag fusion_v9) →
# eval into reports/fusion_v9.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
mkdir -p logs data/scores reports/fusion_v9 reports/metrics

SNAP_TRUST="data/gold/object_epoch_snapshots_fusion_v8_trust.parquet"
SNAP_V9="data/gold/object_epoch_snapshots_fusion_v9_trust.parquet"
DP1_V8="data/gold/dp1_snapshots_fusion_v8.parquet"
DP1_V9="data/gold/dp1_snapshots_fusion_v9.parquet"
SPLIT="data/gold/split_fusion_v8.json"

echo "$(ts) v9 arm — START (NSLOTS=${NSLOTS})"
test -f "${SNAP_TRUST}" || { echo "missing ${SNAP_TRUST} (v8 train not finished?)"; exit 2; }
test -d models/seq_encoder_v9 || { echo "missing models/seq_encoder_v9 (pretrain not finished?)"; exit 2; }

# 1. frozen sequence features → joined v9 snapshots (gold + DP1)
python3 -u scripts/build_seq_features.py \
    --snapshots "${SNAP_TRUST}" \
    --out data/gold/seq_features_v9.parquet \
    --joined-out "${SNAP_V9}" --device auto
python3 -u scripts/build_seq_features.py --dp1 \
    --snapshots "${DP1_V8}" \
    --joined-out "${DP1_V9}" --device auto

# 2. v9 Stage-B train (Stage A reused from the v8 trust snapshot; gates decide
#    seq_features alongside ext/traj/q/aug on cal)
python3 -u scripts/train_fusion_v8.py --n-jobs "${NSLOTS}" \
    --snapshots data/gold/object_epoch_snapshots_fusion_v8.parquet \
    --helpfulness data/gold/expert_helpfulness_fusion_v8.parquet \
    --split "${SPLIT}" \
    --skip-stage-a \
    --output-snapshots "${SNAP_V9}" \
    --trust-dir models/trust_fusion_v8 \
    --followup-dir models/followup_fusion_v9 \
    --conformal-dir models/conformal_fusion_v9 \
    --metrics-out reports/metrics/fusion_v9_train.json

# 3. score (gold + DP1)
python3 -u scripts/score_fusion_v8.py --tag fusion_v9 \
    --snapshots "${SNAP_V9}" \
    --followup-dir models/followup_fusion_v9 \
    --conformal models/conformal_fusion_v9/mondrian_aps.pkl
python3 -u scripts/score_fusion_v8.py --tag fusion_v9 --dp1 \
    --snapshots "${DP1_V9}" \
    --followup-dir models/followup_fusion_v9 \
    --conformal models/conformal_fusion_v9/mondrian_aps.pkl

# 4. eval
python3 -u scripts/eval_fusion_v8.py \
    --pred data/scores/predictions_fusion_v9.parquet \
    --pred-dp1 data/scores/predictions_fusion_v9_dp1.parquet \
    --snapshots "${SNAP_V9}" --split "${SPLIT}" \
    --train-metrics reports/metrics/fusion_v9_train.json \
    --out-dir reports/fusion_v9

echo "$(ts) v9 arm — DONE"
python3 - <<'EOF'
import json, pathlib
p = pathlib.Path("reports/fusion_v9/headline_guards.json")
if p.exists():
    hg = json.loads(p.read_text())
    print("V9 HEADLINE:", json.dumps(hg.get("headline", {}), indent=2)[:900])
ledger = pathlib.Path("reports/metrics/fusion_v9_train.json")
if ledger.exists():
    tr = json.loads(ledger.read_text())
    for e in tr.get("component_gates", tr.get("gates", [])):
        if isinstance(e, dict):
            print("GATE:", e.get("component"), "→", e.get("decision"), e.get("delta_macro_auc"))
EOF
