#!/bin/bash
#$ -N debass_v9c
#$ -cwd -V
#$ -l h_rt=10:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v9c.qsub.out
#$ -e logs/fusion_v9c.qsub.err
# metaDEBASS fusion_v9c — the sequence classifier as a REGISTERED EXPERT,
# integrated through the standard contract, end to end:
#
#   1. local_infer seq_v9 over every labeled object → silver per-epoch scores
#      (single sequential process — never run concurrent local_infer against
#      one silver dir; see the v6e.2 race postmortem)
#   2. gold rebuild _v9c: picks up seq_v9 proj/avail columns automatically
#      + the LSST weak-label harvest (lsst_candidates.csv → 3,998 LSST objects
#      with cached lightcurves enter train/cal as weak truth)
#   3. helpfulness rebuild (gains seq_v9 rows) → Stage A retrains with a
#      seq_v9 trust head → gates → Stage B → score → eval (reports/fusion_v9c)
#
# Locked test stays byte-identical; all new artifacts use the _v9c suffix.
# Submit via jobs/submit_fusion_v9_chain.sh (holds on the GPU pretrain job).
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
mkdir -p logs data/scores reports/fusion_v9c reports/metrics

SNAP="data/gold/object_epoch_snapshots_fusion_v9c.parquet"
SPLIT="data/gold/split_fusion_v9c.json"
HELP="data/gold/expert_helpfulness_fusion_v9c.parquet"
SNAP_TRUST="${SNAP%.parquet}_trust.parquet"

echo "$(ts) v9c — START (NSLOTS=${NSLOTS})"
test -d models/seq_classifier_v9 || { echo "missing models/seq_classifier_v9"; exit 2; }

# Steps 1-3 are idempotent: completed artifacts are reused on resubmission
# (set FUSION_V9C_FORCE=1 to rebuild everything from scratch).
FORCE="${FUSION_V9C_FORCE:-0}"

# 1. seq_v9 inference into silver (sequential; object list = union of labeled
#    sets the gold builder will use — labels.csv covers ZTF+LSST IDs)
if [[ "${FORCE}" == "1" || ! -f data/silver/local_expert_outputs/seq_v9/part-latest.parquet ]]; then
    python3 -u scripts/local_infer.py --expert seq_v9 \
        --from-labels data/labels.csv \
        --lc-dir data/lightcurves --silver-dir data/silver --max-n-det 20
else
    echo "$(ts) [skip] seq_v9 silver exists"
fi

# 2. gold + DP1 rebuild with the new expert and LSST weak labels
if [[ "${FORCE}" == "1" || ! -f "${SNAP}" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --output "${SNAP}" --split-manifest "${SPLIT}" --dp1 \
        --dp1-output data/gold/dp1_snapshots_fusion_v9c.parquet
else
    echo "$(ts) [skip] gold snapshot exists: ${SNAP}"
fi

# 3. helpfulness
if [[ "${FORCE}" == "1" || ! -f "${HELP}" ]]; then
    python3 -u scripts/build_helpfulness_fusion.py \
        --snapshots "${SNAP}" --output "${HELP}"
else
    echo "$(ts) [skip] helpfulness exists: ${HELP}"
fi

# 4. full train (Stage A retrains — seq_v9 gets a trust head; gates decide
#    every component on cal)
python3 -u scripts/train_fusion_v8.py --n-jobs "${NSLOTS}" \
    --snapshots "${SNAP}" --helpfulness "${HELP}" --split "${SPLIT}" \
    --output-snapshots "${SNAP_TRUST}" \
    --trust-dir models/trust_fusion_v9c \
    --followup-dir models/followup_fusion_v9c \
    --conformal-dir models/conformal_fusion_v9c \
    --metrics-out reports/metrics/fusion_v9c_train.json

# 5. score (gold + DP1)
python3 -u scripts/score_fusion_v8.py --tag fusion_v9c \
    --snapshots "${SNAP_TRUST}" \
    --trust-dir models/trust_fusion_v9c \
    --followup-dir models/followup_fusion_v9c \
    --conformal models/conformal_fusion_v9c/mondrian_aps.pkl
python3 -u scripts/score_fusion_v8.py --tag fusion_v9c --dp1 \
    --snapshots data/gold/dp1_snapshots_fusion_v9c.parquet \
    --trust-dir models/trust_fusion_v9c \
    --followup-dir models/followup_fusion_v9c \
    --conformal models/conformal_fusion_v9c/mondrian_aps.pkl

# 6. eval
python3 -u scripts/eval_fusion_v8.py \
    --pred data/scores/predictions_fusion_v9c.parquet \
    --pred-dp1 data/scores/predictions_fusion_v9c_dp1.parquet \
    --snapshots "${SNAP_TRUST}" --split "${SPLIT}" \
    --train-metrics reports/metrics/fusion_v9c_train.json \
    --out-dir reports/fusion_v9c

echo "$(ts) v9c — DONE"
python3 - <<'EOF'
import json, pathlib
hg = pathlib.Path("reports/fusion_v9c/headline_guards.json")
if hg.exists():
    payload = json.loads(hg.read_text())
    print("V9C HEADLINE:", json.dumps(payload.get("headline", {}), indent=2)[:900])
tr = pathlib.Path("reports/metrics/fusion_v9c_train.json")
if tr.exists():
    payload = json.loads(tr.read_text())
    for e in payload.get("component_gates", payload.get("gates", [])):
        if isinstance(e, dict):
            print("GATE:", e.get("component"), "→", e.get("decision"), e.get("delta_macro_auc"))
    sa = payload.get("stage_a", {})
    if isinstance(sa, dict) and "seq_v9" in sa:
        print("seq_v9 trust:", json.dumps(sa["seq_v9"])[:300])
EOF
