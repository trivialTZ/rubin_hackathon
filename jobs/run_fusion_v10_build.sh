#!/bin/bash
#$ -N debass_v10_build
#$ -cwd -V
#$ -l h_rt=06:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v10_build.qsub.out
#$ -e logs/fusion_v10_build.qsub.err
# metaDEBASS fusion_v10 — stage 1 on SCC: the FINAL split manifest
# (data/gold/split_fusion_v10.json) is produced HERE, BEFORE any encoder or
# classifier training (the v10 P0 fix for the v8→v9c chain-ordering cal leak:
# the v9c classifier was fine-tuned on weak LSST objects under the v8 manifest
# and ~20% of them later landed in v9c CAL).  Both training scripts in the
# pretrain job receive this manifest.
#
# The snapshot built here is a LEAN pre-table (experts/traj skipped): its only
# consumers are the split manifest and the classifier's label columns
# (object_id / target_class / label_quality), which do not depend on expert
# blocks.  The full v10 gold table (experts + traj + DP1) is rebuilt by
# jobs/run_fusion_v10_expert.sh AFTER the OOF seq_v9 silver inference, against
# the SAME manifest (seq-train guard armed there).
#
# v10 split hygiene in build_snapshots_fusion.py:
#   * --association-csv groups LSST↔ZTF counterparts into one split cluster
#     for NEW objects (duplicate-photometry defense; missing file → no-op)
#   * --seq-train-ids (belt-and-braces, armed on REBUILDS when a fold_map
#     already exists): objects a seq model trained on are forced into train,
#     never cal; locked-test hits hard-fail.
#
# Idempotent: completed artifacts are reused (FUSION_V10_FORCE=1 rebuilds).
# Submit via jobs/submit_fusion_v10_chain.sh.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
mkdir -p logs data/gold data/scores reports/metrics

SNAP_PRE="data/gold/object_epoch_snapshots_fusion_v10_pre.parquet"
SPLIT="data/gold/split_fusion_v10.json"
FORCE="${FUSION_V10_FORCE:-0}"

echo "$(ts) v10 build — START (NSLOTS=${NSLOTS})"

# Expert-key drift repair: bronze/silver written under pre-v10 code carries
# dated ALeRCE Rubin-stamp expert keys (alerce/stamp_classifier_rubin_beta_
# YYYYMMDD) that the gold builder's exact-key lookup silently drops — the
# fetch-time canonicalization cannot rewrite history, so re-normalize
# bronze → silver whenever a non-canonical key is present in silver.
NEED_RENORM=$(python3 - <<'EOF'
import pathlib
import sys

flag = 0
try:
    import pandas as pd
    p = pathlib.Path("data/silver/broker_events.parquet")
    if p.exists():
        keys = pd.read_parquet(p, columns=["expert_key"])["expert_key"].astype(str)
        flag = int(keys.str.startswith("alerce/stamp_classifier_rubin_beta_").any())
except Exception as exc:
    print(f"renorm probe failed ({exc}) — skipping re-normalize", file=sys.stderr)
print(flag)
EOF
)
if [[ "${NEED_RENORM}" == "1" ]]; then
    echo "$(ts) dated Rubin-stamp expert keys found in silver — re-normalizing bronze → silver"
    python3 -u scripts/normalize.py
else
    echo "$(ts) [skip] silver expert keys already canonical"
fi

# Belt-and-braces: if a v10 classifier already exists (rerun of the chain),
# its train objects must never drift into cal in a rebuilt manifest.
SEQ_TRAIN_ARGS=()
if [[ -f models/seq_classifier_v10/fold_map.json ]]; then
    SEQ_TRAIN_ARGS=(--seq-train-ids models/seq_classifier_v10/fold_map.json)
    echo "$(ts) seq-train guard armed with models/seq_classifier_v10/fold_map.json"
fi

if [[ "${FORCE}" == "1" || ! -f "${SNAP_PRE}" || ! -f "${SPLIT}" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --output "${SNAP_PRE}" --split-manifest "${SPLIT}" \
        --association-csv data/crossmatch/lsst_to_ztf.csv \
        --skip-experts --skip-traj \
        "${SEQ_TRAIN_ARGS[@]}"
else
    echo "$(ts) [skip] pre-snapshot + split manifest exist: ${SNAP_PRE}, ${SPLIT}"
fi

echo "$(ts) v10 build — DONE"
python3 - <<'EOF'
import json
m = json.load(open("data/gold/split_fusion_v10.json"))
print("V10 SPLIT:", {k: m[k] for k in (
    "counts", "n_new", "n_new_test_reassigned_to_train",
    "seq_train_guard", "n_seq_train_diverted_cal_to_train",
    "association_grouped_split", "n_new_quarantined_test_counterparts",
    "n_new_forced_cal_by_association", "n_new_forced_train_by_association") if k in m})
EOF
