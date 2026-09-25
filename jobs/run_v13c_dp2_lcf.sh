#!/bin/bash
#$ -N debass_v13c_dp2
#$ -cwd -V
#$ -l h_rt=02:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -o logs/v13c_dp2_lcf.qsub.out
#$ -e logs/v13c_dp2_lcf.qsub.err
# fusion v13c follow-up for DP2: lc_features_bv did not read the DP2 lightcurves' psfFlux/psfFluxErr (no output on any
# DP2 row). The fix landed after jobs/run_fusion_v13c_gold.sh (7739792) was submitted, and SGE runs the copy taken at
# submit time, so that job built DP2 without it. This job re-runs lc_features_bv for DP2 into silver_v13cloc (the step
# now also in run_fusion_v13c_gold.sh), rebuilds edp2_train/gold/snapshots_v13cloc.parquet (the first build is kept as
# snapshots_v13cloc_nolcf.parquet) and re-scores DP2 with v13c (the first scores kept as *_nolcf).
#   qsub -P pi-brout -hold_jid <v13c train job id> jobs/run_v13c_dp2_lcf.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-8}
S3=data/v13c_salt3
E=data/edp2_train
SFX=v13c
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"
echo "$(ts) v13c DP2 lc_features — START"
test -f "${E}/silver_v13cloc/.v13c_ready" || { echo "run jobs/run_fusion_v13c_gold.sh first"; exit 2; }

# 1. lc_features_bv for DP2, NSLOTS shards in parallel
if [[ ! -f "${E}/silver_v13cloc/.v13c_lcf_ready" ]]; then
    LD="${S3}/dp2_lcf"; pids=()
    for ((k = 0; k < NSLOTS; k++)); do
        d="${LD}/s_${k}"; [[ -f "${d}/.done" ]] && continue
        rm -rf "${d}" && mkdir -p "${d}"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore \
            python3 -u scripts/local_infer.py --expert lc_features_bv --from-labels "${S3}/dp2/s_0/ids.csv" \
            --lc-dir "${E}/lightcurves" --silver-dir "${d}" --max-n-det 20 --shard-id "${k}" --n-shards "${NSLOTS}" \
            > "${d}/local_infer.log" 2>&1 && touch "${d}/.done" &
        pids+=($!)
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "${p}" || true; done
    for ((k = 0; k < NSLOTS; k++)); do
        [[ -f "${LD}/s_${k}/.done" ]] || { echo "DP2 lc_features shard ${k} failed:"; tail -20 "${LD}/s_${k}/local_infer.log"; exit 3; }
    done
    mv "${E}/silver_v13cloc/local_expert_outputs/lc_features_bv" "${E}/silver_v13cloc/_lc_features_bv_v12"
    mkdir -p "${E}/silver_v13cloc/local_expert_outputs/lc_features_bv"
    for ((k = 0; k < NSLOTS; k++)); do
        cp "${LD}/s_${k}/local_expert_outputs/lc_features_bv/part-latest.parquet" \
            "${E}/silver_v13cloc/local_expert_outputs/lc_features_bv/part-shard${k}.parquet"
    done
    touch "${E}/silver_v13cloc/.v13c_lcf_ready"
fi
python3 - "${E}/silver_v13cloc/local_expert_outputs/lc_features_bv" <<'EOF'
import glob, json, sys
import pandas as pd
df = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{sys.argv[1]}/*.parquet")], ignore_index=True)
ok = df["class_probabilities"].map(lambda s: bool(json.loads(s)) if isinstance(s, str) else bool(s)) & df["available"].astype(bool)
print(f"DP2 lc_features_bv: {len(df):,} rows, output on {ok.mean():.1%}")
EOF

# 2. DP2 gold, rebuilt
G="${E}/gold/snapshots_v13cloc.parquet"
if [[ -f "${G}" && ! -f "${E}/gold/snapshots_v13cloc_nolcf.parquet" ]]; then mv "${G}" "${E}/gold/snapshots_v13cloc_nolcf.parquet"; fi
if [[ ! -f "${G}" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --lc-dir "${E}/lightcurves" --silver-dir "${E}/silver_v13cloc" --truth "${E}/truth/object_truth.parquet" --bts "" \
        --labels "${E}/gold/labels_empty.csv" --trust-metadata "${E}/gold/empty_trust_metadata.json" --no-lsst-weak \
        --output "${G}" --split-manifest "${E}/gold/split_v13cloc.json"
fi

# 3. re-score DP2 with v13c (as run_fusion_v13.sh step 4)
P="${E}/scores/predictions_dp2_${SFX}.parquet"
if [[ -f "${P}" && ! -f "${P%.parquet}_nolcf.parquet" ]]; then mv "${P}" "${P%.parquet}_nolcf.parquet"; fi
python3 -u scripts/score_fusion_v11.py --tag "dp2_${SFX}" --snapshots "${G}" \
    --trust-dir "models/trust_fusion_${SFX}" --followup-dir "models/followup_fusion_${SFX}" \
    --blend-dir "models/anchor_blend_${SFX}" --conformal "models/conformal_fusion_${SFX}/mondrian_aps.pkl" \
    --scores-dir "${E}/scores" --no-priority --require-local-experts
echo "$(ts) v13c DP2 lc_features — DONE"
