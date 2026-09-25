#!/bin/bash
#$ -N debass_edp2_v12
#$ -cwd -V
#$ -l h_rt=03:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -o logs/edp2_v12_score.qsub.out
#$ -e logs/edp2_v12_score.qsub.err
# DP2 (EDP2) TNS objects in metaDEBASS format (data/edp2_train, data-rights restricted: stays under the owner-only data/)
# scored with fusion v12 on lightcurve + local experts only (DP2 has no broker outputs). Same sharded local-expert run as
# jobs/run_tnsx_v12_score.sh. Output: data/edp2_train/scores/predictions_dp2_v12loc.parquet, gold/snapshots_v12loc.parquet.
# Also the input for the v13 "DP2 typed SNe, no brokers" acceptance test.
#   qsub -P pi-brout jobs/run_edp2_v12_score.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-8}
E="data/edp2_train"
TAG=${EDP2_TAG:-v12loc}
MSFX=${EDP2_MODEL_SFX:-v12}
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"
mkdir -p "${E}/scores" "${E}/gold" "${E}/_shards" logs
echo "$(ts) edp2 ${TAG} — START (${NSLOTS} slots)"

# 1. local experts: NSLOTS shards in parallel, each into its own silver dir, then merged side by side
SILVER="${E}/silver_${TAG}"
IDS="${E}/_shards/ids.csv"
if [[ ! -f "${E}/_shards/.merged" ]]; then
    { echo object_id; ls "${E}/lightcurves" | sed -n 's/\.json$//p'; } > "${IDS}"
    pids=()
    for ((k = 0; k < NSLOTS; k++)); do
        d="${E}/_shards/s_${k}"
        [[ -f "${d}/.done" ]] && continue
        mkdir -p "${d}"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            python3 -u scripts/local_infer.py --expert all --from-labels "${IDS}" --lc-dir "${E}/lightcurves" \
            --silver-dir "${d}" --max-n-det 20 --shard-id "${k}" --n-shards "${NSLOTS}" \
            > "${d}/local_infer.log" 2>&1 && touch "${d}/.done" &
        pids+=($!)
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "${p}" || true; done
    mkdir -p "${SILVER}"
    cp "${E}/silver/broker_events.parquet" "${E}/silver/broker_outputs.parquet" "${SILVER}/"
    out="${SILVER}/local_expert_outputs"
    rm -rf "${out}.tmp" && mkdir -p "${out}.tmp"
    for ((k = 0; k < NSLOTS; k++)); do
        [[ -f "${E}/_shards/s_${k}/.done" ]] || { echo "shard ${k} failed:"; tail -20 "${E}/_shards/s_${k}/local_infer.log"; exit 3; }
        for f in "${E}/_shards/s_${k}"/local_expert_outputs/*/part-latest.parquet; do
            ex=$(basename "$(dirname "${f}")")
            mkdir -p "${out}.tmp/${ex}" && cp "${f}" "${out}.tmp/${ex}/part-shard${k}.parquet"
        done
    done
    [[ -d "${out}" ]] && mv "${out}" "${out}.old.$(date +%s)"
    mv "${out}.tmp" "${out}"
    touch "${E}/_shards/.merged"
fi
echo "$(ts) local experts done: $(ls "${SILVER}/local_expert_outputs" | tr '\n' ' ')"

# 2. gold (current code; trajectory features on, as in v12 training)
python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
    --lc-dir "${E}/lightcurves" --silver-dir "${SILVER}" --truth "${E}/truth/object_truth.parquet" --bts "" \
    --labels "${E}/gold/labels_empty.csv" --trust-metadata "${E}/gold/empty_trust_metadata.json" --no-lsst-weak \
    --output "${E}/gold/snapshots_${TAG}.parquet" --split-manifest "${E}/gold/split_${TAG}.json"

# 3. score
python3 -u scripts/score_fusion_v11.py --tag "dp2_${TAG}" --snapshots "${E}/gold/snapshots_${TAG}.parquet" \
    --trust-dir "models/trust_fusion_${MSFX}" --followup-dir "models/followup_fusion_${MSFX}" \
    --blend-dir "models/anchor_blend_${MSFX}" --conformal "models/conformal_fusion_${MSFX}/mondrian_aps.pkl" \
    --scores-dir "${E}/scores" --no-priority
echo "$(ts) edp2 ${TAG} — DONE"
ls -la "${E}/scores"
