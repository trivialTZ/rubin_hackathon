#!/bin/bash
#$ -N debass_tnsx_v12
#$ -cwd -V
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -o logs/tnsx_v12_score.qsub.out
#$ -e logs/tnsx_v12_score.qsub.err
# Score the TNS x EDP2 explorer cohort (data/tnsx_eval_20260924, public alert data) with fusion v12, on the inputs
# v12 was trained with: local experts run for every object (the 2026-09-24 cohort gold had none), the fixed CATS
# projector, trajectory features on. One local_infer process per shard, each into its own silver dir (local_infer
# rewrites part-latest.parquet, so shards must not share one); the shard partitions are then copied side by side
# into silver_<sv>/local_expert_outputs/<expert>/ (the builder globs */*.parquet).
# Bundle staged from the Mac: lightcurves_{lsst,ztf}/, silver_{lsst,ztf}/broker_*.parquet, truth/object_truth_{sv}.parquet,
# cohort/{ids_*.txt,labels_*_only.csv}. Pull back scores/predictions_tnsx_<sv>_v12.parquet (+ gold/ for checks).
#   qsub -P pi-brout jobs/run_tnsx_v12_score.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-8}
E="data/tnsx_eval_20260924"
TAG=${TNSX_TAG:-v12}
MSFX=${TNSX_MODEL_SFX:-v12}
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"
mkdir -p "${E}/scores" "${E}/gold" "${E}/_shards" logs
echo "$(ts) tnsx ${TAG} — START (${NSLOTS} slots)"

# 1. local experts: 1 LSST shard + (NSLOTS-1) ZTF shards in parallel
run_shards() {  # $1 = survey, $2 = n shards
    local sv=$1 n=$2
    local ids="${E}/_shards/ids_${sv}.csv"
    local -a pids=()
    { echo object_id; tr -s ' \n' '\n' < "${E}/cohort/ids_${sv}.txt" | grep -v '^$'; } > "${ids}"
    for ((k = 0; k < n; k++)); do
        local d="${E}/_shards/${sv}_${k}"
        if [[ -f "${d}/.done" ]]; then continue; fi
        mkdir -p "${d}"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            python3 -u scripts/local_infer.py --expert all --from-labels "${ids}" --lc-dir "${E}/lightcurves_${sv}" \
            --silver-dir "${d}" --max-n-det 20 --shard-id "${k}" --n-shards "${n}" \
            > "${d}/local_infer.log" 2>&1 && touch "${d}/.done" &
        pids+=($!)
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "${p}" || true; done
}
merge_shards() {  # $1 = survey, $2 = n shards
    local sv=$1 n=$2
    local out="${E}/silver_${sv}/local_expert_outputs"
    for ((k = 0; k < n; k++)); do
        [[ -f "${E}/_shards/${sv}_${k}/.done" ]] || { echo "shard ${sv}_${k} failed:"; tail -20 "${E}/_shards/${sv}_${k}/local_infer.log"; exit 3; }
    done
    rm -rf "${out}.tmp" && mkdir -p "${out}.tmp"
    for ((k = 0; k < n; k++)); do
        for f in "${E}/_shards/${sv}_${k}"/local_expert_outputs/*/part-latest.parquet; do
            local ex; ex=$(basename "$(dirname "${f}")")
            mkdir -p "${out}.tmp/${ex}" && cp "${f}" "${out}.tmp/${ex}/part-shard${k}.parquet"
        done
    done
    [[ -d "${out}" ]] && mv "${out}" "${out}.old.$(date +%s)"
    mv "${out}.tmp" "${out}"
    echo "$(ts) ${sv}: local experts merged -> ${out} ($(ls "${out}" | tr '\n' ' '))"
}
NZ=$((NSLOTS - 1))
if [[ ! -f "${E}/_shards/.merged" ]]; then
    run_shards lsst 1 &
    PL=$!
    run_shards ztf "${NZ}"
    wait "${PL}" || true
    merge_shards lsst 1
    merge_shards ztf "${NZ}"
    touch "${E}/_shards/.merged"
fi
echo "$(ts) local experts done"

# 2. gold (current code: fixed CATS projector; trajectory features on, as in v12 training)
printf '{"train_ids": [], "cal_ids": [], "test_ids": []}\n' > "${E}/gold/empty_trust_metadata.json"
for sv in lsst ztf; do
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --lc-dir "${E}/lightcurves_${sv}" --silver-dir "${E}/silver_${sv}" \
        --truth "${E}/truth/object_truth_${sv}.parquet" --bts "" \
        --labels "${E}/cohort/labels_${sv}_only.csv" \
        --trust-metadata "${E}/gold/empty_trust_metadata.json" --no-lsst-weak \
        --output "${E}/gold/snapshots_${sv}_${TAG}.parquet" --split-manifest "${E}/gold/split_${sv}_${TAG}.json"
done

# 3. score
for sv in lsst ztf; do
    python3 -u scripts/score_fusion_v11.py --tag "tnsx_${sv}_${TAG}" --snapshots "${E}/gold/snapshots_${sv}_${TAG}.parquet" \
        --trust-dir "models/trust_fusion_${MSFX}" --followup-dir "models/followup_fusion_${MSFX}" \
        --blend-dir "models/anchor_blend_${MSFX}" --conformal "models/conformal_fusion_${MSFX}/mondrian_aps.pkl" \
        --scores-dir "${E}/scores" --no-priority
done
echo "$(ts) tnsx ${TAG} — DONE"
ls -la "${E}/scores"
