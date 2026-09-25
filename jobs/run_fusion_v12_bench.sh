#!/bin/bash
#$ -N debass_v12_bench
#$ -cwd -V
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -o logs/fusion_v12_bench.qsub.out
#$ -e logs/fusion_v12_bench.qsub.err
# Score the frozen LSST live benchmark (246 ids) + the 15-object LSST spec hold-out with fusion v12, the v12w control
# arm and v11, on the same objects and the same inputs:
#   * local experts are run for these objects (v11 and v12 both trained with local-expert outputs on LSST rows;
#     the July benchmark gold had none);
#   * Pitt-Google is absent (it is never fetched at scoring time);
#   * v12/v12w are scored on gold built with the fixed CATS projector, v11 on gold built with the pre-fix projector
#     (the one it was trained with), from a copy of scripts/ + src/ with the two old files restored from
#     _backup_pre_v12_20260924/overwritten_files.tgz.
# Bundle staged from the Mac: data/label_refresh_20260924/bench/{lightcurves/, silver/broker_events.parquet,
# truth.parquet, ids.csv}. Scores land in bench/scores/ and are evaluated on the Mac.
#   qsub -P pi-brout -hold_jid <v12w job id> jobs/run_fusion_v12_bench.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-8}
B="data/label_refresh_20260924/bench"
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"
mkdir -p "${B}/scores" "${B}/gold"
EMPTY="${B}/gold/empty_split.json"
printf '{"train_ids": [], "cal_ids": [], "test_ids": []}\n' > "${EMPTY}"
printf 'object_id\n' > "${B}/gold/labels_empty.csv"
echo "$(ts) bench — START"

# 1. local experts (sequential, into the bundle's silver)
if [[ ! -f "${B}/silver/.local_experts_done" ]]; then
    python3 -u scripts/local_infer.py --expert all --from-labels "${B}/ids.csv" \
        --lc-dir "${B}/lightcurves" --silver-dir "${B}/silver" --max-n-det 20
    touch "${B}/silver/.local_experts_done"
fi

# 2. pre-fix code copy for the v11 gold
COMPAT="_v11compat"
if [[ ! -d "${COMPAT}" ]]; then
    mkdir -p "${COMPAT}"
    cp -a scripts src "${COMPAT}/"
    tar xzf _backup_pre_v12_20260924/overwritten_files.tgz -C "${COMPAT}" \
        src/debass_meta/projectors/fink_lsst.py src/debass_meta/features/trajectory.py
    grep -q "_CATS_NONIA = {21, 31}" "${COMPAT}/src/debass_meta/projectors/fink_lsst.py" \
        || { echo "compat copy does not carry the pre-fix projector"; exit 2; }
fi

build() {  # $1 = scripts root, $2 = output
    python3 -u "$1/scripts/build_snapshots_fusion.py" --n-jobs "${NSLOTS}" \
        --lc-dir "${B}/lightcurves" --silver-dir "${B}/silver" --truth "${B}/truth.parquet" \
        --bts "" --labels "${B}/gold/labels_empty.csv" --trust-metadata "${EMPTY}" --no-lsst-weak \
        --output "$2" --split-manifest "${2%.parquet}_split.json"
}
build . "${B}/gold/bench_v12.parquet"
build "${COMPAT}" "${B}/gold/bench_v11compat.parquet"

# 3. score
score() {  # $1 = tag, $2 = gold, $3 = model suffix
    python3 -u scripts/score_fusion_v11.py --tag "$1" --snapshots "$2" \
        --trust-dir "models/trust_fusion_$3" --followup-dir "models/followup_fusion_$3" \
        --blend-dir "models/anchor_blend_$3" --conformal "models/conformal_fusion_$3/mondrian_aps.pkl" \
        --scores-dir "${B}/scores" --no-priority
}
score bench_v12 "${B}/gold/bench_v12.parquet" v12
[[ -f models/followup_fusion_v12w/model.pkl ]] && score bench_v12w "${B}/gold/bench_v12.parquet" v12w
score bench_v11 "${B}/gold/bench_v11compat.parquet" v11
echo "$(ts) bench — DONE"
ls -la "${B}/scores"
