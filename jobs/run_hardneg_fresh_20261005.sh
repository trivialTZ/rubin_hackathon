#!/bin/bash -l
#$ -N hn_fresh
#$ -cwd -V
#$ -l h_rt=12:00:00
#$ -l mem_per_core=4G
#$ -pe omp 16
#$ -o logs/hardneg_fresh_20261005.qsub.out
#$ -e logs/hardneg_fresh_20261005.qsub.err
# metaDEBASS fresh hard-negative test (docs/fusion_v13g_plan.md, "v13j and a fresh test"): fresh-A of the never-fetched
# TEST-routed reserve (data/gold/lsst_hardneg_fresh_A_20261005.json, built by data/hardneg_fresh_20261005/tools/
# make_fresh.py). Alert data (ALeRCE, Fink, Lasair, Babamul; lightcurves; fetched on a login node), silver, every local expert, the serving
# gold with the v13g recipe (all-negative fallback), then v13f, v13g and v13j scored once, full inputs.
# fresh-B (data/gold/lsst_hardneg_fresh_B_20261005.json) is never touched here.
#   qsub -P pi-brout jobs/run_hardneg_fresh_20261005.sh   (step 5 waits for the v13j run to finish)
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
set -a; source .env 2>/dev/null || true; set +a
export XDG_CACHE_HOME=/projectnb/pi-brout/tztang/cache MPLCONFIGDIR=/projectnb/pi-brout/tztang/cache/mpl \
       ASTROPY_CACHE_DIR=/projectnb/pi-brout/tztang/cache/astropy PIP_CACHE_DIR=/projectnb/pi-brout/tztang/cache/pip
mkdir -p "${MPLCONFIGDIR}" "${ASTROPY_CACHE_DIR}"
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
F=data/hardneg_fresh_20261005
B=data/label_refresh_20260924/bench
MAN=data/gold/lsst_hardneg_fresh_A_20261005.json
MODELS=(v13f v13g v13j)
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"
mkdir -p logs "${F}/gold" "${F}/_shards"
echo "$(ts) fresh hard negatives — START (NSLOTS=${NSLOTS})"
for f in "${MAN}" "${F}/fresh_A_truth.parquet" "${F}/cohort/ids_lsst.txt"; do test -f "$f" || { echo "missing $f"; exit 2; }; done
python3 - "${MAN}" "${F}" <<'EOF'
import json, sys
import pandas as pd
man, f = sys.argv[1:3]
ids = {str(o) for o in json.load(open(man))["test_ids"]}
assert ids == set(open(f"{f}/cohort/ids_lsst.txt").read().split()), "fetch list differs from the fresh-A manifest"
assert ids == set(pd.read_parquet(f"{f}/fresh_A_truth.parquet").object_id.astype(str)), "truth differs from the manifest"
print("fresh-A:", len(ids), "objects")
EOF
# 1. alert data: fetched on a login node beforehand (some compute queues, e.g. e8, have no outbound network):
#    (cd data/hardneg_fresh_20261005 && nice python3 -u tools/run_fetch.py --survey lsst --brokers alerce,fink,lasair,babamul --parallel 3 --jobs 1)
#    every (step, chunk) must have its marker
n_ids=$(wc -l < "${F}/cohort/ids_lsst.txt"); n_chunks=$(( (n_ids + 39) / 40 ))
n_done=$(ls "${F}/logs/done" 2>/dev/null | wc -l)
[[ "${n_done}" -eq $(( 5 * n_chunks )) ]] || { echo "fetch incomplete: ${n_done} of $(( 5 * n_chunks )) markers"; exit 2; }
n_lc=$(ls "${F}/lightcurves" | wc -l); echo "$(ts) lightcurves: ${n_lc} of ${n_ids}"

# 2. silver
python3 -u scripts/normalize.py --bronze-dir "${F}/bronze_lsst" --silver-dir "${F}/silver"
python3 - "${F}" <<'EOF'
import sys
import pandas as pd
f = sys.argv[1]
ev = pd.read_parquet(f"{f}/silver/broker_events.parquet")
print("silver events", len(ev), "objects", ev.object_id.astype(str).nunique(),
      "per broker", ev.groupby("broker").object_id.nunique().to_dict())
pd.DataFrame({"object_id": open(f"{f}/cohort/ids_lsst.txt").read().split()}).to_csv(f"{f}/_shards/ids_all.csv", index=False)
EOF

# 3. local experts, one process per shard (local_infer rewrites part-latest.parquet, so shards never share a silver dir)
if [[ ! -f "${F}/_shards/.merged" ]]; then
    pids=()
    for ((k = 0; k < NSLOTS; k++)); do
        d="${F}/_shards/s_${k}"
        [[ -f "${d}/.done" ]] && continue
        rm -rf "${d}" && mkdir -p "${d}"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore \
            python3 -u scripts/local_infer.py --expert all --from-labels "${F}/_shards/ids_all.csv" \
            --lc-dir "${F}/lightcurves" --silver-dir "${d}" --max-n-det 20 --shard-id "${k}" --n-shards "${NSLOTS}" \
            > "${d}/local_infer.log" 2>&1 && touch "${d}/.done" &
        pids+=($!)
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "${p}" || true; done
    out="${F}/silver/local_expert_outputs"
    rm -rf "${out}.tmp" && mkdir -p "${out}.tmp"
    for ((k = 0; k < NSLOTS; k++)); do
        [[ -f "${F}/_shards/s_${k}/.done" ]] || { echo "shard ${k} failed:"; tail -20 "${F}/_shards/s_${k}/local_infer.log"; exit 3; }
        for f in "${F}/_shards/s_${k}"/local_expert_outputs/*/part-latest.parquet; do
            ex=$(basename "$(dirname "${f}")")
            mkdir -p "${out}.tmp/${ex}" && cp "${f}" "${out}.tmp/${ex}/part-shard${k}.parquet"
        done
    done
    [[ -d "${out}" ]] && mv "${out}" "${out}.old.$(date +%s)"
    mv "${out}.tmp" "${out}"
    touch "${F}/_shards/.merged"
fi
echo "$(ts) local experts: $(ls "${F}/silver/local_expert_outputs" | tr '\n' ' ')"

# 4. serving gold, the v13g recipe
GOLD="${F}/gold/fresh_A_v13g.parquet"
[[ -f "${GOLD}" ]] || python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" --lc-dir "${F}/lightcurves" \
    --silver-dir "${F}/silver" --truth "${F}/fresh_A_truth.parquet" --bts "" --labels "${B}/gold/labels_empty.csv" \
    --trust-metadata "${B}/gold/empty_split.json" --no-lsst-weak --output "${GOLD}" \
    --split-manifest "${F}/gold/fresh_A_v13g_split.json" --lsst-all-negative-fallback

# 5. v13f, v13g, v13j scored once, full inputs (v13j: wait for the last file its run writes, at most 3 h)
for ((i = 0; i < 180; i++)); do [[ -f data/tnsx_eval_20260924/scores/predictions_tnsx_ztf_v13j.parquet ]] && break; sleep 60; done
for m in "${MODELS[@]}"; do test -f "models/followup_fusion_${m}/model.pkl" || { echo "no model ${m}"; exit 2; }; done
ABL=(); for m in "${MODELS[@]}"; do ABL+=(--model "${m}"); done
python3 -u scripts/eval_input_ablation.py --gold "${GOLD}" --truth "${F}/fresh_A_truth.parquet" --manifest "${MAN}" \
    "${ABL[@]}" --variant full --out-dir "${F}/ablate"
echo "$(ts) fresh hard negatives — DONE"
