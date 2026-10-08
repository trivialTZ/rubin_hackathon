#!/bin/bash
#$ -N debass_v13k_gold
#$ -cwd -V
#$ -l h_rt=08:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v13k_gold.qsub.out
#$ -e logs/fusion_v13k_gold.qsub.err
# metaDEBASS fusion v13k, step 1 of 2 (docs/fusion_v13k_plan.md): the v13g golds rebuilt without future-epoch leaks.
#   * gold.select_events_asof no longer falls back to every timed local re-run when none is at or before the alert
#     (an epoch before an expert's first output got the expert's LATER outputs: lc_features_bv below 4 detections,
#     early SALT3 / SNGuess / SuperNNova failures); build_snapshots_fusion.py now refuses to write a gold with any
#     future-dated non-static selection.
#   * lc_features_bv retrained on the v13g split's train_ids only (it was trained on every labelled object, every
#     fusion test object included), same truth and lightcurves as before, then re-run on every set. The new outputs
#     replace the lc_features_bv partition in copies of each silver (silver_v13k / *_v13k / *_v13kloc); every other
#     expert's outputs are copied unchanged.
#   * Golds: training (v13g's split as the locked split, so no assignment moves; asserted), benchmark, hard-negative
#     TEST, explorer cohort, DP2, fresh-A. Fresh-B stays unfetched.
#   qsub -P pi-brout jobs/run_fusion_v13k_gold.sh
#   qsub -P pi-brout -hold_jid <this job id> -v FUSION_V13_ARM=v13k jobs/run_fusion_v13.sh
# FUSION_V13K_OLD_LCF=1 (isolation arm v13ka, docs/fusion_v13k_plan.md "v13k result"): the as-of fix only. No
#   lc_features_bv retrain or re-run; the golds are rebuilt from the v13g-era silvers, outputs tagged v13ka.
#   qsub -P pi-brout -v FUSION_V13K_OLD_LCF=1 jobs/run_fusion_v13k_gold.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
H=data/hardneg_20261004
B=data/label_refresh_20260924/bench
T=data/tnsx_eval_20260924
E=data/edp2_train
F=data/hardneg_fresh_20261005
UNION=data/gold/lsst_locked_union_v13g.json
SPLIT_G=data/gold/split_fusion_v13g.json
LCF=artifacts/local_experts/lc_features_v13k
SCR=data/v13k_lcf
OLD_LCF="${FUSION_V13K_OLD_LCF:-0}"
TAG=v13k; [[ "${OLD_LCF}" == "1" ]] && TAG=v13ka
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"
mkdir -p logs data/gold "${SCR}"
echo "$(ts) fusion ${TAG} gold — START (NSLOTS=${NSLOTS})"
for f in "${SPLIT_G}" "${UNION}" data/truth/object_truth.parquet data/truth/object_truth_v13g.parquet \
         data/gold/lsst_hardneg_fresh_A_20261005.json "${F}/fresh_A_truth.parquet" "${H}/test_truth.parquet"; do
    test -f "$f" || { echo "missing input $f"; exit 2; }
done

if [[ "${OLD_LCF}" != "1" ]]; then
# 1. lc_features_bv head on the train split only (the April head's truth and lightcurves otherwise)
if [[ ! -f "${LCF}/model.pkl" ]]; then
    python3 -u scripts/train_lc_features_head.py --truth data/truth/object_truth.parquet \
        --lightcurves-dir data/lightcurves_v13g --train-split "${SPLIT_G}" --output-dir "${LCF}"
fi
python3 - "${LCF}/metadata.json" "${SPLIT_G}" <<'EOF'
import json, sys
m, s = (json.load(open(p)) for p in sys.argv[1:3])
held = {str(o) for k in ("cal_ids", "test_ids", "quarantined_ids") for o in s.get(k, [])}
print(f"lc_features_v13k: {m['n_train_rows']} training rows ({m.get('n_labelled_train_split')} of "
      f"{m.get('n_labelled_before_split')} labelled objects in train_ids), held-out ids {len(held)}")
assert m.get("train_split") and m["n_train_rows"] > 1000
EOF
export DEBASS_LC_FEATURES_MODEL="${LCF}/model.pkl"

# 2. lc_features_bv re-run per set (objects = those with lc_features_bv rows in the old silver), into scratch shards;
#    the new silver = a real copy of the old one with only the lc_features_bv partition replaced
relcf() {  # $1 name, $2 old silver, $3 lightcurve dir, $4 new silver
    local name=$1 old=$2 lc=$3 new=$4 d k
    [[ -f "${new}/.v13k_ready" ]] && { echo "$(ts) ${name}: ${new} ready"; return 0; }
    mkdir -p "${SCR}/${name}"
    python3 - "${old}" "${SCR}/${name}/ids.csv" <<'EOF'
import glob, sys
import pandas as pd
old, out = sys.argv[1:3]
parts = glob.glob(f"{old}/local_expert_outputs/lc_features_bv/*.parquet")
ids = sorted(pd.concat([pd.read_parquet(p, columns=["object_id"]) for p in parts])["object_id"].astype(str).unique())
pd.DataFrame({"object_id": ids}).to_csv(out, index=False)
print(f"{len(ids)} objects")
EOF
    local pids=()
    for ((k = 0; k < NSLOTS; k++)); do
        d="${SCR}/${name}/s_${k}"
        [[ -f "${d}/.done" ]] && continue
        rm -rf "${d}" && mkdir -p "${d}"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore \
            python3 -u scripts/local_infer.py --expert lc_features_bv --from-labels "${SCR}/${name}/ids.csv" \
            --lc-dir "${lc}" --silver-dir "${d}" --max-n-det 20 --shard-id "${k}" --n-shards "${NSLOTS}" \
            > "${d}/local_infer.log" 2>&1 && touch "${d}/.done" &
        pids+=($!)
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "${p}" || true; done
    for ((k = 0; k < NSLOTS; k++)); do
        [[ -f "${SCR}/${name}/s_${k}/.done" ]] || { echo "${name} shard ${k} failed:"; tail -20 "${SCR}/${name}/s_${k}/local_infer.log"; exit 3; }
    done
    rm -rf "${new}.tmp" && cp -r "${old}" "${new}.tmp"
    rm -rf "${new}.tmp/local_expert_outputs/lc_features_bv" && mkdir -p "${new}.tmp/local_expert_outputs/lc_features_bv"
    for ((k = 0; k < NSLOTS; k++)); do
        f="${SCR}/${name}/s_${k}/local_expert_outputs/lc_features_bv/part-latest.parquet"
        [[ -f "${f}" ]] && cp "${f}" "${new}.tmp/local_expert_outputs/lc_features_bv/part-shard${k}.parquet"
    done
    python3 - "${old}" "${new}.tmp" "${LCF}/model.pkl" <<'EOF'
import glob, json, sys
import pandas as pd
old, new, model = sys.argv[1:4]
rd = lambda d: pd.concat([pd.read_parquet(p) for p in glob.glob(f"{d}/local_expert_outputs/lc_features_bv/*.parquet")])
a, b = rd(old), rd(new)
key = ["object_id", "n_det"]
a["object_id"], b["object_id"] = a.object_id.astype(str), b.object_id.astype(str)
assert set(map(tuple, a[key].values)) == set(map(tuple, b[key].values)), "lc_features_bv (object, n_det) set changed"
ne = b.class_probabilities.astype(str) != "{}"
assert not (b.available.astype(bool) & ~ne).any(), "available rows without probabilities"
print(f"lc_features_bv rows {len(b):,} (available {int(b.available.astype(bool).sum()):,}; "
      f"old available-with-empty {int((a.available.astype(bool) & (a.class_probabilities.astype(str) == '{}')).sum()):,})")
EOF
    [[ -d "${new}" ]] && mv "${new}" "${new}.old.$(date +%s)"
    mv "${new}.tmp" "${new}"
    touch "${new}/.v13k_ready"
    echo "$(ts) ${name}: ${new} written"
}
relcf train   data/silver_v13g         data/lightcurves_v13g    data/silver_v13k
relcf bench   "${B}/silver_v13c"       "${B}/lightcurves"       "${B}/silver_v13k"
relcf hardneg "${H}/silver"            "${H}/lightcurves"       "${H}/silver_v13k"
relcf tnsx_lsst "${T}/silver_lsst_v13c" "${T}/lightcurves_lsst" "${T}/silver_lsst_v13k"
relcf tnsx_ztf  "${T}/silver_ztf_v13c"  "${T}/lightcurves_ztf"  "${T}/silver_ztf_v13k"
relcf dp2     "${E}/silver_v13cloc"    "${E}/lightcurves"       "${E}/silver_v13kloc"
relcf fresh   "${F}/silver"            "${F}/lightcurves"       "${F}/silver_v13k"
fi
# silvers per set: the re-run copies (v13k) or the v13g-era originals (v13ka)
if [[ "${OLD_LCF}" == "1" ]]; then
    SV_TRAIN=data/silver_v13g; SV_B="${B}/silver_v13c"; SV_H="${H}/silver"; SV_T=v13c; SV_E="${E}/silver_v13cloc"; SV_F="${F}/silver"
else
    SV_TRAIN=data/silver_v13k; SV_B="${B}/silver_v13k"; SV_H="${H}/silver_v13k"; SV_T=v13k; SV_E="${E}/silver_v13kloc"; SV_F="${F}/silver_v13k"
fi

# 3. training gold (the builder asserts the as-of audit), v13g's split kept verbatim, DP1 + helpfulness
SNAP=data/gold/object_epoch_snapshots_fusion_${TAG}.parquet
SPLIT=data/gold/split_fusion_${TAG}.json
if [[ ! -f "${SNAP}" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --lc-dir data/lightcurves_v13g --silver-dir "${SV_TRAIN}" --truth data/truth/object_truth_v13g.parquet \
        --output "${SNAP}" --split-manifest "${SPLIT}" \
        --trust-metadata "${SPLIT_G}" \
        --association-csv data/crossmatch/lsst_to_ztf.csv \
        --seq-train-ids models/seq_classifier_v11/fold_map.json \
        --lsst-live-locked "${UNION}" --lsst-all-negative-fallback \
        --dp1 --dp1-output data/gold/dp1_snapshots_fusion_${TAG}.parquet
fi
python3 - "${SPLIT}" "${SPLIT_G}" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("train_ids", "cal_ids", "test_ids", "quarantined_ids"):
    x, y = set(map(str, a.get(k, []))), set(map(str, b.get(k, [])))
    print(f"split {k}: new {len(x):,} vs v13g {len(y):,}")
    assert x == y, f"split {k} differs from v13g"
EOF
HELP=data/gold/expert_helpfulness_fusion_${TAG}.parquet
[[ -f "${HELP}" ]] || python3 -u scripts/build_helpfulness_fusion.py --snapshots "${SNAP}" --output "${HELP}"
echo "$(ts) training gold done"

# 4. serving golds (same recipe as v13g's)
EMPTY="${B}/gold/empty_split.json"
build() {  # $1 lc dir, $2 silver, $3 truth, $4 labels, $5 trust metadata, $6 output, $7 split, then extra flags
    local lc=$1 sv=$2 tr=$3 lb=$4 md=$5 out=$6 sp=$7; shift 7
    [[ -f "${out}" ]] && return 0
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" --lc-dir "${lc}" --silver-dir "${sv}" \
        --truth "${tr}" --bts "" --labels "${lb}" --trust-metadata "${md}" --no-lsst-weak \
        --output "${out}" --split-manifest "${sp}" "$@"
}
build "${B}/lightcurves" "${SV_B}" "${B}/truth.parquet" "${B}/gold/labels_empty.csv" "${EMPTY}" \
    "${B}/gold/bench_${TAG}.parquet" "${B}/gold/bench_${TAG}_split.json" --lsst-all-negative-fallback
build "${H}/lightcurves_test" "${SV_H}" "${H}/test_truth.parquet" "${B}/gold/labels_empty.csv" "${EMPTY}" \
    "${H}/gold/test_${TAG}.parquet" "${H}/gold/test_${TAG}_split.json" --lsst-all-negative-fallback
for sv in lsst ztf; do
    build "${T}/lightcurves_${sv}" "${T}/silver_${sv}_${SV_T}" "${T}/truth/object_truth_${sv}.parquet" \
        "${T}/cohort/labels_${sv}_only.csv" "${T}/gold/empty_trust_metadata.json" \
        "${T}/gold/snapshots_${sv}_${TAG}.parquet" "${T}/gold/split_${sv}_${TAG}.json" --lsst-all-negative-fallback
done
build "${E}/lightcurves" "${SV_E}" "${E}/truth/object_truth.parquet" "${E}/gold/labels_empty.csv" \
    "${E}/gold/empty_trust_metadata.json" "${E}/gold/snapshots_${TAG}loc.parquet" "${E}/gold/split_${TAG}loc.json" \
    --lsst-all-negative-fallback
build "${F}/lightcurves" "${SV_F}" "${F}/fresh_A_truth.parquet" "${B}/gold/labels_empty.csv" "${EMPTY}" \
    "${F}/gold/fresh_A_${TAG}.parquet" "${F}/gold/fresh_A_${TAG}_split.json" --lsst-all-negative-fallback
echo "$(ts) serving golds done"

# 5. how much each rebuilt gold changed vs its v13g twin (rows, and rows whose local-expert block differs)
python3 - "${TAG}" <<'EOF'
import sys
import pandas as pd, pyarrow.parquet as pq
T = sys.argv[1]
pairs = {"train": ("data/gold/object_epoch_snapshots_fusion_v13g.parquet", "data/gold/object_epoch_snapshots_fusion_" + T + ".parquet"),
         "bench": ("data/label_refresh_20260924/bench/gold/bench_v13g.parquet", "data/label_refresh_20260924/bench/gold/bench_" + T + ".parquet"),
         "hardneg": ("data/hardneg_20261004/gold/test_v13g.parquet", "data/hardneg_20261004/gold/test_" + T + ".parquet"),
         "fresh_A": ("data/hardneg_fresh_20261005/gold/fresh_A_v13g.parquet", "data/hardneg_fresh_20261005/gold/fresh_A_" + T + ".parquet")}
key = ["object_id", "n_det", "alert_jd"]
for name, (a, b) in pairs.items():
    av = [c for c in pq.read_schema(a).names if c.startswith("avail__") and c in set(pq.read_schema(b).names)]
    ga, gb = pd.read_parquet(a, columns=key + av), pd.read_parquet(b, columns=key + av)
    m = ga.merge(gb, on=key, suffixes=("_g", "_k"))
    ch = {c[len("avail__"):]: int((m[c + "_g"].fillna(0) != m[c + "_k"].fillna(0)).sum()) for c in av}
    print(f"{name}: rows v13g {len(ga):,} {T} {len(gb):,} matched {len(m):,}; "
          f"availability flips {dict((k, v) for k, v in ch.items() if v)}")
EOF
echo "$(ts) fusion ${TAG} gold — DONE"
