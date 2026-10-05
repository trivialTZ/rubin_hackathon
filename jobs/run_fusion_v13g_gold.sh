#!/bin/bash
#$ -N debass_v13g_gold
#$ -cwd -V
#$ -l h_rt=08:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v13g_gold.qsub.out
#$ -e logs/fusion_v13g_gold.qsub.err
# metaDEBASS fusion v13g, step 1 of 2 (docs/fusion_v13g_plan.md): hard-negative local experts, the v13g training gold
# and the serving golds; scores the deployed v13f on the hard-negative test set.
#   * Hard negatives (data/hardneg_20261004, staged from the Mac: lightcurves/, silver/broker_events.parquet,
#     test_truth.parquet, train_truth.parquet; manifest data/gold/lsst_hardneg_test_20261004.json): every local expert
#     for every object, one local_infer process per shard (local_infer rewrites part-latest.parquet, so shards never
#     share a silver dir), then the partitions side by side in silver/local_expert_outputs/<expert>/.
#   * Training gold: v13c's lightcurves / silver / truth plus the hard-negative TRAIN half, built with
#     --lsst-all-negative-fallback (keeps the Rubin spectroscopic SNe whose alert lightcurve has only negative
#     detections) and the v13c split as the locked split, so every v13c train/cal/test assignment stays verbatim and
#     only new objects are routed. The frozen benchmark and the hard-negative TEST ids are both quarantined (G6 on
#     their union).
#   * Serving golds with the same fallback: benchmark, hard-negative TEST, explorer cohort, DP2; plus the hard-negative
#     TEST gold without it (the v13f recipe).
#   qsub -P pi-brout jobs/run_fusion_v13g_gold.sh
#   qsub -P pi-brout -hold_jid <this job id> -v FUSION_V13_ARM=v13g jobs/run_fusion_v13.sh
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
HN_MANIFEST=data/gold/lsst_hardneg_test_20261004.json
UNION=data/gold/lsst_locked_union_v13g.json
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"
mkdir -p logs data/gold "${H}/gold" "${H}/_shards"
echo "$(ts) fusion v13g gold — START (NSLOTS=${NSLOTS})"
for f in "${H}/test_truth.parquet" "${H}/train_truth.parquet" "${H}/silver/broker_events.parquet" "${HN_MANIFEST}" \
         data/gold/split_fusion_v13c.json data/truth/object_truth_v12_merged.parquet; do
    test -f "$f" || { echo "missing input $f"; exit 2; }
done

# 0. id lists; the TEST and TRAIN lightcurves split into their own dirs (the builder takes every lightcurve stem that
#    has truth, so a TEST lightcurve must never sit in a training lightcurve dir)
python3 - "${H}" "${HN_MANIFEST}" <<'EOF'
import json, os, sys
import pandas as pd
h, man = sys.argv[1:3]
test = pd.read_parquet(f"{h}/test_truth.parquet")["object_id"].astype(str)
train = pd.read_parquet(f"{h}/train_truth.parquet")["object_id"].astype(str)
frozen = {str(o) for o in json.load(open(man))["test_ids"]}
assert set(test) == frozen, "test_truth ids differ from the frozen hard-negative manifest"
assert not set(test) & set(train), "hard-negative TEST and TRAIN overlap"
pd.DataFrame({"object_id": pd.concat([test, train])}).to_csv(f"{h}/_shards/ids_all.csv", index=False)
for name, ids in (("lightcurves_test", test), ("lightcurves_train", train)):
    os.makedirs(f"{h}/{name}", exist_ok=True)
    n = 0
    for o in ids:
        src, dst = f"{h}/lightcurves/{o}.json", f"{h}/{name}/{o}.json"
        if os.path.exists(src) and not os.path.exists(dst):
            os.link(src, dst)
        n += os.path.exists(dst)
    print(f"{name}: {n} of {len(ids)} lightcurves")
EOF

# 1. local experts for every hard-negative object
if [[ ! -f "${H}/_shards/.merged" ]]; then
    pids=()
    for ((k = 0; k < NSLOTS; k++)); do
        d="${H}/_shards/s_${k}"
        [[ -f "${d}/.done" ]] && continue
        rm -rf "${d}" && mkdir -p "${d}"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore \
            python3 -u scripts/local_infer.py --expert all --from-labels "${H}/_shards/ids_all.csv" \
            --lc-dir "${H}/lightcurves" --silver-dir "${d}" --max-n-det 20 --shard-id "${k}" --n-shards "${NSLOTS}" \
            > "${d}/local_infer.log" 2>&1 && touch "${d}/.done" &
        pids+=($!)
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "${p}" || true; done
    out="${H}/silver/local_expert_outputs"
    rm -rf "${out}.tmp" && mkdir -p "${out}.tmp"
    for ((k = 0; k < NSLOTS; k++)); do
        [[ -f "${H}/_shards/s_${k}/.done" ]] || { echo "shard ${k} failed:"; tail -20 "${H}/_shards/s_${k}/local_infer.log"; exit 3; }
        for f in "${H}/_shards/s_${k}"/local_expert_outputs/*/part-latest.parquet; do
            ex=$(basename "$(dirname "${f}")")
            mkdir -p "${out}.tmp/${ex}" && cp "${f}" "${out}.tmp/${ex}/part-shard${k}.parquet"
        done
    done
    [[ -d "${out}" ]] && mv "${out}" "${out}.old.$(date +%s)"
    mv "${out}.tmp" "${out}"
    touch "${H}/_shards/.merged"
fi
echo "$(ts) hard negatives: local experts in ${H}/silver/local_expert_outputs ($(ls "${H}/silver/local_expert_outputs" | tr '\n' ' '))"

# 2. training inputs: v13c's plus the hard-negative TRAIN half
python3 - "${H}" "${UNION}" "${HN_MANIFEST}" <<'EOF'
import glob, json, os, shutil, sys
import pandas as pd
h, union, man = sys.argv[1:4]
train = set(pd.read_parquet(f"{h}/train_truth.parquet")["object_id"].astype(str))
# 2a. quarantine manifest: frozen benchmark test ids + hard-negative test ids
bench = json.load(open("data/gold/lsst_live_locked_test.json"))
hn = json.load(open(man))
ids = sorted({str(o) for o in bench["test_ids"]} | {str(o) for o in hn["test_ids"]})
json.dump({"test_ids": ids, "source": "lsst_live_locked_test.json test_ids + lsst_hardneg_test_20261004.json test_ids",
           "policy": "fusion v13g build-time quarantine (G6) only; not an evaluation manifest"}, open(union, "w"), indent=1)
print(f"quarantine union: {len(ids)} ids")
# 2b. truth
t = pd.read_parquet("data/truth/object_truth_v12_merged.parquet")
t["object_id"] = t["object_id"].astype(str)
n = pd.read_parquet(f"{h}/train_truth.parquet")
n["object_id"] = n["object_id"].astype(str)
# A few hard negatives sit in the v12 truth as class-less TNS AT matches (stale_xmatch / tns_untyped, no lightcurve
# there): the hard-negative row replaces them. A classified overlap would be a label conflict.
both = t["object_id"].isin(set(n["object_id"]))
assert t.loc[both, "final_class_ternary"].isna().all(), "hard-negative TRAIN id has a class in the training truth"
print(f"truth v13g: {int(both.sum())} class-less v12 rows replaced by hard-negative rows")
t = t[~both]
n = n[[c for c in n.columns if c in t.columns]]
out = pd.concat([t, n], ignore_index=True)
out.to_parquet("data/truth/object_truth_v13g.parquet", index=False)
print(f"truth v13g: {len(t):,} + {len(n):,} hard-negative TRAIN rows")
# 2c. lightcurves: hard links of v12's files (subdirectories such as dp1/ as symlinks) plus the TRAIN lightcurves;
#     a marker file, so a half-built dir from an interrupted run is never reused
dst = "data/lightcurves_v13g"
if not os.path.exists(f"{dst}/.v13g_ready"):
    if os.path.exists(dst):
        shutil.move(dst, f"{dst}.partial")
    os.makedirs(dst)
    for p in glob.glob("data/lightcurves_v12/*"):
        q = f"{dst}/{os.path.basename(p)}"
        if os.path.isdir(p):
            os.symlink(os.path.abspath(p), q)
        else:
            os.link(p, q)
    for p in glob.glob(f"{h}/lightcurves_train/*.json"):
        q = f"{dst}/{os.path.basename(p)}"
        if not os.path.exists(q):
            os.link(p, q)
    open(f"{dst}/.v13g_ready", "w").close()
print(f"lightcurves v13g: {len(os.listdir(dst)):,} entries")
# 2d. silver: a real copy of v13c's (local_infer rewrites in place) plus the TRAIN rows
dst = "data/silver_v13g"
if not os.path.exists(f"{dst}/.v13g_ready"):
    if os.path.exists(dst):
        shutil.move(dst, f"{dst}.old")
    shutil.copytree("data/silver_v13c", dst, symlinks=True)
    be = pd.read_parquet(f"{dst}/broker_events.parquet")
    nb = pd.read_parquet(f"{h}/silver/broker_events.parquet")
    nb = nb[nb["object_id"].astype(str).isin(train)]
    nb = nb[[c for c in nb.columns if c in be.columns]].copy()
    for c in nb.columns:   # the v13c silver stores every column as pandas "string" (alert_id '1.70...e+17', projections '0.714')
        if str(be[c].dtype) == "string" and str(nb[c].dtype) != "string":
            nb[c] = nb[c].astype("string")
    pd.concat([be, nb], ignore_index=True).to_parquet(f"{dst}/broker_events.parquet", index=False)
    print(f"silver v13g broker events: {len(be):,} + {len(nb):,}")
    for d in sorted(glob.glob(f"{h}/silver/local_expert_outputs/*")):
        ex = os.path.basename(d)
        parts = [pd.read_parquet(p) for p in glob.glob(f"{d}/*.parquet")]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df[df["object_id"].astype(str).isin(train)]
        os.makedirs(f"{dst}/local_expert_outputs/{ex}", exist_ok=True)
        df.to_parquet(f"{dst}/local_expert_outputs/{ex}/part-hardneg.parquet", index=False)
        print(f"silver v13g {ex}: + {len(df):,} rows")
    open(f"{dst}/.v13g_ready", "w").close()
EOF

# 3. training gold + DP1 + helpfulness
SNAP=data/gold/object_epoch_snapshots_fusion_v13g.parquet
SPLIT=data/gold/split_fusion_v13g.json
if [[ ! -f "${SNAP}" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --lc-dir data/lightcurves_v13g --silver-dir data/silver_v13g --truth data/truth/object_truth_v13g.parquet \
        --output "${SNAP}" --split-manifest "${SPLIT}" \
        --trust-metadata data/gold/split_fusion_v13c.json \
        --association-csv data/crossmatch/lsst_to_ztf.csv \
        --seq-train-ids models/seq_classifier_v11/fold_map.json \
        --lsst-live-locked "${UNION}" --lsst-all-negative-fallback \
        --dp1 --dp1-output data/gold/dp1_snapshots_fusion_v13g.parquet
fi
python3 - "${SPLIT}" data/gold/split_fusion_v13c.json "${H}" <<'EOF'
import json, sys
import pandas as pd
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("train_ids", "cal_ids", "test_ids"):
    x, y = set(map(str, a.get(k, []))), set(map(str, b.get(k, [])))
    print(f"split {k}: v13g {len(x):,} vs v13c {len(y):,}; only v13g {len(x - y)}, only v13c {len(y - x)}")
    assert not (y - x), f"a v13c {k} id moved or vanished"
tr = set(pd.read_parquet(f"{sys.argv[3]}/train_truth.parquet")["object_id"].astype(str))
te = set(pd.read_parquet(f"{sys.argv[3]}/test_truth.parquet")["object_id"].astype(str))
allv = set(map(str, a["train_ids"])) | set(map(str, a["cal_ids"])) | set(map(str, a["test_ids"]))
print(f"hard-negative TRAIN in train {len(tr & set(map(str, a['train_ids'])))}, cal {len(tr & set(map(str, a['cal_ids'])))}; "
      f"hard-negative TEST in any split: {len(te & allv)}")
assert not te & allv, "hard-negative TEST ids entered the training split"
EOF
HELP=data/gold/expert_helpfulness_fusion_v13g.parquet
if [[ ! -f "${HELP}" ]]; then
    python3 -u scripts/build_helpfulness_fusion.py --snapshots "${SNAP}" --output "${HELP}"
fi
python3 - "${SNAP}" <<'EOF'
import sys
import pandas as pd
g = pd.read_parquet(sys.argv[1], columns=["object_id", "survey", "target_class", "label_quality", "lc_fallback_all_negative"])
l = g[g["survey"].astype(str).str.upper() == "LSST"]
fb = l[l["lc_fallback_all_negative"] == 1]
print(f"LSST rows {len(l):,}; fallback rows {len(fb):,} on {fb['object_id'].nunique()} objects; "
      f"by class {fb.drop_duplicates('object_id')['target_class'].value_counts().to_dict()}")
EOF
echo "$(ts) training gold done"

# 4. serving golds (current code, fallback on), and the hard-negative TEST gold without it
EMPTY="${B}/gold/empty_split.json"
build() {  # $1 lc dir, $2 silver, $3 truth, $4 labels, $5 trust metadata, $6 output, $7 split, then extra flags
    local lc=$1 sv=$2 tr=$3 lb=$4 md=$5 out=$6 sp=$7; shift 7
    [[ -f "${out}" ]] && return 0
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" --lc-dir "${lc}" --silver-dir "${sv}" \
        --truth "${tr}" --bts "" --labels "${lb}" --trust-metadata "${md}" --no-lsst-weak \
        --output "${out}" --split-manifest "${sp}" "$@"
}
build "${B}/lightcurves" "${B}/silver_v13c" "${B}/truth.parquet" "${B}/gold/labels_empty.csv" "${EMPTY}" \
    "${B}/gold/bench_v13g.parquet" "${B}/gold/bench_v13g_split.json" --lsst-all-negative-fallback
build "${H}/lightcurves_test" "${H}/silver" "${H}/test_truth.parquet" "${B}/gold/labels_empty.csv" "${EMPTY}" \
    "${H}/gold/test_v13g.parquet" "${H}/gold/test_v13g_split.json" --lsst-all-negative-fallback
build "${H}/lightcurves_test" "${H}/silver" "${H}/test_truth.parquet" "${B}/gold/labels_empty.csv" "${EMPTY}" \
    "${H}/gold/test_v13c.parquet" "${H}/gold/test_v13c_split.json"
for sv in lsst ztf; do
    build "${T}/lightcurves_${sv}" "${T}/silver_${sv}_v13c" "${T}/truth/object_truth_${sv}.parquet" \
        "${T}/cohort/labels_${sv}_only.csv" "${T}/gold/empty_trust_metadata.json" \
        "${T}/gold/snapshots_${sv}_v13g.parquet" "${T}/gold/split_${sv}_v13g.json" --lsst-all-negative-fallback
done
build "${E}/lightcurves" "${E}/silver_v13cloc" "${E}/truth/object_truth.parquet" "${E}/gold/labels_empty.csv" \
    "${E}/gold/empty_trust_metadata.json" "${E}/gold/snapshots_v13gloc.parquet" "${E}/gold/split_v13gloc.json" \
    --lsst-all-negative-fallback
echo "$(ts) serving golds done"

# 5. the deployed v13f on the hard-negative TEST set (its own recipe, no fallback) and on the v13g golds
python3 -u scripts/eval_input_ablation.py --gold "${H}/gold/test_v13c.parquet" --truth "${H}/test_truth.parquet" \
    --manifest "${HN_MANIFEST}" --model v13f --variant full --out-dir "${H}/ablate_v13f_nofb"
python3 -u scripts/eval_input_ablation.py --gold "${H}/gold/test_v13g.parquet" --truth "${H}/test_truth.parquet" \
    --manifest "${HN_MANIFEST}" --model v13f --out-dir "${H}/ablate_v13f"
echo "$(ts) fusion v13g gold — DONE"
