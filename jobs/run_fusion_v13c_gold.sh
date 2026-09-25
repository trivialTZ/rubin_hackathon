#!/bin/bash
#$ -N debass_v13c_gold
#$ -cwd -V
#$ -l h_rt=08:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v13c_gold.qsub.out
#$ -e logs/fusion_v13c_gold.qsub.err
# metaDEBASS fusion v13c, step 2 of 3: new silvers and every gold rebuilt.
#   * Silvers: a real copy of each v12-era silver (local_infer rewrites files in place, so no hard links) with
#     local_expert_outputs/salt3_chi2 replaced by the step-1 shards (jobs/run_v13c_salt3_array.sh); the old SALT3 rows
#     are kept next to it as _salt3_chi2_v12/ (outside the builder's */*.parquet glob).
#   * Golds: ingest/gold.py now times local-expert rows with a NaN alert_jd from alert_mjd. The v12w gold took every
#     LSST salt3_chi2 / alerce_lc row as untimed, so the as-of join averaged all epochs of the object, later ones
#     included (a look-ahead, and a train/serve skew: serving golds were timed).
#     training gold + DP1 + helpfulness *_fusion_v13c (the v12w recipe: v12 labels + ALeRCE-stamp weak LSST rows),
#     frozen LSST benchmark bench/gold/bench_v13c.parquet, explorer cohort tnsx gold/snapshots_{lsst,ztf}_v13c.parquet,
#     DP2 edp2_train/gold/snapshots_v13cloc.parquet.
#   qsub -P pi-brout -hold_jid <step-1 array id> jobs/run_fusion_v13c_gold.sh
#   qsub -P pi-brout -hold_jid <this job id> -v FUSION_V13_ARM=v13c jobs/run_fusion_v13.sh     (step 3)
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
N=${N_SHARDS:-48}
S3=data/v13c_salt3
B=data/label_refresh_20260924/bench
T=data/tnsx_eval_20260924
E=data/edp2_train
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"
mkdir -p logs data/gold
echo "$(ts) fusion v13c gold — START (NSLOTS=${NSLOTS})"

# 1. silvers
new_silver() {  # $1 = SALT3 set, $2 = source silver, $3 = new silver
    local set=$1 src=$2 dst=$3 k f
    for ((k = 0; k < N; k++)); do
        [[ -f "${S3}/${set}/s_${k}/.done" ]] || { echo "SALT3 shard ${set}/s_${k} not done"; exit 3; }
    done
    if [[ ! -f "${dst}/.v13c_ready" ]]; then
        if [[ -e "${dst}" ]]; then mv "${dst}" "${dst}.old_$(date +%Y%m%d%H%M%S)"; fi
        cp -a "${src}" "${dst}"
        mv "${dst}/local_expert_outputs/salt3_chi2" "${dst}/_salt3_chi2_v12"
        mkdir -p "${dst}/local_expert_outputs/salt3_chi2"
        for ((k = 0; k < N; k++)); do
            f="${S3}/${set}/s_${k}/local_expert_outputs/salt3_chi2/part-latest.parquet"
            if [[ -f "${f}" ]]; then cp "${f}" "${dst}/local_expert_outputs/salt3_chi2/part-shard${k}.parquet"; fi
        done
        touch "${dst}/.v13c_ready"
    fi
    python3 - "${dst}" <<'EOF'
import glob, json, sys
import numpy as np, pandas as pd
d = sys.argv[1]
new = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{d}/local_expert_outputs/salt3_chi2/*.parquet")], ignore_index=True)
old = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{d}/_salt3_chi2_v12/*.parquet")], ignore_index=True)
def summ(df):
    p = df["class_probabilities"].map(lambda s: (json.loads(s) if isinstance(s, str) else (s or {})).get("Ia"))
    p = pd.to_numeric(p, errors="coerce")
    ok = p.notna()
    return (f"{len(df):,} rows, {df['object_id'].astype(str).nunique():,} objects, p(Ia) on {ok.mean():.1%}, "
            f"degenerate {((p[ok] - 0.5).abs() > 0.49).mean():.1%}")
print(f"{d}: SALT3 v12 {summ(old)}")
print(f"{d}: SALT3 v13c {summ(new)}; alert_jd NaN {int(new['alert_jd'].isna().sum())}")
EOF
}
new_silver train data/silver_v12 data/silver_v13c
new_silver bench "${B}/silver" "${B}/silver_v13c"
new_silver tnsx_lsst "${T}/silver_lsst" "${T}/silver_lsst_v13c"
new_silver tnsx_ztf "${T}/silver_ztf" "${T}/silver_ztf_v13c"
new_silver dp2 "${E}/silver_v12loc" "${E}/silver_v13cloc"
echo "$(ts) silvers ready"

# 2. serving golds, with the same builder flags as jobs/run_fusion_v12_bench.sh, run_tnsx_v12_score.sh and
#    run_edp2_v12_score.sh (current code)
EMPTY="${B}/gold/empty_split.json"
test -f "${EMPTY}" && test -f "${B}/gold/labels_empty.csv" || { echo "missing bench empty split/labels"; exit 2; }
if [[ ! -f "${B}/gold/bench_v13c.parquet" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --lc-dir "${B}/lightcurves" --silver-dir "${B}/silver_v13c" --truth "${B}/truth.parquet" \
        --bts "" --labels "${B}/gold/labels_empty.csv" --trust-metadata "${EMPTY}" --no-lsst-weak \
        --output "${B}/gold/bench_v13c.parquet" --split-manifest "${B}/gold/bench_v13c_split.json"
fi
for sv in lsst ztf; do
    if [[ ! -f "${T}/gold/snapshots_${sv}_v13c.parquet" ]]; then
        python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
            --lc-dir "${T}/lightcurves_${sv}" --silver-dir "${T}/silver_${sv}_v13c" \
            --truth "${T}/truth/object_truth_${sv}.parquet" --bts "" \
            --labels "${T}/cohort/labels_${sv}_only.csv" \
            --trust-metadata "${T}/gold/empty_trust_metadata.json" --no-lsst-weak \
            --output "${T}/gold/snapshots_${sv}_v13c.parquet" --split-manifest "${T}/gold/split_${sv}_v13c.json"
    fi
done
if [[ ! -f "${E}/gold/snapshots_v13cloc.parquet" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --lc-dir "${E}/lightcurves" --silver-dir "${E}/silver_v13cloc" --truth "${E}/truth/object_truth.parquet" --bts "" \
        --labels "${E}/gold/labels_empty.csv" --trust-metadata "${E}/gold/empty_trust_metadata.json" --no-lsst-weak \
        --output "${E}/gold/snapshots_v13cloc.parquet" --split-manifest "${E}/gold/split_v13cloc.json"
fi
echo "$(ts) serving golds done"

# 3. training gold + DP1 (the v12w recipe of jobs/run_fusion_v12.sh step 4, without --no-lsst-weak)
SNAP=data/gold/object_epoch_snapshots_fusion_v13c.parquet
SPLIT=data/gold/split_fusion_v13c.json
if [[ ! -f "${SNAP}" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --lc-dir data/lightcurves_v12 --silver-dir data/silver_v13c --truth data/truth/object_truth_v12_merged.parquet \
        --output "${SNAP}" --split-manifest "${SPLIT}" \
        --association-csv data/crossmatch/lsst_to_ztf.csv \
        --seq-train-ids models/seq_classifier_v11/fold_map.json \
        --lsst-live-locked data/gold/lsst_live_locked_test.json \
        --dp1 --dp1-output data/gold/dp1_snapshots_fusion_v13c.parquet
fi
python3 - "${SPLIT}" data/gold/split_fusion_v12w.json <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("train_ids", "cal_ids", "test_ids"):
    x, y = set(map(str, a.get(k, []))), set(map(str, b.get(k, [])))
    print(f"split {k}: v13c {len(x):,} vs v12w {len(y):,}; only v13c {len(x - y)}, only v12w {len(y - x)}")
EOF

# 4. helpfulness
HELP=data/gold/expert_helpfulness_fusion_v13c.parquet
if [[ ! -f "${HELP}" ]]; then
    python3 -u scripts/build_helpfulness_fusion.py --snapshots "${SNAP}" --output "${HELP}"
fi

# 5. what changed in the gold: SALT3 degeneracy and alerce_lc on LSST rows, v12w vs v13c
python3 - data/gold/object_epoch_snapshots_fusion_v12w.parquet "${SNAP}" <<'EOF'
import sys
import pandas as pd
for path in sys.argv[1:3]:
    cols = ["object_id", "n_det", "survey", "avail__salt3_chi2", "proj__salt3_chi2__p_snia",
            "avail__alerce_lc", "proj__alerce_lc__p_snia"]
    g = pd.read_parquet(path, columns=cols)
    for sv, q in g.groupby("survey"):
        s = q[q["avail__salt3_chi2"] == 1]["proj__salt3_chi2__p_snia"]
        a = q[q["avail__alerce_lc"] == 1]["proj__alerce_lc__p_snia"]
        print(f"{path.split('/')[-1]} {sv}: rows {len(q):,}; salt3 avail {q['avail__salt3_chi2'].mean():.1%}, "
              f"degenerate {((s - 0.5).abs() > 0.49).mean():.1%}; alerce_lc avail {q['avail__alerce_lc'].mean():.1%}, "
              f"p_snia sd {a.std():.3f}")
EOF
echo "$(ts) fusion v13c gold — DONE"
