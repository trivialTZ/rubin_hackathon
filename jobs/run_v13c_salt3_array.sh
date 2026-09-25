#!/bin/bash
#$ -N debass_v13c_salt3
#$ -cwd -V
#$ -l h_rt=03:00:00
#$ -l mem_per_core=4G
#$ -pe omp 1
#$ -j y
#$ -o logs/v13c_salt3.$TASK_ID.out
# metaDEBASS fusion v13c, step 1 of 3 (docs/fusion_v13_plan.md, "SALT3 on Rubin" and "v13c"): re-fit the SALT3 local
# expert with the v13c mapping (experts/local/salt3_fit.py ia_probability: per-point AIC likelihood ratio, Birge-rescaled,
# stable sigmoid; amplitudes >= 0; failed fits unavailable) for every object set whose gold carries it. The old silvers
# keep only p(Ia), so every epoch is re-fitted. All sets go through scripts/local_infer.py, the path the serving golds
# used (the ZTF training rows came from collect_epoch_history.py before), and the rows now carry alert_jd + survey.
# Task k runs shard k-1 of every set into data/v13c_salt3/<set>/s_<k-1>/ (resumable: finished shards have .done).
#   qsub -P pi-brout -t 1-48 jobs/run_v13c_salt3_array.sh
#   qsub -P pi-brout -hold_jid <this job id> jobs/run_fusion_v13c_gold.sh     (step 2)
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
N=${N_SHARDS:-48}
K=$((SGE_TASK_ID - 1))
[[ "${SGE_TASK_LAST:-${N}}" == "${N}" ]] || { echo "array size ${SGE_TASK_LAST} != N_SHARDS ${N}"; exit 2; }
OUT=data/v13c_salt3
declare -A LC=(
    [train]=data/lightcurves_v12
    [bench]=data/label_refresh_20260924/bench/lightcurves
    [tnsx_lsst]=data/tnsx_eval_20260924/lightcurves_lsst
    [tnsx_ztf]=data/tnsx_eval_20260924/lightcurves_ztf
    [dp2]=data/edp2_train/lightcurves
)
echo "$(ts) v13c SALT3 task ${SGE_TASK_ID}/${N} on $(hostname)"
for set in train bench tnsx_lsst tnsx_ztf dp2; do
    d="${OUT}/${set}/s_${K}"
    [[ -f "${d}/.done" ]] && { echo "$(ts) [skip] ${set} shard ${K}"; continue; }
    rm -rf "${d}" && mkdir -p "${d}"
    python3 - "${set}" "${d}/ids.csv" <<'EOF'
import glob, os, sys
import pandas as pd
s, out = sys.argv[1], sys.argv[2]
if s == "train":      # every object the v12-era training silver ran local experts on
    ids = set()
    for e in ("salt3_chi2", "alerce_lc"):
        p = f"data/silver_v12/local_expert_outputs/{e}/part-latest.parquet"
        ids |= set(pd.read_parquet(p, columns=["object_id"])["object_id"].astype(str))
elif s == "bench":    # frozen LSST benchmark + spec hold-out
    ids = set(pd.read_csv("data/label_refresh_20260924/bench/ids.csv", dtype=str)["object_id"])
elif s.startswith("tnsx_"):
    ids = set(open(f"data/tnsx_eval_20260924/cohort/ids_{s.split('_')[1]}.txt").read().split())
else:
    ids = {os.path.basename(p)[:-5] for p in glob.glob("data/edp2_train/lightcurves/*.json")}
pd.DataFrame({"object_id": sorted(ids)}).to_csv(out, index=False)
EOF
    echo "$(ts) ${set} shard ${K}/${N}: $(($(wc -l < "${d}/ids.csv") - 1)) objects in the set"
    python3 -u scripts/local_infer.py --expert salt3_chi2 --from-labels "${d}/ids.csv" --lc-dir "${LC[${set}]}" \
        --silver-dir "${d}" --max-n-det 20 --shard-id "${K}" --n-shards "${N}"
    touch "${d}/.done"
done
echo "$(ts) task ${SGE_TASK_ID} done"
