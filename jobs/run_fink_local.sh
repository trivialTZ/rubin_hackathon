#!/bin/bash
#$ -N fink_local_rubin
#$ -P pi-brout
#$ -V
#$ -l h_rt=04:00:00
#$ -l mem_per_core=4G
#$ -pe omp 4
#$ -o /projectnb/pi-brout/tztang/fink_local/logs/
#$ -e /projectnb/pi-brout/tztang/fink_local/logs/
# Fink-equivalent Rubin SNN + CATS, per epoch, for every Rubin cohort (scripts/fink_local_rubin.py; nothing here touches
# the repository venv). Array task i of N scores the objects with index % N == i-1 of every cohort.
#   qsub -P pi-brout -t 1-16 -tc 8 jobs/run_fink_local.sh                  # all cohorts (the array range goes on the
#   qsub -P pi-brout -t 1-4 -v COHORTS=bench jobs/run_fink_local.sh        # command line: a '#$ -t' header breaks qsub)
# Outputs: $FL/outputs/<cohort>/local_expert_outputs/{fink_rubin_snn,fink_rubin_cats}/part-<shard>.parquet
# Environment, venv and the pinned fink-science clone: $FL/env.sh (/projectnb/pi-brout/tztang/fink_local).
set -euo pipefail
export FL=${FL:-/projectnb/pi-brout/tztang/fink_local}
source "$FL/env.sh"
export OMP_NUM_THREADS=${NSLOTS:-4} MKL_NUM_THREADS=${NSLOTS:-4} TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=${NSLOTS:-4}
RUNNER=${FINK_RUNNER:-$FL/code/fink_local_rubin.py}
R=${DEBASS_DATA_ROOT:-/project/pi-brout/rubin_hackathon}/data
COHORTS=${COHORTS:-"lsst_v13g hardneg bench tnsx dp2"}
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
echo "$(ts) fink_local_rubin task ${SGE_TASK_ID:-1}/${SGE_TASK_LAST:-1} cohorts: ${COHORTS}"
for c in ${COHORTS}; do
    case "$c" in
        lsst_v13g) LC=$R/lightcurves_v13g; CAP=200; EXTRA="" ;;
        hardneg)   LC=$R/hardneg_20261004/lightcurves; CAP=50; EXTRA="" ;;
        bench)     LC=$R/label_refresh_20260924/bench/lightcurves; CAP=50; EXTRA="" ;;
        tnsx)      LC=$R/tnsx_eval_20260924/lightcurves_lsst; CAP=50; EXTRA="" ;;
        dp2)       LC=$R/edp2_train/lightcurves; CAP=1000000; EXTRA="--dp2" ;;  # DP2 photometry: restricted, stays on SCC
        *) echo "unknown cohort $c"; exit 2 ;;
    esac
    echo "$(ts) cohort $c  ($LC)"
    python "$RUNNER" score --cohort "$c" --lc-dir "$LC" --cap "$CAP" ${EXTRA} \
        --shard-id $(( ${SGE_TASK_ID:-1} - 1 )) --n-shards "${SGE_TASK_LAST:-1}" \
        2> >(grep -v -E "UserWarning|Expected:|Received:|warnings.warn" >&2)
done
echo "$(ts) done"
