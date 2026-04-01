#!/bin/bash -l
# submit_cpu_prep.sh — Submit only the CPU-safe SCC stages.
#
# Recommended workflow:
#   1. Submit CPU prep from a login node:
#        bash jobs/submit_cpu_prep.sh --limit 2000
#   2. After those jobs finish, enter your GPU session/node and resume:
#        bash -l jobs/run_gpu_resume.sh
#
# CPU chain:
#   download_training → backfill(array) → build_epochs
#
# With --local-classifiers (replaces broker API scores):
#   ... → build_epochs → train_alerce_lc → local_infer_cpu → rebuild_epochs
#
# With --skip-backfill --local-classifiers (fully offline, no API calls):
#   build_epochs → train_alerce_lc → local_infer_cpu → rebuild_epochs
#
# GPU resume picks up from the finished trust-aware CPU artifacts and runs:
#   local_infer_gpu → train_expert_trust → train_followup → score_all

set -euo pipefail

DEBASS_LIMIT=${DEBASS_LIMIT:-2000}
DEBASS_MAX_N_DET=${DEBASS_MAX_N_DET:-20}
DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
SKIP_DOWNLOAD=0
SKIP_BACKFILL=0
LOCAL_CLASSIFIERS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            shift
            DEBASS_LIMIT="$1"
            ;;
        --limit=*)
            DEBASS_LIMIT="${1#*=}"
            ;;
        --max-n-det)
            shift
            DEBASS_MAX_N_DET="$1"
            ;;
        --max-n-det=*)
            DEBASS_MAX_N_DET="${1#*=}"
            ;;
        --skip-download)
            SKIP_DOWNLOAD=1
            ;;
        --skip-backfill)
            SKIP_DOWNLOAD=1
            SKIP_BACKFILL=1
            ;;
        --local-classifiers)
            LOCAL_CLASSIFIERS=1
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

export DEBASS_ROOT DEBASS_LIMIT DEBASS_MAX_N_DET DEBASS_PYTHON_MODULE DEBASS_VENV

if [ -z "${DEBASS_ROOT:-}" ]; then
    echo "ERROR: DEBASS_ROOT is not set."
    echo "  export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon"
    exit 1
fi

if [ ! -f "$DEBASS_VENV/bin/activate" ]; then
    echo "ERROR: virtualenv not found at $DEBASS_VENV"
    exit 1
fi

mkdir -p "$DEBASS_ROOT/logs"

echo "======================================="
echo " DEBASS SCC CPU Prep Submission"
echo "======================================="
echo " DEBASS_ROOT:          $DEBASS_ROOT"
echo " DEBASS_LIMIT:         $DEBASS_LIMIT objects"
echo " DEBASS_MAX_N_DET:     $DEBASS_MAX_N_DET"
echo " DEBASS_PYTHON_MODULE: $DEBASS_PYTHON_MODULE"
echo " DEBASS_VENV:          $DEBASS_VENV"
echo "======================================="

cd "$DEBASS_ROOT"

echo " SKIP_DOWNLOAD:        $SKIP_DOWNLOAD"
echo " SKIP_BACKFILL:        $SKIP_BACKFILL"
echo " LOCAL_CLASSIFIERS:    $LOCAL_CLASSIFIERS"
echo "======================================="

# --- Step 1: Download ---
JID_DL=""
if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    JID_DL=$(qsub -terse -V jobs/download_training.sh)
    echo "[1] download_training   → job $JID_DL"
else
    echo "[1] download_training   → SKIPPED"
fi

# --- Step 2: Backfill (API calls) ---
JID_BF=""
if [[ "$SKIP_BACKFILL" -eq 0 ]]; then
    HOLD_ARG=""
    [[ -n "$JID_DL" ]] && HOLD_ARG="-hold_jid $JID_DL"
    JID_BF_RAW=$(qsub -terse -V $HOLD_ARG jobs/backfill.sh)
    JID_BF=$(echo "$JID_BF_RAW" | cut -d. -f1)
    echo "[2] backfill array      → job $JID_BF"
else
    echo "[2] backfill            → SKIPPED (--skip-backfill)"
fi

# --- Step 3: Build epochs ---
HOLD_ARG=""
[[ -n "$JID_BF" ]] && HOLD_ARG="-hold_jid $JID_BF"
[[ -z "$JID_BF" && -n "$JID_DL" ]] && HOLD_ARG="-hold_jid $JID_DL"
JID_EP=$(qsub -terse -V $HOLD_ARG jobs/build_epochs.sh)
echo "[3] build_epochs        → job $JID_EP"

# --- Step 4-5: Local classifiers (optional, CPU-only) ---
JID_LC=""
JID_LI=""
if [[ "$LOCAL_CLASSIFIERS" -eq 1 ]]; then
    JID_LC=$(qsub -terse -V -hold_jid "$JID_EP" jobs/train_alerce_lc.sh)
    echo "[4] train_alerce_lc     → job $JID_LC  (holds on $JID_EP)"

    JID_LI=$(qsub -terse -V -hold_jid "$JID_LC" jobs/local_infer_cpu.sh)
    echo "[5] local_infer_cpu     → job $JID_LI  (holds on $JID_LC)"

    # Rebuild epoch table with local classifier scores merged in
    JID_EP2=$(qsub -terse -V -hold_jid "$JID_LI" jobs/build_epochs.sh)
    echo "[6] rebuild_epochs      → job $JID_EP2 (holds on $JID_LI)"
    JID_EP="$JID_EP2"
fi

cat > "$DEBASS_ROOT/logs/cpu_prep_latest.json" <<EOF
{
  "download_job_id": "${JID_DL:-skipped}",
  "backfill_job_id": "${JID_BF:-skipped}",
  "build_job_id": "$JID_EP",
  "train_alerce_lc_job_id": "${JID_LC:-skipped}",
  "local_infer_cpu_job_id": "${JID_LI:-skipped}",
  "local_classifiers": $LOCAL_CLASSIFIERS,
  "limit": $DEBASS_LIMIT,
  "max_n_det": $DEBASS_MAX_N_DET
}
EOF

echo ""
echo "CPU prep submitted. Monitor with:"
echo "  qstat -u $USER"
echo "  tail -f $DEBASS_ROOT/logs/build_epochs.*.log"
echo ""
echo "After CPU prep finishes:"
echo "  $DEBASS_VENV/bin/python scripts/summarize_cpu_prep.py"
echo ""
echo "For GPU stage (SuperNNova/ParSNIP):"
echo "  cd $DEBASS_ROOT && bash -l jobs/run_gpu_resume.sh"
