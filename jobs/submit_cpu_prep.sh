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
# GPU resume picks up from the finished trust-aware CPU artifacts and runs:
#   local_infer_gpu → train_expert_trust → train_followup → score_all

set -euo pipefail

DEBASS_LIMIT=${DEBASS_LIMIT:-2000}
DEBASS_MAX_N_DET=${DEBASS_MAX_N_DET:-20}
DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_meta_env}

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

JID_DL=$(qsub -terse -V jobs/download_training.sh)
echo "[1] download_training   → job $JID_DL"

JID_BF=$(qsub -terse -V -hold_jid "$JID_DL" jobs/backfill.sh)
echo "[2] backfill array      → job $JID_BF  (holds on $JID_DL)"

JID_EP=$(qsub -terse -V -hold_jid "$JID_BF" jobs/build_epochs.sh)
echo "[3] build_epochs        → job $JID_EP  (holds on $JID_BF)"

cat > "$DEBASS_ROOT/logs/cpu_prep_latest.json" <<EOF
{
  "download_job_id": "$JID_DL",
  "backfill_job_id": "$JID_BF",
  "build_job_id": "$JID_EP",
  "download_log": "$DEBASS_ROOT/logs/download.$JID_DL.log",
  "backfill_log": "$DEBASS_ROOT/logs/backfill.$JID_BF.log",
  "build_log": "$DEBASS_ROOT/logs/build_epochs.$JID_EP.log",
  "limit": $DEBASS_LIMIT,
  "max_n_det": $DEBASS_MAX_N_DET
}
EOF

echo ""
echo "CPU prep submitted. Monitor with:"
echo "  qstat -u $USER"
echo "  tail -f $DEBASS_ROOT/logs/build_epochs.<job_id>.log"
echo "  $DEBASS_VENV/bin/python scripts/watch_cpu_prep.py"
echo ""
echo "After CPU prep finishes, summarize the downloaded CPU artifacts with:"
echo "  $DEBASS_VENV/bin/python scripts/summarize_cpu_prep.py --json-out reports/summary/cpu_prep_summary.json"
echo ""
echo "After build_epochs finishes and you are on a GPU node, resume with:"
echo "  cd $DEBASS_ROOT"
echo "  bash -l jobs/run_gpu_resume.sh"
