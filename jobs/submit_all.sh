#!/bin/bash -l
# submit_all.sh — Submit the full DEBASS trust-aware SCC pipeline.
#
# Usage:
#   cd $DEBASS_ROOT
#   bash jobs/submit_all.sh
#   bash jobs/submit_all.sh --limit 500      # smaller run
#   bash jobs/submit_all.sh --gpu            # include GPU local expert jobs
#   bash jobs/submit_all.sh --limit 2000 --gpu
#   bash jobs/submit_all.sh --baseline       # benchmark-only baseline chain
#
# Prefer jobs/submit_cpu_prep.sh + jobs/run_gpu_resume.sh when the GPU stage
# will be launched manually from an interactive GPU allocation.
#
# Default trust-aware chain:
#   download_training → backfill(array) → build_epochs → [local_infer_gpu] → train_expert_trust → train_followup → score_all
#
# Baseline chain (explicit only):
#   download_training → backfill(array) → build_epochs → [local_infer_gpu] → train_early → score_all

set -euo pipefail

DEBASS_LIMIT=${DEBASS_LIMIT:-2000}
DEBASS_MAX_N_DET=${DEBASS_MAX_N_DET:-20}
DEBASS_N_EST=${DEBASS_N_EST:-500}
DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_meta_env}
DEBASS_GPU_EXPERTS=${DEBASS_GPU_EXPERTS:-parsnip}
DEBASS_SCORE_MODE=${DEBASS_SCORE_MODE:-trust}
RUN_GPU=false
RUN_BASELINE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            shift
            DEBASS_LIMIT="$1"
            ;;
        --limit=*)
            DEBASS_LIMIT="${1#*=}"
            ;;
        --gpu)
            RUN_GPU=true
            ;;
        --baseline)
            RUN_BASELINE=true
            ;;
        --max-n-det)
            shift
            DEBASS_MAX_N_DET="$1"
            ;;
        --max-n-det=*)
            DEBASS_MAX_N_DET="${1#*=}"
            ;;
        --n-est)
            shift
            DEBASS_N_EST="$1"
            ;;
        --n-est=*)
            DEBASS_N_EST="${1#*=}"
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$RUN_BASELINE" = true ]; then
    DEBASS_SCORE_MODE=baseline
fi

export DEBASS_ROOT DEBASS_LIMIT DEBASS_MAX_N_DET DEBASS_N_EST
export DEBASS_PYTHON_MODULE DEBASS_VENV DEBASS_GPU_EXPERTS DEBASS_SCORE_MODE

if [ -z "${DEBASS_ROOT:-}" ]; then
    echo "ERROR: DEBASS_ROOT is not set."
    echo "  export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon"
    exit 1
fi

mkdir -p "$DEBASS_ROOT/logs"

echo "======================================="
echo " DEBASS SCC Pipeline Submission"
echo "======================================="
echo " DEBASS_ROOT:      $DEBASS_ROOT"
echo " DEBASS_LIMIT:     $DEBASS_LIMIT objects"
echo " DEBASS_MAX_N_DET: $DEBASS_MAX_N_DET"
echo " DEBASS_N_EST:     $DEBASS_N_EST"
echo " Python module:    $DEBASS_PYTHON_MODULE"
echo " Virtualenv:       $DEBASS_VENV"
echo " GPU experts:      $RUN_GPU"
echo " Expert list:      $DEBASS_GPU_EXPERTS"
echo " Baseline mode:    $RUN_BASELINE"
echo " Score mode:       $DEBASS_SCORE_MODE"
echo "======================================="

cd "$DEBASS_ROOT"

JID_DL=$(qsub -terse -V jobs/download_training.sh)
echo "[1] download_training   → job $JID_DL"

JID_BF=$(qsub -terse -V -hold_jid "$JID_DL" jobs/backfill.sh)
echo "[2] backfill array      → job $JID_BF  (holds on $JID_DL)"

JID_EP=$(qsub -terse -V -hold_jid "$JID_BF" jobs/build_epochs.sh)
echo "[3] build_epochs        → job $JID_EP  (holds on $JID_BF)"

if [ "$RUN_GPU" = true ]; then
    JID_GPU=$(qsub -terse -V -hold_jid "$JID_EP" jobs/local_infer_gpu.sh)
    echo "[4] local_infer_gpu     → job $JID_GPU (holds on $JID_EP)"
    TRAIN_HOLD="$JID_GPU"
else
    echo "[4] GPU job skipped     (add --gpu to include local expert reruns)"
    TRAIN_HOLD="$JID_EP"
fi

if [ "$RUN_BASELINE" = true ]; then
    JID_TR=$(qsub -terse -V -hold_jid "$TRAIN_HOLD" jobs/train_early.sh)
    echo "[5] train_early         → job $JID_TR  (holds on $TRAIN_HOLD)"
    JID_SC=$(qsub -terse -V -hold_jid "$JID_TR" jobs/score_all.sh)
    echo "[6] score_all           → job $JID_SC  (holds on $JID_TR)"
else
    JID_TRUST=$(qsub -terse -V -hold_jid "$TRAIN_HOLD" jobs/train_expert_trust.sh)
    echo "[5] train_expert_trust  → job $JID_TRUST (holds on $TRAIN_HOLD)"
    JID_FOLLOW=$(qsub -terse -V -hold_jid "$JID_TRUST" jobs/train_followup.sh)
    echo "[6] train_followup      → job $JID_FOLLOW (holds on $JID_TRUST)"
    JID_SC=$(qsub -terse -V -hold_jid "$JID_FOLLOW" jobs/score_all.sh)
    echo "[7] score_all           → job $JID_SC  (holds on $JID_FOLLOW)"
fi

echo ""
echo "All jobs submitted. Monitor with:"
echo "  qstat -u $USER"
echo "  tail -f $DEBASS_ROOT/logs/download.*.log"
echo ""
if [ "$RUN_BASELINE" = true ]; then
    echo "Final outputs (baseline benchmark):"
    echo "  $DEBASS_ROOT/models/early_meta/early_meta.pkl"
    echo "  $DEBASS_ROOT/reports/metrics/early_meta_metrics.json"
    echo "  $DEBASS_ROOT/reports/scores/scores_ndet{3,5,10}.txt"
else
    echo "Final outputs (trust-aware):"
    echo "  $DEBASS_ROOT/models/trust/"
    echo "  $DEBASS_ROOT/models/followup/"
    echo "  $DEBASS_ROOT/reports/metrics/expert_trust_metrics.json"
    echo "  $DEBASS_ROOT/reports/metrics/followup_metrics.json"
    echo "  $DEBASS_ROOT/reports/scores/scores_ndet{3,5,10}.jsonl"
fi
