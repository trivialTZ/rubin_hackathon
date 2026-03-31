#!/bin/bash -l
# submit_all.sh — Submit the full DEBASS early-epoch training pipeline to BU SCC.
#
# Usage:
#   cd $DEBASS_ROOT
#   bash jobs/submit_all.sh
#   bash jobs/submit_all.sh --limit 500      # smaller run
#   bash jobs/submit_all.sh --gpu            # include GPU local expert jobs
#   bash jobs/submit_all.sh --limit 2000 --gpu
#
# Prefer jobs/submit_cpu_prep.sh + jobs/run_gpu_resume.sh when the GPU stage
# will be launched manually from an interactive GPU allocation.
#
# Chain:
#   download_training → build_epochs → [local_infer_gpu] → train_early → score_all
#
# Requirements:
#   - DEBASS_ROOT must be set (e.g. export DEBASS_ROOT=/projectnb/myproject/rubin_hackathon)
#   - virtualenv at $HOME/debass_meta_env must exist (see README_scc.md)
#   - logs/ directory will be created automatically

set -euo pipefail

# ------------------------------------------------------------------ #
# Defaults                                                             #
# ------------------------------------------------------------------ #
DEBASS_LIMIT=${DEBASS_LIMIT:-2000}
DEBASS_MAX_N_DET=${DEBASS_MAX_N_DET:-20}
DEBASS_N_EST=${DEBASS_N_EST:-500}
DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
DEBASS_GPU_EXPERTS=${DEBASS_GPU_EXPERTS:-parsnip}
RUN_GPU=false

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

export DEBASS_ROOT DEBASS_LIMIT DEBASS_MAX_N_DET DEBASS_N_EST
export DEBASS_PYTHON_MODULE DEBASS_VENV DEBASS_GPU_EXPERTS

# Verify DEBASS_ROOT is set
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
echo "======================================="

cd "$DEBASS_ROOT"

# Step 1: Download training data (network-bound, 1 core)
JID_DL=$(qsub -terse -V jobs/download_training.sh)
echo "[1] download_training   → job $JID_DL"

# Step 2: Build epoch feature table (waits for download)
JID_EP=$(qsub -terse -V -hold_jid "$JID_DL" jobs/build_epochs.sh)
echo "[2] build_epochs        → job $JID_EP  (holds on $JID_DL)"

# Step 3 (optional): GPU local expert inference
if [ "$RUN_GPU" = true ]; then
    JID_GPU=$(qsub -terse -V -hold_jid "$JID_EP" jobs/local_infer_gpu.sh)
    echo "[3] local_infer_gpu     → job $JID_GPU (holds on $JID_EP)"
    TRAIN_HOLD="$JID_GPU"
else
    echo "[3] GPU job skipped     (add --gpu to include SuperNNova/ParSNIP)"
    TRAIN_HOLD="$JID_EP"
fi

# Step 4: Train early-epoch meta-classifier
JID_TR=$(qsub -terse -V -hold_jid "$TRAIN_HOLD" jobs/train_early.sh)
echo "[4] train_early         → job $JID_TR  (holds on $TRAIN_HOLD)"

# Step 5: Score all objects + write follow-up reports
JID_SC=$(qsub -terse -V -hold_jid "$JID_TR" jobs/score_all.sh)
echo "[5] score_all           → job $JID_SC  (holds on $JID_TR)"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  qstat -u $USER"
echo "  tail -f $DEBASS_ROOT/logs/download.*.log"
echo ""
echo "Final outputs:"
echo "  $DEBASS_ROOT/models/early_meta/early_meta.pkl"
echo "  $DEBASS_ROOT/reports/metrics/early_meta_metrics.json"
echo "  $DEBASS_ROOT/reports/scores/scores_ndet{3,5,10}.txt"
