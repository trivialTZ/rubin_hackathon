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
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_env}
DEBASS_GPU_EXPERTS=${DEBASS_GPU_EXPERTS:-parsnip}
DEBASS_SCORE_MODE=${DEBASS_SCORE_MODE:-trust}
DEBASS_QSUB_COMMON_ARGS=${DEBASS_QSUB_COMMON_ARGS:-}
DEBASS_QSUB_CPU_ARGS=${DEBASS_QSUB_CPU_ARGS:-}
DEBASS_QSUB_CPU_TRAIN_ARGS=${DEBASS_QSUB_CPU_TRAIN_ARGS:-}
DEBASS_QSUB_GPU_ARGS=${DEBASS_QSUB_GPU_ARGS:-}
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
export DEBASS_QSUB_COMMON_ARGS DEBASS_QSUB_CPU_ARGS DEBASS_QSUB_CPU_TRAIN_ARGS DEBASS_QSUB_GPU_ARGS

if [ -z "${DEBASS_ROOT:-}" ]; then
    echo "ERROR: DEBASS_ROOT is not set."
    echo "  export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon"
    exit 1
fi

mkdir -p "$DEBASS_ROOT/logs"

read -r -a QSUB_COMMON_ARR <<< "$DEBASS_QSUB_COMMON_ARGS"
read -r -a QSUB_CPU_ARR <<< "$DEBASS_QSUB_CPU_ARGS"
read -r -a QSUB_CPU_TRAIN_ARR <<< "$DEBASS_QSUB_CPU_TRAIN_ARGS"
read -r -a QSUB_GPU_ARR <<< "$DEBASS_QSUB_GPU_ARGS"

qsub_submit() {
    local scope="$1"
    shift

    local -a cmd=(qsub -terse -V "${QSUB_COMMON_ARR[@]}")
    case "$scope" in
        cpu)
            cmd+=("${QSUB_CPU_ARR[@]}")
            ;;
        cpu_train)
            cmd+=("${QSUB_CPU_ARR[@]}" "${QSUB_CPU_TRAIN_ARR[@]}")
            ;;
        gpu)
            cmd+=("${QSUB_GPU_ARR[@]}")
            ;;
        *)
            echo "ERROR: unknown qsub scope '$scope'" >&2
            return 1
            ;;
    esac
    cmd+=("$@")
    "${cmd[@]}"
}

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
echo " QSUB common:      ${DEBASS_QSUB_COMMON_ARGS:-<none>}"
echo " QSUB CPU:         ${DEBASS_QSUB_CPU_ARGS:-<none>}"
echo " QSUB CPU train:   ${DEBASS_QSUB_CPU_TRAIN_ARGS:-<none>}"
echo " QSUB GPU:         ${DEBASS_QSUB_GPU_ARGS:-<none>}"
echo "======================================="

cd "$DEBASS_ROOT"

JID_DL=$(qsub_submit cpu jobs/download_training.sh)
echo "[1] download_training   → job $JID_DL"

JID_BF=$(qsub_submit cpu -hold_jid "$JID_DL" jobs/backfill.sh)
echo "[2] backfill array      → job $JID_BF  (holds on $JID_DL)"

JID_EP=$(qsub_submit cpu -hold_jid "$JID_BF" jobs/build_epochs.sh)
echo "[3] build_epochs        → job $JID_EP  (holds on $JID_BF)"

if [ "$RUN_GPU" = true ]; then
    JID_GPU=$(qsub_submit gpu -hold_jid "$JID_EP" jobs/local_infer_gpu.sh)
    echo "[4] local_infer_gpu     → job $JID_GPU (holds on $JID_EP)"
    TRAIN_HOLD="$JID_GPU"
else
    echo "[4] GPU job skipped     (add --gpu to include local expert reruns)"
    TRAIN_HOLD="$JID_EP"
fi

if [ "$RUN_BASELINE" = true ]; then
    JID_TR=$(qsub_submit cpu_train -hold_jid "$TRAIN_HOLD" jobs/train_early.sh)
    echo "[5] train_early         → job $JID_TR  (holds on $TRAIN_HOLD)"
    JID_SC=$(qsub_submit cpu -hold_jid "$JID_TR" jobs/score_all.sh)
    echo "[6] score_all           → job $JID_SC  (holds on $JID_TR)"
else
    JID_TRUST=$(qsub_submit cpu_train -hold_jid "$TRAIN_HOLD" jobs/train_expert_trust.sh)
    echo "[5] train_expert_trust  → job $JID_TRUST (holds on $TRAIN_HOLD)"
    JID_FOLLOW=$(qsub_submit cpu_train -hold_jid "$JID_TRUST" jobs/train_followup.sh)
    echo "[6] train_followup      → job $JID_FOLLOW (holds on $JID_TRUST)"
    JID_SC=$(qsub_submit cpu -hold_jid "$JID_FOLLOW" jobs/score_all.sh)
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
