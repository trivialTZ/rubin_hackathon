#!/bin/bash -l
# run_gpu_resume.sh — Resume the DEBASS pipeline from a prepared CPU state.
#
# Run this after jobs/submit_cpu_prep.sh has finished and you are already
# inside a GPU shell/session/node obtained via qrsh or SCC OnDemand.
#
# Default GPU expert list is "parsnip" because the current repository can load
# ParSNIP weights, while SuperNNova remains stub-only until its loader is
# implemented. Override if needed:
#   bash -l jobs/run_gpu_resume.sh --experts=parsnip,supernnova
#
# Baseline resume is available only with --baseline.

set -euo pipefail

DEBASS_GPU_EXPERTS=${DEBASS_GPU_EXPERTS:-parsnip}
DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_meta_meta_env}
RUN_BASELINE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experts)
            shift
            DEBASS_GPU_EXPERTS="$1"
            ;;
        --experts=*)
            DEBASS_GPU_EXPERTS="${1#*=}"
            ;;
        --baseline)
            RUN_BASELINE=true
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

export DEBASS_GPU_EXPERTS="${DEBASS_GPU_EXPERTS//,/ }"
export DEBASS_PYTHON_MODULE DEBASS_VENV

if [ -z "${DEBASS_ROOT:-}" ]; then
    echo "ERROR: DEBASS_ROOT is not set."
    echo "  export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon"
    exit 1
fi

if [ ! -f "$DEBASS_VENV/bin/activate" ]; then
    echo "ERROR: virtualenv not found at $DEBASS_VENV"
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module load "$DEBASS_PYTHON_MODULE"
    hash -r
fi

cd "$DEBASS_ROOT"

if [ ! -f data/labels.csv ]; then
    echo "ERROR: data/labels.csv not found."
    echo "Run CPU prep first: bash jobs/submit_cpu_prep.sh"
    exit 1
fi

if ! ls data/lightcurves/*.json >/dev/null 2>&1; then
    echo "ERROR: no cached lightcurves found in data/lightcurves/."
    echo "Run CPU prep first: bash jobs/submit_cpu_prep.sh"
    exit 1
fi

if [ ! -f data/silver/broker_events.parquet ]; then
    echo "ERROR: data/silver/broker_events.parquet not found."
    echo "Run CPU prep first: bash jobs/submit_cpu_prep.sh"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Run this script from a qrsh/OnDemand GPU session."
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
"$DEBASS_VENV/bin/python" - <<'PY'
import os
import sys

experts = os.environ.get("DEBASS_GPU_EXPERTS", "").split()
needs_torch = any(expert in {"parsnip", "supernnova"} for expert in experts)

try:
    import torch
except Exception as exc:
    if needs_torch:
        print(f"ERROR: torch import failed: {exc}")
        sys.exit(1)
    print(f"Torch preflight skipped: {exc}")
    sys.exit(0)

print(f"torch {torch.__version__}")
print(f"built with CUDA: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"device {idx}: {torch.cuda.get_device_name(idx)}")

if needs_torch and (not torch.cuda.is_available() or torch.cuda.device_count() < 1):
    print("ERROR: torch cannot see an assigned CUDA device in this session.")
    sys.exit(1)
PY

echo "======================================="
echo " DEBASS GPU Resume"
echo "======================================="
echo " DEBASS_ROOT:          $DEBASS_ROOT"
echo " DEBASS_GPU_EXPERTS:   $DEBASS_GPU_EXPERTS"
echo " DEBASS_MAX_N_DET:     ${DEBASS_MAX_N_DET:-20}"
echo " DEBASS_N_EST:         ${DEBASS_N_EST:-500}"
echo " DEBASS_PYTHON_MODULE: $DEBASS_PYTHON_MODULE"
echo " DEBASS_VENV:          $DEBASS_VENV"
echo " Baseline mode:        $RUN_BASELINE"
echo "======================================="

if [ "$RUN_BASELINE" = true ]; then
    export DEBASS_SCORE_MODE=baseline
    bash -l jobs/train_early.sh
    bash -l jobs/score_all.sh
else
    bash -l jobs/local_infer_gpu.sh
    if [ ! -f data/gold/expert_helpfulness.parquet ]; then
        echo "ERROR: trust-aware CPU artifacts missing (data/gold/expert_helpfulness.parquet)."
        echo "Run CPU prep first or use --baseline for benchmark mode."
        exit 1
    fi
    bash -l jobs/train_expert_trust.sh
    bash -l jobs/train_followup.sh
    bash -l jobs/score_all.sh
fi

echo ""
echo "GPU resume finished."
echo "Outputs:"
echo "  $DEBASS_ROOT/models/trust/"
echo "  $DEBASS_ROOT/models/followup/"
echo "  $DEBASS_ROOT/reports/metrics/"
echo "  $DEBASS_ROOT/reports/scores/"
