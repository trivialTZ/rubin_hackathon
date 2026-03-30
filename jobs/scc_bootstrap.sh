#!/bin/bash
# scc_bootstrap.sh — One-command BU SCC bootstrap + CPU prep submission.
#
# Run from the repo root or anywhere inside the repo:
#   bash -l jobs/scc_bootstrap.sh --limit 2000
#
# What it does:
#   1. Verifies the checkout contains the required repo files.
#   2. Loads the requested SCC Python module.
#   3. Creates/activates the virtualenv.
#   4. Installs env/requirements.txt.
#   5. Submits the CPU prep jobs unless --skip-submit is given.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEBASS_ROOT="${DEBASS_ROOT:-$REPO_ROOT}"
DEBASS_VENV="${DEBASS_VENV:-$DEBASS_ROOT/.venv}"
DEBASS_PYTHON_MODULE="${DEBASS_PYTHON_MODULE:-python3/3.10.12}"
DEBASS_LIMIT="${DEBASS_LIMIT:-2000}"
DEBASS_MAX_N_DET="${DEBASS_MAX_N_DET:-20}"
SUBMIT_CPU=true

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
        --python-module)
            shift
            DEBASS_PYTHON_MODULE="$1"
            ;;
        --python-module=*)
            DEBASS_PYTHON_MODULE="${1#*=}"
            ;;
        --venv)
            shift
            DEBASS_VENV="$1"
            ;;
        --venv=*)
            DEBASS_VENV="${1#*=}"
            ;;
        --skip-submit)
            SUBMIT_CPU=false
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

cd "$DEBASS_ROOT"

required_files=(
    "env/requirements.txt"
    "jobs/submit_cpu_prep.sh"
    "jobs/run_gpu_resume.sh"
    "scripts/download_alerce_training.py"
)

missing=()
for path in "${required_files[@]}"; do
    if [ ! -e "$path" ]; then
        missing+=("$path")
    fi
done

if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: this SCC checkout is incomplete. Missing required repo files:"
    for path in "${missing[@]}"; do
        echo "  $path"
    done
    echo ""
    echo "Sync or update the repository before running bootstrap."
    exit 1
fi

if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: the 'module' command is unavailable in this shell."
    echo "Run from a normal SCC login shell or use: bash -l jobs/scc_bootstrap.sh"
    exit 1
fi

if ! module -t avail "$DEBASS_PYTHON_MODULE" 2>&1 | grep -Fq "$DEBASS_PYTHON_MODULE"; then
    echo "ERROR: SCC Python module not found: $DEBASS_PYTHON_MODULE"
    echo "Available python3 modules:"
    module -t avail python3 2>&1 | sed 's/^/  /'
    exit 1
fi

export DEBASS_ROOT DEBASS_VENV DEBASS_PYTHON_MODULE DEBASS_LIMIT DEBASS_MAX_N_DET

echo "======================================="
echo " DEBASS SCC Bootstrap"
echo "======================================="
echo " DEBASS_ROOT:          $DEBASS_ROOT"
echo " DEBASS_VENV:          $DEBASS_VENV"
echo " DEBASS_PYTHON_MODULE: $DEBASS_PYTHON_MODULE"
echo " DEBASS_LIMIT:         $DEBASS_LIMIT"
echo " DEBASS_MAX_N_DET:     $DEBASS_MAX_N_DET"
echo " Submit CPU prep:      $SUBMIT_CPU"
echo "======================================="

module load "$DEBASS_PYTHON_MODULE"

CURRENT_MM="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

if [ -x "$DEBASS_VENV/bin/python" ]; then
    VENV_MM="$("$DEBASS_VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$VENV_MM" != "$CURRENT_MM" ]; then
        echo "Existing virtualenv uses Python $VENV_MM, but module $DEBASS_PYTHON_MODULE provides Python $CURRENT_MM."
        echo "Recreating $DEBASS_VENV to match the loaded SCC Python."
        rm -rf "$DEBASS_VENV"
    fi
fi

if [ ! -d "$DEBASS_VENV" ]; then
    python3 -m venv "$DEBASS_VENV"
fi

source "$DEBASS_VENV/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install -r env/requirements.txt

if [ "$SUBMIT_CPU" = true ]; then
    bash jobs/submit_cpu_prep.sh --limit "$DEBASS_LIMIT" --max-n-det "$DEBASS_MAX_N_DET"
else
    echo ""
    echo "Bootstrap finished. Submit CPU prep with:"
    echo "  bash jobs/submit_cpu_prep.sh --limit $DEBASS_LIMIT --max-n-det $DEBASS_MAX_N_DET"
fi
