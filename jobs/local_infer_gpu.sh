#!/bin/bash -l
#$ -N debass_local_infer_gpu
#$ -l h_rt=12:00:00
#$ -l mem_per_core=16G
#$ -pe omp 8
#$ -l gpus=1
#$ -q l40s
#$ -j y
#$ -o logs/local_infer_gpu.$JOB_ID.log
#$ -hold_jid debass_build_epochs
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# DEBASS SCC GPU job: run configured local experts on truncated lightcurves.
# Populates snn_prob_ia / parsnip_prob_ia columns in the epoch table when the
# corresponding expert produces live model outputs.
#
# Prerequisites:
#   - data/lightcurves/*.json populated by download_training.sh
#   - artifacts/local_experts/<expert>/ contains any required model artifacts
#     (for ParSNIP: model.pt + classifier.pkl)
#   - DEBASS_GPU_EXPERTS controls which experts are run (default: parsnip)
#   - requested experts must be live-model ready; stub mode is treated as an error
#
# GPU options (check available with: qgpus -v)
#   A40:  40GB VRAM, good for SuperNNova/ParSNIP
#   A100: 80GB VRAM, fastest
#   H200: 141GB VRAM, most powerful
# Change gpu_type above to match what is available.

set -euo pipefail

DEBASS_PYTHON_MODULE=${DEBASS_PYTHON_MODULE:-python3/3.10.12}
DEBASS_VENV=${DEBASS_VENV:-$HOME/debass_env}
DEBASS_GPU_EXPERTS=${DEBASS_GPU_EXPERTS:-parsnip}
: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

module load "$DEBASS_PYTHON_MODULE"
if [ ! -f "$DEBASS_VENV/bin/activate" ]; then
    echo "ERROR: virtualenv not found at $DEBASS_VENV"
    exit 1
fi
source "$DEBASS_VENV/bin/activate"

cd "$DEBASS_ROOT"

echo "=== DEBASS: GPU local inference ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "Experts: $DEBASS_GPU_EXPERTS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for expert in $DEBASS_GPU_EXPERTS; do
    python scripts/local_infer.py \
        --expert      "$expert" \
        --from-labels data/labels.csv \
        --lc-dir      data/lightcurves \
        --silver-dir  data/silver \
        --max-n-det   "${DEBASS_MAX_N_DET:-20}" \
        --require-live-model
done

# Rebuild epoch table with new local expert scores filled in
python scripts/build_epoch_table_from_lc.py \
    --lc-dir     data/lightcurves \
    --silver-dir data/silver \
    --epochs-dir data/epochs \
    --labels     data/labels.csv \
    --max-n-det  "${DEBASS_MAX_N_DET:-20}"

echo "Done: $(date)"
