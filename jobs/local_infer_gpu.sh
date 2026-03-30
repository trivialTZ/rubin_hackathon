#!/bin/bash -l
#$ -N debass_local_infer_gpu
#$ -l h_rt=12:00:00
#$ -l mem_per_core=16G
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=A40
#$ -l gpu_memory=24G
#$ -j y
#$ -o logs/local_infer_gpu.$JOB_ID.log
#$ -hold_jid debass_build_epochs
# Uncomment and set your project if on Med Campus:
##$ -P your_project_name

# DEBASS SCC GPU job: run SuperNNova and ParSNIP on truncated lightcurves.
# Populates snn_prob_ia and parsnip_prob_ia columns in the epoch table.
#
# Prerequisites:
#   - data/lightcurves/*.json populated by download_training.sh
#   - artifacts/local_experts/supernnova/ contains trained SNN weights
#   - artifacts/local_experts/parsnip/model.pt exists
#
# GPU options (check available with: qgpus -v)
#   A40:  40GB VRAM, good for SuperNNova/ParSNIP
#   A100: 80GB VRAM, fastest
#   H200: 141GB VRAM, most powerful
# Change gpu_type above to match what is available.

set -euo pipefail

module load python3/3.13.8
module load pytorch
source "$HOME/debass_env/bin/activate"

cd "$DEBASS_ROOT"

echo "=== DEBASS: GPU local inference ==="
echo "Node:  $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run SuperNNova — scores every (object, n_det) epoch from truncated lightcurves
python scripts/local_infer.py \
    --expert      supernnova \
    --from-labels data/labels.csv \
    --lc-dir      data/lightcurves \
    --silver-dir  data/silver \
    --max-n-det   "${DEBASS_MAX_N_DET:-20}"

# Run ParSNIP — appends to same local_expert_outputs.json
python scripts/local_infer.py \
    --expert      parsnip \
    --from-labels data/labels.csv \
    --lc-dir      data/lightcurves \
    --silver-dir  data/silver \
    --max-n-det   "${DEBASS_MAX_N_DET:-20}"

# Rebuild epoch table with new local expert scores filled in
python scripts/build_epoch_table_from_lc.py \
    --lc-dir     data/lightcurves \
    --silver-dir data/silver \
    --epochs-dir data/epochs \
    --labels     data/labels.csv \
    --max-n-det  "${DEBASS_MAX_N_DET:-20}"

echo "Done: $(date)"
