#!/bin/bash
#$ -N debass_v9_pretrain
#$ -cwd -V
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -o logs/fusion_v9_pretrain.qsub.out
#$ -e logs/fusion_v9_pretrain.qsub.err
# metaDEBASS fusion_v9 — GPU stage: SSL pretraining of the sequence encoder
# (13,207-lightcurve corpus minus cal/test) + supervised fine-tune into the
# seq_v9 ternary classifier (ZTF labels + weak LSST candidates).
# No queue pin: the scheduler picks any free GPU (H200 at night).
# Requires the fusion_v8 split manifest + snapshot (built inside the v8 job).
# Submit via jobs/submit_fusion_v9_chain.sh (handles -hold_jid ordering).
set -euo pipefail

# ARCHIVED (v9c). scripts/train_seq_classifier.py now implements the v10
# protocol (4-way head, --oof-folds 5 default, grouped weak-label loss,
# inner-val early stopping): re-running this job would OVERWRITE
# models/seq_classifier_v9 — the artifact behind the archived
# reports_from_scc/fusion_v9c results AND the seq_v9 expert's fallback dir —
# with a v10-protocol model trained against the STALE v8 manifest (whose
# fold_map would disagree with the active split, voiding OOF routing).
# Use jobs/run_fusion_v10_pretrain.sh instead.
if [[ "${FUSION_V9_LEGACY_OK:-0}" != "1" ]]; then
    echo "REFUSING to run archived v9 pretrain job (would clobber models/seq_classifier_v9" \
         "with a v10-protocol model against the stale v8 manifest)." >&2
    echo "Use jobs/run_fusion_v10_pretrain.sh; set FUSION_V9_LEGACY_OK=1 to override." >&2
    exit 3
fi

cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
mkdir -p logs models

echo "$(ts) v9 pretrain — START"
python3 -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 1. SSL: heteroscedastic next-detection forecasting (corpus excludes cal/test)
python3 -u scripts/train_seq_encoder.py \
    --lc-dir data/lightcurves \
    --split data/gold/split_fusion_v8.json \
    --out models/seq_encoder_v9 \
    --device auto --epochs 40 --batch 512

# 2. Fine-tune into the seq_v9 classifier (train split only; cal = early stop
#    + temperature; weak LSST candidates from lsst_candidates.csv join train)
python3 -u scripts/train_seq_classifier.py \
    --snapshots data/gold/object_epoch_snapshots_fusion_v8.parquet \
    --split data/gold/split_fusion_v8.json \
    --lc-dir data/lightcurves \
    --encoder models/seq_encoder_v9 \
    --out models/seq_classifier_v9 \
    --device auto
echo "$(ts) v9 pretrain — DONE"
