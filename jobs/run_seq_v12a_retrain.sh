#!/bin/bash -l
#$ -P pi-brout
#$ -N seq_v12a
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=12:00:00
#$ -o logs/seq_v12a.qsub.out
#$ -e logs/seq_v12a.qsub.err
#
# seq_v12a — GRU retrain with the live-LSST fixes (2026-07-07):
#   * marginalized coarse-label loss (weak tiers supervise SN-vs-non_sn only;
#     spec-only type supervision) — kills the lsst_weak_included type-head bug
#   * length-deconfounded balanced sampling + random-phase windows
#   * TNS-backlog spec LSST objects (--extra: 41 real + 260 GP-augmented)
#   * --allow-missing-survey-eval is REQUIRED and CONSCIOUS: the v11 split's
#     internal test frame is ZTF-only (the very blind spot that shipped
#     seq_v11). The REAL LSST gate is external: the frozen live benchmark
#     (data/gold/lsst_live_locked_test.json) + the odd-hash backlog holdout,
#     scored after this job by scripts/eval_lsst_live_bench.py.
#
# Variants via env: SEQ_V12_ENCODER (default supervised v11 encoder;
# set models/seq_encoder_ssl_v1 for the SSL variant), SEQ_V12_OUT.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
export PYTHONPATH=src
ts() { date -u +%H:%M:%S; }

ENCODER="${SEQ_V12_ENCODER:-models/seq_encoder_v11}"
OUT="${SEQ_V12_OUT:-models/seq_classifier_v12a}"
echo "$(ts) seq_v12a retrain START (encoder=${ENCODER}, out=${OUT}, NSLOTS=${NSLOTS:-?})"

python3 -u scripts/train_seq_classifier.py \
  --snapshots data/gold/object_epoch_snapshots_fusion_v11.parquet \
  --split data/gold/split_fusion_v11.json \
  --truth-table data/truth/object_truth_v11.parquet \
  --lc-dir data/lightcurves \
  --encoder "$ENCODER" \
  --seq-schema v11 --surveys both --oof-folds 5 --final-eval \
  --balanced-sampling --random-windows \
  --extra-lc-dir data/backlog_lsst/lightcurves \
  --extra-truth data/truth/backlog_train_truth.parquet \
  --extra-lc-dir data/augmented_lsst/lightcurves \
  --extra-truth data/augmented_lsst/truth_augmented.parquet \
  --allow-missing-survey-eval \
  --out "$OUT" --device auto

echo "$(ts) seq_v12a retrain DONE -> ${OUT}"
