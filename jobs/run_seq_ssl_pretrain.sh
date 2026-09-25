#!/bin/bash
#$ -N debass_seq_ssl
#$ -P pi-brout
#$ -cwd -V
#$ -l h_rt=12:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -o logs/seq_ssl_pretrain.qsub.out
#$ -e logs/seq_ssl_pretrain.qsub.err
# metaDEBASS WP3 — self-supervised pretraining on UNLABELED LSST alerts.
#
# The GRU sequence encoder has barely seen LSST cadence.  LSST publishes
# millions of unlabeled alert lightcurves; this stage (1) harvests a slice of
# them from Lasair and (2) pretrains the encoder with a NEXT-DETECTION objective
# (predict token k+1's delta-t + flux via Huber, band via cross-entropy).  The
# artifact ({encoder.pt, norm_stats.json, config.json}) is drop-in for
# scripts/train_seq_classifier.py --encoder (the same contract as
# train_seq_encoder.py's SSL output).
#
# gpu_c=8.0 keeps us off P100 (sm_60) nodes that crash torch>=2.x wheels
# (NEVER bare -l gpus=1 alone).
#
# Env knobs:
#   SEQ_SSL_TARGET   unlabeled objects to harvest        (default 20000)
#   SEQ_SSL_SCHEMA   v9|v11 tokenization                 (default v11)
#   SEQ_SSL_EPOCHS   pretraining epochs                  (default 30)
#   SEQ_SSL_FORCE    =1 to rebuild completed artifacts   (default 0)
#
# Idempotent: an existing encoder.pt / a harvest with enough lightcurves are
# reused unless SEQ_SSL_FORCE=1.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
mkdir -p logs models

TARGET="${SEQ_SSL_TARGET:-20000}"
SCHEMA="${SEQ_SSL_SCHEMA:-v11}"
EPOCHS="${SEQ_SSL_EPOCHS:-30}"
FORCE="${SEQ_SSL_FORCE:-0}"
PRETRAIN_DIR="data/pretrain_lsst"
LC_DIR="${PRETRAIN_DIR}/lightcurves"
OUT="models/seq_encoder_ssl_v1"

echo "$(ts) seq SSL pretrain — START (target=${TARGET}, schema=${SCHEMA}, epochs=${EPOCHS})"
python3 -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 1. Harvest unlabeled LSST lightcurves (resumable; benchmark ids excluded).
N_HAVE=$(ls "${LC_DIR}"/*.json 2>/dev/null | wc -l | tr -d ' ')
if [[ "${FORCE}" == "1" || "${N_HAVE}" -lt "$(( TARGET / 2 ))" ]]; then
    python3 -u scripts/harvest_unlabeled_lsst.py \
        --target "${TARGET}" \
        --min-ndet 3 \
        --out-dir "${PRETRAIN_DIR}" \
        --benchmark-manifest data/gold/lsst_live_locked_test.json
else
    echo "$(ts) [skip] harvest — ${N_HAVE} lightcurves already in ${LC_DIR}"
fi

# 2. Self-supervised next-detection pretraining.
if [[ "${FORCE}" == "1" || ! -f "${OUT}/encoder.pt" ]]; then
    python3 -u scripts/pretrain_seq_ssl.py \
        --lc-dir "${LC_DIR}" \
        --seq-schema "${SCHEMA}" \
        --surveys both --per-survey-norm \
        --epochs "${EPOCHS}" --batch-size 512 \
        --out "${OUT}" \
        --device auto
else
    echo "$(ts) [skip] SSL encoder exists: ${OUT}"
fi

echo "$(ts) seq SSL pretrain — DONE"
python3 - <<'EOF'
import json, pathlib
p = pathlib.Path("models/seq_encoder_ssl_v1/config.json")
if p.exists():
    meta = json.loads(p.read_text())
    print(f"== seq_encoder_ssl_v1 (schema={meta.get('seq_schema')}, "
          f"cont_dim={meta.get('cont_dim')}, corpus={meta.get('corpus_objects')}, "
          f"per_survey={meta.get('per_survey_stats_fitted')})")
    print(f"   val total: {meta.get('init_val_total')} -> {meta.get('best_val_total')}")
    print("   warm-start:  scripts/train_seq_classifier.py "
          f"--encoder models/seq_encoder_ssl_v1 --seq-schema {meta.get('seq_schema')} ...")
EOF
