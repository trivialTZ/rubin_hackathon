#!/bin/bash
#$ -N debass_v11_pretrain
#$ -cwd -V
#$ -l h_rt=12:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -o logs/fusion_v11_pretrain.qsub.out
#$ -e logs/fusion_v11_pretrain.qsub.err
# metaDEBASS fusion_v11 — GPU stage: the seq_v11 sequence arm (P5, all gated),
# trained against the FINAL v11 split manifest from run_fusion_v11_build.sh.
# gpu_c=8.0 keeps us off P100 (sm_60) nodes that crash torch>=2.x wheels
# (NEVER bare -l gpus=1 alone).
#
# v11 sequence schema (B8): negatives as tokens (is_negative channel + signed
# flux). --seq-schema v11 is written into the artifact meta so seq_v9.py routes
# per-artifact; v9/v10 artifacts (cont_dim=9) stay bit-identical. The expert
# reads the v11 artifact via env DEBASS_SEQ_V11_MODEL — we NEVER point
# DEBASS_SEQ_V9_MODEL at a v11 artifact.
#
#   0. (optional) ELAsTiCC2 SN-class pretraining corpus (~3-4 GB selective
#      fetch; FUSION_V11_ELASTICC2=1 to enable, else skipped — sim-to-real gap
#      means it is pretraining only, never eval).
#   1. SSL encoder (models/seq_encoder_v11): ztf+lsst + DP1 cadence, MINUS v11
#      cal/test + LSST↔ZTF association counterparts; per-survey NormStats.
#   2. Deployed classifier (models/seq_classifier_v11): both surveys, --oof-folds
#      5 → fold_{k}/ + fold_map.json so the expert scores every train object with
#      the fold model that never saw it (OOF honesty).
#
# Idempotent: completed artifacts are reused (FUSION_V11_FORCE=1 rebuilds).
# Submit via jobs/submit_fusion_v11_chain.sh (handles -hold_jid ordering).
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
mkdir -p logs models

SNAP="data/gold/object_epoch_snapshots_fusion_v11.parquet"
SPLIT="data/gold/split_fusion_v11.json"
TRUTH_V11="data/truth/object_truth_v11.parquet"
FORCE="${FUSION_V11_FORCE:-0}"

echo "$(ts) v11 pretrain — START"
test -f "${SPLIT}" || { echo "missing ${SPLIT} (v11 build not finished?)"; exit 2; }
test -f "${SNAP}"  || { echo "missing ${SNAP} (v11 build not finished?)"; exit 2; }
python3 -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 0. optional ELAsTiCC2 SN-class corpus (selective per-class fetch; resume-aware)
if [[ "${FUSION_V11_ELASTICC2:-0}" == "1" ]]; then
    if [[ "${FORCE}" == "1" || ! -d data/elasticc2 ]]; then
        python3 -u scripts/fetch_elasticc2.py --out data/elasticc2 || {
            echo "$(ts) WARN: ELAsTiCC2 fetch failed — proceeding without pretrain corpus" >&2; }
    else
        echo "$(ts) [skip] ELAsTiCC2 corpus exists: data/elasticc2"
    fi
else
    echo "$(ts) [skip] ELAsTiCC2 pretrain corpus (FUSION_V11_ELASTICC2!=1)"
fi

# 1. SSL encoder (v11 schema; corpus excludes v11 cal/test + assoc counterparts)
if [[ "${FORCE}" == "1" || ! -f models/seq_encoder_v11/encoder.pt ]]; then
    python3 -u scripts/train_seq_encoder.py \
        --lc-dir data/lightcurves \
        --split "${SPLIT}" \
        --association-csv data/crossmatch/lsst_to_ztf.csv \
        --dp1-lc-dir data/lightcurves/dp1 \
        --surveys both --per-survey-norm \
        --seq-schema v11 \
        --out models/seq_encoder_v11 \
        --device auto --epochs 40 --batch 512
else
    echo "$(ts) [skip] SSL encoder exists: models/seq_encoder_v11"
fi

# 2. Deployed classifier: both surveys, K-fold OOF artifacts (v11 manifest+schema)
if [[ "${FORCE}" == "1" || ! -f models/seq_classifier_v11/fold_map.json ]]; then
    python3 -u scripts/train_seq_classifier.py \
        --snapshots "${SNAP}" \
        --split "${SPLIT}" \
        --truth-table "${TRUTH_V11}" \
        --lc-dir data/lightcurves \
        --encoder models/seq_encoder_v11 \
        --seq-schema v11 \
        --surveys both --oof-folds 5 \
        --final-eval \
        --out models/seq_classifier_v11 \
        --device auto
else
    echo "$(ts) [skip] deployed classifier exists: models/seq_classifier_v11"
fi

echo "$(ts) v11 pretrain — DONE"
python3 - <<'EOF'
import json, pathlib
p = pathlib.Path("models/seq_classifier_v11/config.json")
if p.exists():
    meta = json.loads(p.read_text())
    print(f"== seq_classifier_v11 (surveys={meta.get('surveys')}, "
          f"oof_folds={meta.get('oof_folds')}, seq_schema={meta.get('seq_schema')}, "
          f"final_eval={meta.get('final_eval')})")
    td = meta.get("test_diagnostics_headline", {}) or {}
    if meta.get("final_eval") and "skipped" not in td:
        for ndet in ("n_det=3", "n_det=5", "n_det=10"):
            entry = td.get(ndet, {}) or {}
            for slc in ("all", "ztf", "lsst"):
                block = entry.get(slc, {}) or {}
                aucs = {k: round(v, 3) for k, v in block.items() if k.startswith("auc_")}
                if aucs:
                    print(f"   {ndet:9s} {slc:4s} n={block.get('n_objects')}: {aucs}")
EOF
