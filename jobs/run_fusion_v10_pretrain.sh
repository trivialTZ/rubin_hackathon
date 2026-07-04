#!/bin/bash
#$ -N debass_v10_pretrain
#$ -cwd -V
#$ -l h_rt=12:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -o logs/fusion_v10_pretrain.qsub.out
#$ -e logs/fusion_v10_pretrain.qsub.err
# metaDEBASS fusion_v10 — GPU stage: SSL pretraining + classifier fine-tunes,
# ALL against the FINAL v10 split manifest built by run_fusion_v10_build.sh
# (the chain-ordering fix: training never precedes the manifest it obeys).
# gpu_c=8.0 keeps us off P100 (sm_60) nodes that crash torch>=2.x wheels.
#
#   1. SSL encoder (models/seq_encoder_v10): ztf+lsst cached lightcurves
#      + DP1 parquet lightcurves, MINUS v10 cal/test objects AND their
#      LSST↔ZTF association counterparts; per-survey NormStats (pooled
#      fallback below 200 sequences/survey).
#   2. Deployed classifier (models/seq_classifier_v10): both surveys,
#      --oof-folds 5 → fold_{k}/ + fold_map.json so experts/local/seq_v9.py
#      scores every train object with the fold model that never saw it.
#      Fold models warm-start from the SSL encoder only — never from a
#      classifier that saw their fold — keeping OOF routing honest.
#   3. Staged ZTF→LSST arm (the explicit transfer recipe): ZTF-only
#      fine-tune → LSST-only adaptation at low LR warm-started from it
#      (models/seq_classifier_v10_ztf → ..._lsst_adapt).  Diagnostics-only:
#      it is NOT the deployed expert because a staged fold model would
#      inherit ZTF-fold photometry through the warm start, voiding OOF
#      honesty.  ONLY the deployed artifact reads the locked test
#      (--final-eval): the staged arms report per-survey CAL
#      (model-selection) AUCs — comparing variants on the locked test would
#      turn the pre-registered headline set into a model-selection set.
#      Any future promotion decision must be made on cal metrics and then
#      confirmed once on the locked test as a NEW pre-registered headline.
#      Skip with FUSION_V10_SKIP_STAGED=1.
#
# Idempotent: completed artifacts are reused (FUSION_V10_FORCE=1 rebuilds).
# Submit via jobs/submit_fusion_v10_chain.sh (handles -hold_jid ordering).
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
mkdir -p logs models

SNAP_PRE="data/gold/object_epoch_snapshots_fusion_v10_pre.parquet"
SPLIT="data/gold/split_fusion_v10.json"
FORCE="${FUSION_V10_FORCE:-0}"

echo "$(ts) v10 pretrain — START"
test -f "${SPLIT}" || { echo "missing ${SPLIT} (v10 build not finished?)"; exit 2; }
test -f "${SNAP_PRE}" || { echo "missing ${SNAP_PRE} (v10 build not finished?)"; exit 2; }
python3 -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 1. SSL: heteroscedastic next-detection forecasting (corpus excludes v10
#    cal/test + association counterparts; includes DP1 LSST cadence)
if [[ "${FORCE}" == "1" || ! -f models/seq_encoder_v10/encoder.pt ]]; then
    python3 -u scripts/train_seq_encoder.py \
        --lc-dir data/lightcurves \
        --split "${SPLIT}" \
        --association-csv data/crossmatch/lsst_to_ztf.csv \
        --dp1-lc-dir data/lightcurves/dp1 \
        --surveys both --per-survey-norm \
        --out models/seq_encoder_v10 \
        --device auto --epochs 40 --batch 512
else
    echo "$(ts) [skip] SSL encoder exists: models/seq_encoder_v10"
fi

# 2. Deployed classifier: both surveys, K-fold OOF artifacts (v10 manifest)
if [[ "${FORCE}" == "1" || ! -f models/seq_classifier_v10/fold_map.json ]]; then
    python3 -u scripts/train_seq_classifier.py \
        --snapshots "${SNAP_PRE}" \
        --split "${SPLIT}" \
        --truth-table data/truth/object_truth.parquet \
        --lc-dir data/lightcurves \
        --encoder models/seq_encoder_v10 \
        --surveys both --oof-folds 5 \
        --final-eval \
        --out models/seq_classifier_v10 \
        --device auto
else
    echo "$(ts) [skip] deployed classifier exists: models/seq_classifier_v10"
fi

# 3. Staged ZTF→LSST transfer arm (diagnostics-only; see header)
if [[ "${FUSION_V10_SKIP_STAGED:-0}" != "1" ]]; then
    if [[ "${FORCE}" == "1" || ! -f models/seq_classifier_v10_ztf/classifier.pt ]]; then
        python3 -u scripts/train_seq_classifier.py \
            --snapshots "${SNAP_PRE}" \
            --split "${SPLIT}" \
            --truth-table data/truth/object_truth.parquet \
            --lc-dir data/lightcurves \
            --encoder models/seq_encoder_v10 \
            --surveys ztf --oof-folds 0 \
            --out models/seq_classifier_v10_ztf \
            --device auto
    else
        echo "$(ts) [skip] ZTF stage exists: models/seq_classifier_v10_ztf"
    fi
    if [[ "${FORCE}" == "1" || ! -f models/seq_classifier_v10_lsst_adapt/classifier.pt ]]; then
        python3 -u scripts/train_seq_classifier.py \
            --snapshots "${SNAP_PRE}" \
            --split "${SPLIT}" \
            --truth-table data/truth/object_truth.parquet \
            --lc-dir data/lightcurves \
            --encoder models/seq_encoder_v10 \
            --init-from models/seq_classifier_v10_ztf \
            --surveys lsst --oof-folds 0 \
            --lr 1e-4 --encoder-lr 3e-5 --freeze-epochs 0 \
            --out models/seq_classifier_v10_lsst_adapt \
            --device auto
    else
        echo "$(ts) [skip] LSST-adapt stage exists: models/seq_classifier_v10_lsst_adapt"
    fi
else
    echo "$(ts) [skip] staged ZTF→LSST arm (FUSION_V10_SKIP_STAGED=1)"
fi

echo "$(ts) v10 pretrain — DONE"
python3 - <<'EOF'
import json, pathlib

def show(diag, banner):
    print(banner)
    for ndet in ("n_det=3", "n_det=5", "n_det=10"):
        entry = diag.get(ndet, {}) or {}
        for slc in ("all", "ztf", "lsst"):
            block = entry.get(slc, {}) or {}
            aucs = {k: round(v, 3) for k, v in block.items() if k.startswith("auc_")}
            if aucs:
                print(f"   {ndet:9s} {slc:4s} n={block.get('n_objects')}: {aucs}")

for name in ("seq_classifier_v10", "seq_classifier_v10_ztf", "seq_classifier_v10_lsst_adapt"):
    p = pathlib.Path("models") / name / "config.json"
    if not p.exists():
        continue
    meta = json.loads(p.read_text())
    print(f"== {name} (surveys={meta.get('surveys')}, oof_folds={meta.get('oof_folds')}, "
          f"final_eval={meta.get('final_eval')})")
    test_diag = meta.get("test_diagnostics_headline", {}) or {}
    if meta.get("final_eval") and "skipped" not in test_diag:
        show(test_diag, f"   HEADLINE (locked test, {test_diag.get('n_test_objects')} objects) "
                        f"— deployed artifact only:")
    else:
        # Staged/diagnostic arms never read the locked test: compare on cal.
        cal_diag = meta.get("cal_diagnostics_model_selection", {}) or {}
        show(cal_diag, "   CAL (MODEL-SELECTION ONLY — never a headline; "
                       "locked test untouched):")
EOF
