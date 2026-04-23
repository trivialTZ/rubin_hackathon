#!/bin/bash
# DEBASS — honest alerce_lc retrain: train ONLY on train-fold objects to eliminate
# cohort-overlap leakage with downstream trust/followup heads.
#   qsub -N debass_alerce_honest -cwd -V -l h_rt=02:00:00 -l mem_per_core=8G -pe omp 4 \
#        -o logs/alerce_lc_honest.out -e logs/alerce_lc_honest.err jobs/run_alerce_lc_honest.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) export train-fold list"
python3 -u scripts/export_train_fold.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v2.parquet \
    --out-train data/labels_train_fold.csv \
    --out-cal   data/labels_cal_fold.csv \
    --out-test  data/labels_test_fold.csv

echo "$(ts) backup existing alerce_lc artifacts"
if [ -f artifacts/local_experts/alerce_lc/model.pkl ]; then
    cp artifacts/local_experts/alerce_lc/model.pkl artifacts/local_experts/alerce_lc/model_allobj.pkl
fi
if [ -f data/silver/local_expert_outputs/alerce_lc/part-latest.parquet ]; then
    cp data/silver/local_expert_outputs/alerce_lc/part-latest.parquet data/silver/local_expert_outputs/alerce_lc/part-allobj.parquet
fi

echo "$(ts) train alerce_lc on train-fold only"
python3 -u scripts/train_alerce_lc.py \
    --from-labels data/labels_train_fold.csv \
    --n-estimators 300 \
    --max-n-det 20 \
    --n-jobs 4

echo "$(ts) alerce_lc honest retrain done"
ls -la artifacts/local_experts/alerce_lc/
