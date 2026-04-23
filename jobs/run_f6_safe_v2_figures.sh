#!/bin/bash
# DEBASS Light-up — F6-SAFE-v2: regenerate analyze + figures from SAFE-v2 snapshots (+SALT3).
#   qsub -N debass_f6safev2 -cwd -V -l h_rt=01:00:00 -l mem_per_core=8G -pe omp 2 \
#        -o logs/f6safev2.qsub.out -e logs/f6safev2.qsub.err jobs/run_f6_safe_v2_figures.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) F6-SAFE-v2 — no-leakage analyze + figures (6 clean experts w/ SALT3)"

python3 -u scripts/analyze_results.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v2.parquet \
    --snapshots-raw data/gold/object_epoch_snapshots_safe_v2.parquet \
    --helpfulness data/gold/expert_helpfulness_safe_v2.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics_safe_v2.json \
    --followup-metrics reports/metrics/followup_metrics_safe_v2.json \
    --output-dir reports_safe_v2

python3 -u scripts/make_paper_figures.py \
    --latency reports_safe_v2/metrics/latency_curves.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics_safe_v2.json \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v2.parquet \
    --trust-models-dir models/trust_safe_v2 \
    --out-dir paper/debass_aas/figures_safe_v2

python3 -u scripts/plot_ablation.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v2.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics_safe_v2.json \
    --out-dir reports_safe_v2/plots

python3 -u scripts/plot_per_class_confusion.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v2.parquet \
    --out-dir reports_safe_v2/plots

python3 -u scripts/cross_validate_ensemble.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v2.parquet \
    --output-dir reports_safe_v2/metrics || echo "[warn] CV skipped"
python3 -u scripts/analyze_failure_modes.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v2.parquet \
    --truth data/truth/object_truth.parquet \
    --out-dir reports_safe_v2 || echo "[warn] failure modes skipped"

echo "$(ts) F6-SAFE-v2 DONE"
echo "  trust heads (safe-v2): $(ls -1 models/trust_safe_v2/ | grep -v 'metadata.json$' | wc -l)"
echo "  followup AUC (safe-v2): $(python3 -c 'import json; print(json.load(open(\"reports/metrics/followup_metrics_safe_v2.json\"))[\"test_calibrated\"][\"roc_auc\"])')"
