#!/bin/bash
# DEBASS Light-up — F6-SAFE-v3: figures from 7-clean-expert snapshots.
#   qsub -N debass_f6safev4 -cwd -V -l h_rt=01:00:00 -l mem_per_core=8G -pe omp 2 \
#        -o logs/f6safev4.qsub.out -e logs/f6safev4.qsub.err jobs/run_f6_safe_v4_figures.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
echo "$(ts) F6-SAFE-v3 — analyze + figures (7 clean experts)"

python3 -u scripts/analyze_results.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v4.parquet \
    --snapshots-raw data/gold/object_epoch_snapshots_safe_v4.parquet \
    --helpfulness data/gold/expert_helpfulness_safe_v4.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics_safe_v4.json \
    --followup-metrics reports/metrics/followup_metrics_safe_v4.json \
    --output-dir reports_safe_v4

python3 -u scripts/make_paper_figures.py \
    --latency reports_safe_v4/metrics/latency_curves.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics_safe_v4.json \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v4.parquet \
    --trust-models-dir models/trust_safe_v4 \
    --out-dir paper/debass_aas/figures_safe_v4

python3 -u scripts/plot_ablation.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v4.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics_safe_v4.json \
    --out-dir reports_safe_v4/plots
python3 -u scripts/plot_per_class_confusion.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v4.parquet \
    --out-dir reports_safe_v4/plots
python3 -u scripts/cross_validate_ensemble.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v4.parquet \
    --output-dir reports_safe_v4/metrics || true
python3 -u scripts/analyze_failure_modes.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe_v4.parquet \
    --truth data/truth/object_truth.parquet \
    --out-dir reports_safe_v4 || true
echo "$(ts) F6-SAFE-v3 DONE"
