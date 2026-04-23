#!/bin/bash
# DEBASS Light-up — F6/F7 pipeline: analyze + figures + inventory + honesty pass
# Run from /project/pi-brout/rubin_hackathon after F3 trust heads are trained.
#   qsub -N debass_f6 -cwd -V -l h_rt=01:00:00 -l mem_per_core=8G -pe omp 2 \
#        -o logs/f6.qsub.out -e logs/f6.qsub.err jobs/run_f6_figures.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) ============================================================"
echo "$(ts) F6 — Analyze + regenerate all figures + emit inventory"
echo "$(ts) ============================================================"

echo "$(ts) Step 1/6: analyze_results (latency_curves.parquet)"
python3 -u scripts/analyze_results.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --snapshots-raw data/gold/object_epoch_snapshots.parquet \
    --helpfulness data/gold/expert_helpfulness.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics.json \
    --followup-metrics reports/metrics/followup_metrics.json \
    --output-dir reports

echo "$(ts) Step 2/6: make_paper_figures (4 core PDFs)"
python3 -u scripts/make_paper_figures.py \
    --latency reports/metrics/latency_curves.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics.json \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --trust-models-dir models/trust \
    --out-dir paper/debass_aas/figures

echo "$(ts) Step 3/6: plot_ablation (fig_ablation.pdf)"
python3 -u scripts/plot_ablation.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics.json \
    --out-dir reports/plots

echo "$(ts) Step 4/6: plot_per_class_confusion"
python3 -u scripts/plot_per_class_confusion.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --out-dir reports/plots

echo "$(ts) Step 5/6: emit_expert_inventory"
python3 -u scripts/emit_expert_inventory.py \
    --silver data/silver/broker_events.parquet \
    --trust-dir models/trust \
    --metrics-json reports/metrics/expert_trust_metrics.json \
    --output-json reports/expert_inventory.json \
    --output-md reports/expert_inventory.md

echo "$(ts) Step 6/6: honesty pass — CV + failure modes"
python3 -u scripts/cross_validate_ensemble.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --output-dir reports/metrics || echo "[warn] CV skipped: $?"
python3 -u scripts/analyze_failure_modes.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --truth data/truth/object_truth.parquet \
    --out-dir reports || echo "[warn] failure modes skipped: $?"

echo "$(ts) ============================================================"
echo "$(ts) F6/F7 DONE. Summary:"
echo "$(ts)   Trust heads:   $(ls -1 models/trust/ | grep -v 'metadata.json$' | wc -l)"
echo "$(ts)   Latency table: reports/metrics/latency_curves.parquet"
echo "$(ts)   Paper figures: paper/debass_aas/figures/fig_*.pdf"
echo "$(ts)   Inventory:     reports/expert_inventory.md"
echo "$(ts)   CV metrics:    reports/metrics/cv_summary.json"
echo "$(ts)   Failure modes: reports/failure_modes.csv"
echo "$(ts) ============================================================"
