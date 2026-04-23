#!/bin/bash
# DEBASS Light-up — F6-SAFE: regenerate analyze + figures + ablation + CV from SAFE snapshots
# Run from /project/pi-brout/rubin_hackathon AFTER F3-SAFE finishes.
#   qsub -N debass_f6safe -cwd -V -l h_rt=01:00:00 -l mem_per_core=8G -pe omp 2 \
#        -o logs/f6safe.qsub.out -e logs/f6safe.qsub.err jobs/run_f6_safe_figures.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) ============================================================"
echo "$(ts) F6-SAFE — no-leakage analyze + figures"
echo "$(ts) ============================================================"

echo "$(ts) Step 1/5: analyze_results (SAFE)"
python3 -u scripts/analyze_results.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe.parquet \
    --snapshots-raw data/gold/object_epoch_snapshots_safe.parquet \
    --helpfulness data/gold/expert_helpfulness_safe.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics_safe.json \
    --followup-metrics reports/metrics/followup_metrics_safe.json \
    --output-dir reports_safe

echo "$(ts) Step 2/5: make_paper_figures (SAFE)"
python3 -u scripts/make_paper_figures.py \
    --latency reports_safe/metrics/latency_curves.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics_safe.json \
    --snapshots data/gold/object_epoch_snapshots_trust_safe.parquet \
    --trust-models-dir models/trust_safe \
    --out-dir paper/debass_aas/figures_safe

echo "$(ts) Step 3/5: plot_ablation (SAFE)"
python3 -u scripts/plot_ablation.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics_safe.json \
    --out-dir reports_safe/plots

echo "$(ts) Step 4/5: plot_per_class_confusion (SAFE)"
python3 -u scripts/plot_per_class_confusion.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe.parquet \
    --out-dir reports_safe/plots

echo "$(ts) Step 5/5: honesty — CV + failure modes (SAFE)"
python3 -u scripts/cross_validate_ensemble.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe.parquet \
    --output-dir reports_safe/metrics || echo "[warn] CV skipped: $?"
python3 -u scripts/analyze_failure_modes.py \
    --snapshots data/gold/object_epoch_snapshots_trust_safe.parquet \
    --truth data/truth/object_truth.parquet \
    --out-dir reports_safe || echo "[warn] failure modes skipped: $?"

echo "$(ts) ============================================================"
echo "$(ts) F6-SAFE DONE. Side-by-side:"
echo "$(ts)   UNSAFE trust heads:  $(ls -1 models/trust/ 2>/dev/null | grep -v 'metadata.json$' | wc -l)"
echo "$(ts)   SAFE   trust heads:  $(ls -1 models/trust_safe/ 2>/dev/null | grep -v 'metadata.json$' | wc -l)"
echo "$(ts)   UNSAFE followup AUC: $(python3 -c 'import json; print(json.load(open(\"reports/metrics/followup_metrics.json\"))[\"test\"][\"roc_auc\"])' 2>/dev/null || echo '?')"
echo "$(ts)   SAFE   followup AUC: $(python3 -c 'import json; print(json.load(open(\"reports/metrics/followup_metrics_safe.json\"))[\"test\"][\"roc_auc\"])' 2>/dev/null || echo '?')"
echo "$(ts)   UNSAFE figures: paper/debass_aas/figures/"
echo "$(ts)   SAFE   figures: paper/debass_aas/figures_safe/"
echo "$(ts) ============================================================"
