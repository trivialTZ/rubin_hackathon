#!/bin/bash -l
# Submit the full Phase 1 pipeline chain on SCC.
#
# This assumes:
#   1. SNN inference has completed (data/silver/local_expert_outputs/supernnova/part-latest.parquet exists)
#   2. All other bronze/silver data is ready
#   3. TNS truth crossmatch is done (data/truth/object_truth.parquet exists)
#
# Usage:
#   bash jobs/submit_phase1_chain.sh                     # submit all steps
#   bash jobs/submit_phase1_chain.sh --after-snn <JID>   # hold on SNN job
#
# Each step depends on the previous one via -hold_jid.

set -euo pipefail

: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

SNN_HOLD_JID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --after-snn)
            SNN_HOLD_JID="$2"
            shift 2
            ;;
        *)
            echo "Unknown flag: $1"
            exit 1
            ;;
    esac
done

COMMON="-V -P pi-brout"
HOLD_SNN=""
if [ -n "$SNN_HOLD_JID" ]; then
    HOLD_SNN="-hold_jid $SNN_HOLD_JID"
fi

echo "=== DEBASS Phase 1 Pipeline Chain ==="
echo "DEBASS_ROOT: $DEBASS_ROOT"
echo ""

# Step 1: Rebuild gold tables (object_epoch_snapshots + expert_helpfulness)
JID1=$(qsub -terse $COMMON $HOLD_SNN \
    -pe omp 1 -l mem_per_core=8G -l h_rt=02:00:00 \
    -N debass_rebuild_gold -j y -o 'logs/rebuild_gold.$JOB_ID.log' \
    -b y "cd $DEBASS_ROOT && \
          module load \${DEBASS_PYTHON_MODULE:-python3/3.10.12} && \
          source \${DEBASS_VENV:-\$HOME/debass_meta_env}/bin/activate && \
          echo '=== Rebuilding gold tables ===' && echo 'Start:' \$(date) && \
          python3 scripts/build_object_epoch_snapshots.py && \
          python3 scripts/build_expert_helpfulness.py && \
          echo 'Done:' \$(date)")
echo "Step 1: Rebuild gold tables   → Job $JID1"

# Step 2: Train expert trust heads
JID2=$(qsub -terse $COMMON -hold_jid "$JID1" \
    -pe omp 4 -l mem_per_core=8G -l h_rt=04:00:00 \
    -N debass_meta_train_expert_trust -j y -o 'logs/train_expert_trust.$JOB_ID.log' \
    -b y "cd $DEBASS_ROOT && \
          module load \${DEBASS_PYTHON_MODULE:-python3/3.10.12} && \
          source \${DEBASS_VENV:-\$HOME/debass_meta_env}/bin/activate && \
          export OMP_NUM_THREADS=\${NSLOTS:-4} && \
          export OPENBLAS_NUM_THREADS=\${NSLOTS:-4} && \
          export MKL_NUM_THREADS=\${NSLOTS:-4} && \
          python3 scripts/train_expert_trust.py \
              --snapshots data/gold/object_epoch_snapshots.parquet \
              --helpfulness data/gold/expert_helpfulness.parquet \
              --models-dir models/trust \
              --output-snapshots data/gold/object_epoch_snapshots_trust.parquet \
              --n-jobs \${NSLOTS:-4}")
echo "Step 2: Train expert trust    → Job $JID2  (hold on $JID1)"

# Step 3: Train follow-up model
JID3=$(qsub -terse $COMMON -hold_jid "$JID2" \
    -pe omp 4 -l mem_per_core=8G -l h_rt=02:00:00 \
    -N debass_meta_train_followup -j y -o 'logs/train_followup.$JOB_ID.log' \
    -b y "cd $DEBASS_ROOT && \
          module load \${DEBASS_PYTHON_MODULE:-python3/3.10.12} && \
          source \${DEBASS_VENV:-\$HOME/debass_meta_env}/bin/activate && \
          export OMP_NUM_THREADS=\${NSLOTS:-4} && \
          export OPENBLAS_NUM_THREADS=\${NSLOTS:-4} && \
          export MKL_NUM_THREADS=\${NSLOTS:-4} && \
          python3 scripts/train_followup.py \
              --snapshots data/gold/object_epoch_snapshots_trust.parquet \
              --trust-models-dir models/trust \
              --model-dir models/followup \
              --n-jobs \${NSLOTS:-4}")
echo "Step 3: Train follow-up       → Job $JID3  (hold on $JID2)"

# Step 4: Score at n_det = 3, 5, 10
JID4=$(qsub -terse $COMMON -hold_jid "$JID3" \
    -pe omp 1 -l mem_per_core=8G -l h_rt=01:00:00 \
    -N debass_meta_score_all -j y -o 'logs/score_all.$JOB_ID.log' \
    -b y "cd $DEBASS_ROOT && \
          module load \${DEBASS_PYTHON_MODULE:-python3/3.10.12} && \
          source \${DEBASS_VENV:-\$HOME/debass_meta_env}/bin/activate && \
          mkdir -p reports/scores && \
          for N in 3 5 10; do \
              echo \"Scoring n_det=\$N...\"; \
              python3 scripts/score_nightly.py \
                  --from-labels data/labels.csv \
                  --n-det \$N \
                  --snapshots data/gold/object_epoch_snapshots_trust.parquet \
                  --trust-models-dir models/trust \
                  --followup-model-dir models/followup \
                  --output reports/scores/scores_ndet\${N}.jsonl; \
          done && echo 'Done:' \$(date)")
echo "Step 4: Score all (n=3,5,10)  → Job $JID4  (hold on $JID3)"

# Step 5: Analyze results
JID5=$(qsub -terse $COMMON -hold_jid "$JID4" \
    -pe omp 1 -l mem_per_core=8G -l h_rt=00:30:00 \
    -N debass_meta_analyze -j y -o 'logs/analyze_results.$JOB_ID.log' \
    -b y "cd $DEBASS_ROOT && \
          module load \${DEBASS_PYTHON_MODULE:-python3/3.10.12} && \
          source \${DEBASS_VENV:-\$HOME/debass_meta_env}/bin/activate && \
          python3 scripts/analyze_results.py \
              --scores reports/scores/scores_ndet3.jsonl && \
          echo '' && echo '=== RESULTS SUMMARY ===' && \
          cat reports/results_summary.txt")
echo "Step 5: Analyze results       → Job $JID5  (hold on $JID4)"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  qstat -u \$USER"
echo ""
echo "When complete, check:"
echo "  cat \$DEBASS_ROOT/reports/results_summary.txt"
echo "  cat \$DEBASS_ROOT/logs/analyze_results.\${JID5%.}.log"
