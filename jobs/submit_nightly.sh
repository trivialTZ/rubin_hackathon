#!/bin/bash -l
# Submit the nightly discovery + scoring pipeline on SCC.
#
# This assumes trained models already exist (from submit_phase1_chain.sh).
# No retraining happens — only discover → fetch → backfill → normalize →
# build snapshots → score.
#
# Usage:
#   bash jobs/submit_nightly.sh                         # today's date
#   bash jobs/submit_nightly.sh --date 2026-04-02       # specific date
#   bash jobs/submit_nightly.sh --lookback-days 7       # wider lookback
#   bash jobs/submit_nightly.sh --dry-run               # print commands only
#   bash jobs/submit_nightly.sh --local                 # run locally (no SGE)
#
# Cron setup (SCC login node):
#   0 10 * * * source ~/.bashrc && export DEBASS_ROOT=/projectnb/pi-brout/rubin_hackathon && bash $DEBASS_ROOT/jobs/submit_nightly.sh >> $DEBASS_ROOT/logs/cron_nightly.log 2>&1

set -euo pipefail

: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

# Defaults
DATE=$(date -u +%Y-%m-%d)
LOOKBACK_DAYS=2
MAX_CANDIDATES=500
DRY_RUN=false
LOCAL_MODE=false
MODELS_DIR="${DEBASS_ROOT}/published_results/models"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)         DATE="$2"; shift 2 ;;
        --lookback-days) LOOKBACK_DAYS="$2"; shift 2 ;;
        --max-candidates) MAX_CANDIDATES="$2"; shift 2 ;;
        --models-dir)   MODELS_DIR="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --local)        LOCAL_MODE=true; shift ;;
        *)              echo "Unknown flag: $1"; exit 1 ;;
    esac
done

NIGHTLY_DIR="${DEBASS_ROOT}/data/nightly/${DATE}"
CANDIDATES="${NIGHTLY_DIR}/candidates.csv"
LC_DIR="${NIGHTLY_DIR}/lightcurves"
BRONZE_DIR="${NIGHTLY_DIR}/bronze"
SILVER_DIR="${NIGHTLY_DIR}/silver"
GOLD_DIR="${NIGHTLY_DIR}/gold"
SCORES_DIR="${NIGHTLY_DIR}/scores"
TRUST_DIR="${MODELS_DIR}/trust"
FOLLOWUP_DIR="${MODELS_DIR}/followup"

# Global truth table (if it exists — snapshots handle missing truth gracefully)
TRUTH_PATH="${DEBASS_ROOT}/data/truth/object_truth.parquet"

echo "=== DEBASS Nightly Pipeline ==="
echo "Date:        ${DATE}"
echo "Lookback:    ${LOOKBACK_DAYS} days"
echo "Nightly dir: ${NIGHTLY_DIR}"
echo "Models:      ${MODELS_DIR}"
echo "Mode:        $(${LOCAL_MODE} && echo 'local' || echo 'SGE')"
echo ""

mkdir -p "${NIGHTLY_DIR}" "${DEBASS_ROOT}/logs"

# -----------------------------------------------------------------------
# Local mode: run everything sequentially in the current shell
# -----------------------------------------------------------------------
if ${LOCAL_MODE}; then
    echo "=== Running locally ==="

    echo "[1/7] Discovering candidates..."
    python3 scripts/discover_nightly.py \
        --lookback-days "${LOOKBACK_DAYS}" \
        --max-candidates "${MAX_CANDIDATES}" \
        --output "${CANDIDATES}"

    if [ ! -s "${CANDIDATES}" ] || [ "$(wc -l < "${CANDIDATES}")" -le 1 ]; then
        echo "No candidates found. Exiting."
        exit 0
    fi

    echo "[2/7] Fetching lightcurves..."
    python3 scripts/fetch_lightcurves.py \
        --from-labels "${CANDIDATES}" \
        --lc-dir "${LC_DIR}"

    echo "[3/7] Backfilling broker scores..."
    python3 scripts/backfill.py \
        --broker all \
        --from-labels "${CANDIDATES}" \
        --bronze-dir "${BRONZE_DIR}"

    echo "[4/7] Normalizing bronze → silver..."
    python3 scripts/normalize.py \
        --bronze-dir "${BRONZE_DIR}" \
        --silver-dir "${SILVER_DIR}"

    echo "[5/7] Building epoch snapshots..."
    python3 scripts/build_object_epoch_snapshots.py \
        --lc-dir "${LC_DIR}" \
        --silver-dir "${SILVER_DIR}" \
        --gold-dir "${GOLD_DIR}" \
        --objects-csv "${CANDIDATES}" \
        --allow-unsafe-latest-snapshot \
        --truth "${TRUTH_PATH}"

    SNAPSHOT="${GOLD_DIR}/object_epoch_snapshots.parquet"
    if [ ! -f "${SNAPSHOT}" ]; then
        # Try the _trust variant
        SNAPSHOT="${GOLD_DIR}/object_epoch_snapshots_trust.parquet"
    fi

    echo "[6/7] Scoring (n_det=3,4,5)..."
    mkdir -p "${SCORES_DIR}"
    for N in 3 4 5; do
        python3 scripts/score_nightly.py \
            --snapshots "${SNAPSHOT}" \
            --trust-models-dir "${TRUST_DIR}" \
            --followup-model-dir "${FOLLOWUP_DIR}" \
            --n-det "${N}" \
            --output "${SCORES_DIR}/scores_ndet${N}.jsonl"
    done

    echo "[7/7] Summarizing..."
    python3 scripts/summarize_nightly.py \
        --scores-dir "${SCORES_DIR}" \
        --output "${NIGHTLY_DIR}/summary.txt"

    echo ""
    echo "=== Done ==="
    echo "Scores:  ${SCORES_DIR}/"
    echo "Summary: ${NIGHTLY_DIR}/summary.txt"
    exit 0
fi

# -----------------------------------------------------------------------
# SGE mode: submit job chain with dependencies
# -----------------------------------------------------------------------
COMMON="-V -P ${SGE_PROJECT:-pi-brout}"
PYSETUP="module load \${DEBASS_PYTHON_MODULE:-python3/3.10.12} && source \${DEBASS_VENV:-\$HOME/debass_meta_env}/bin/activate"

if ${DRY_RUN}; then
    echo "[dry-run] Would submit 7-step SGE chain. Commands below:"
    echo ""
fi

submit_or_echo() {
    local name="$1"; shift
    if ${DRY_RUN}; then
        echo "  [dry-run] ${name}:" >&2
        echo "  qsub $@" >&2
        echo "" >&2
        echo "DRYRUN_${name}"
    else
        qsub -terse "$@"
    fi
}

# Step 1: Discover candidates
JID1=$(submit_or_echo "discover" $COMMON \
    -pe omp 1 -l mem_per_core=4G -l h_rt=00:30:00 \
    -N debass_nightly_discover -j y -o "logs/nightly_discover_${DATE}.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          python3 scripts/discover_nightly.py \
              --lookback-days ${LOOKBACK_DAYS} \
              --max-candidates ${MAX_CANDIDATES} \
              --output ${CANDIDATES}")
echo "Step 1: Discover              → Job ${JID1}"

# Step 2: Fetch lightcurves
JID2=$(submit_or_echo "fetch_lc" $COMMON -hold_jid "${JID1}" \
    -pe omp 1 -l mem_per_core=4G -l h_rt=01:00:00 \
    -N debass_nightly_fetch_lc -j y -o "logs/nightly_fetch_lc_${DATE}.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          python3 scripts/fetch_lightcurves.py \
              --from-labels ${CANDIDATES} \
              --lc-dir ${LC_DIR}")
echo "Step 2: Fetch lightcurves     → Job ${JID2}  (hold on ${JID1})"

# Step 3: Backfill broker scores
JID3=$(submit_or_echo "backfill" $COMMON -hold_jid "${JID2}" \
    -pe omp 1 -l mem_per_core=4G -l h_rt=01:00:00 \
    -N debass_nightly_backfill -j y -o "logs/nightly_backfill_${DATE}.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          python3 scripts/backfill.py \
              --broker all \
              --from-labels ${CANDIDATES} \
              --bronze-dir ${BRONZE_DIR}")
echo "Step 3: Backfill brokers      → Job ${JID3}  (hold on ${JID2})"

# Step 4: Normalize bronze → silver
JID4=$(submit_or_echo "normalize" $COMMON -hold_jid "${JID3}" \
    -pe omp 1 -l mem_per_core=4G -l h_rt=00:30:00 \
    -N debass_nightly_normalize -j y -o "logs/nightly_normalize_${DATE}.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          python3 scripts/normalize.py \
              --bronze-dir ${BRONZE_DIR} \
              --silver-dir ${SILVER_DIR}")
echo "Step 4: Normalize             → Job ${JID4}  (hold on ${JID3})"

# Step 5: Build epoch snapshots
JID5=$(submit_or_echo "snapshots" $COMMON -hold_jid "${JID4}" \
    -pe omp 1 -l mem_per_core=8G -l h_rt=01:00:00 \
    -N debass_nightly_snapshots -j y -o "logs/nightly_snapshots_${DATE}.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          python3 scripts/build_object_epoch_snapshots.py \
              --lc-dir ${LC_DIR} \
              --silver-dir ${SILVER_DIR} \
              --gold-dir ${GOLD_DIR} \
              --objects-csv ${CANDIDATES} \
              --allow-unsafe-latest-snapshot \
              --truth ${TRUTH_PATH}")
echo "Step 5: Build snapshots       → Job ${JID5}  (hold on ${JID4})"

# Step 6: Score at n_det=3,4,5
JID6=$(submit_or_echo "score" $COMMON -hold_jid "${JID5}" \
    -pe omp 1 -l mem_per_core=8G -l h_rt=00:30:00 \
    -N debass_nightly_score -j y -o "logs/nightly_score_${DATE}.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          mkdir -p ${SCORES_DIR} && \
          SNAPSHOT=${GOLD_DIR}/object_epoch_snapshots.parquet && \
          [ -f \${SNAPSHOT} ] || SNAPSHOT=${GOLD_DIR}/object_epoch_snapshots_trust.parquet && \
          for N in 3 4 5; do \
              echo \"Scoring n_det=\${N}...\"; \
              python3 scripts/score_nightly.py \
                  --snapshots \${SNAPSHOT} \
                  --trust-models-dir ${TRUST_DIR} \
                  --followup-model-dir ${FOLLOWUP_DIR} \
                  --n-det \${N} \
                  --output ${SCORES_DIR}/scores_ndet\${N}.jsonl; \
          done && echo 'Done:' \$(date)")
echo "Step 6: Score (n=3,4,5)       → Job ${JID6}  (hold on ${JID5})"

# Step 7: Summarize
JID7=$(submit_or_echo "summarize" $COMMON -hold_jid "${JID6}" \
    -pe omp 1 -l mem_per_core=4G -l h_rt=00:15:00 \
    -N debass_nightly_summary -j y -o "logs/nightly_summary_${DATE}.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          python3 scripts/summarize_nightly.py \
              --scores-dir ${SCORES_DIR} \
              --output ${NIGHTLY_DIR}/summary.txt && \
          echo '' && echo '=== NIGHTLY SUMMARY ===' && \
          cat ${NIGHTLY_DIR}/summary.txt")
echo "Step 7: Summarize             → Job ${JID7}  (hold on ${JID6})"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  qstat -u \$USER"
echo ""
echo "When complete, check:"
echo "  cat ${NIGHTLY_DIR}/summary.txt"
echo "  ls ${SCORES_DIR}/"
