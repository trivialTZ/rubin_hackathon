#!/bin/bash -l
# Submit the full v2 retrain pipeline on SCC.
#
# This script:
#   1. Re-fetches ALeRCE data (parallel) to get BHRF/ATAT/stamp_2025_beta scores
#   2. Optionally updates TNS bulk CSV and re-runs crossmatch
#   3. Rebuilds gold tables with 51 LC features + 13 expert keys + survey flag
#   4. Retrains expert trust + followup models
#   5. Scores at n_det=3,5,10 and analyzes results
#
# IMPORTANT: This does NOT re-fetch lightcurves or Fink/Lasair data.
#            Existing bronze/silver/lightcurve caches are reused.
#            Only ALeRCE is re-queried (to pick up new classifiers).
#
# Env vars you need (in .env or exported):
#   LASAIR_TOKEN         — Lasair API token (get from https://lasair.lsst.ac.uk)
#   LASAIR_ENDPOINT      — https://lasair.lsst.ac.uk for LSST mode (optional)
#   TNS_API_KEY          — TNS API key (only if --update-tns)
#   TNS_TNS_ID           — TNS bot ID (only if --update-tns)
#   TNS_MARKER_NAME      — TNS marker name (only if --update-tns)
#
# Usage:
#   bash jobs/submit_retrain_v2.sh                     # retrain only
#   bash jobs/submit_retrain_v2.sh --update-tns        # also refresh TNS CSV
#   bash jobs/submit_retrain_v2.sh --update-alerce     # also re-fetch ALeRCE
#   bash jobs/submit_retrain_v2.sh --update-alerce --update-tns  # both
#   bash jobs/submit_retrain_v2.sh --dry-run           # print commands only
#   bash jobs/submit_retrain_v2.sh --local             # run locally (no SGE)
#
# How to get LASAIR_TOKEN:
#   1. Go to https://lasair.lsst.ac.uk
#   2. Create an account / log in
#   3. Click your username → "My Profile" → "API Token"
#   4. Copy the token and add to your .env:
#        LASAIR_TOKEN=your_token_here
#        LASAIR_ENDPOINT=https://lasair.lsst.ac.uk
#
# How to get TNS credentials:
#   1. Go to https://www.wis-tns.org
#   2. Register as a bot user (or use your group's existing bot)
#   3. Under "My Bots" → copy API Key, Bot ID, Marker Name
#   4. Add to your .env:
#        TNS_API_KEY=your_key_here
#        TNS_TNS_ID=12345
#        TNS_MARKER_NAME=your_marker_name

set -euo pipefail

: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

# Defaults
UPDATE_ALERCE=false
UPDATE_TNS=false
DRY_RUN=false
LOCAL_MODE=false
PARALLEL=8
LABELS="${DEBASS_ROOT}/data/labels.csv"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --update-alerce)  UPDATE_ALERCE=true; shift ;;
        --update-tns)     UPDATE_TNS=true; shift ;;
        --parallel)       PARALLEL="$2"; shift 2 ;;
        --labels)         LABELS="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --local)          LOCAL_MODE=true; shift ;;
        *)                echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "=== DEBASS v2 Retrain Pipeline ==="
echo "DEBASS_ROOT:    ${DEBASS_ROOT}"
echo "Labels:         ${LABELS}"
echo "Update ALeRCE:  ${UPDATE_ALERCE} (parallel=${PARALLEL})"
echo "Update TNS:     ${UPDATE_TNS}"
echo "Mode:           $(${LOCAL_MODE} && echo 'local' || echo 'SGE')"
echo ""

mkdir -p "${DEBASS_ROOT}/logs"

# -----------------------------------------------------------------------
# Local mode: run everything sequentially in the current shell
# -----------------------------------------------------------------------
if ${LOCAL_MODE}; then
    echo "=== Running locally ==="
    cd "${DEBASS_ROOT}"

    # Step 0: Install/upgrade dependencies
    echo "[0/7] Checking Python environment..."
    python3 --version
    pip install -q -r env/requirements.txt

    # Step 1: (Optional) Re-fetch ALeRCE to get BHRF/ATAT/Rubin classifiers
    if ${UPDATE_ALERCE}; then
        echo "[1/7] Re-fetching ALeRCE data (parallel=${PARALLEL})..."
        python3 scripts/backfill.py \
            --broker alerce \
            --from-labels "${LABELS}" \
            --parallel "${PARALLEL}"
    else
        echo "[1/7] Skipping ALeRCE re-fetch (use --update-alerce to enable)"
    fi

    # Step 2: (Optional) Update TNS CSV and re-run crossmatch
    if ${UPDATE_TNS}; then
        echo "[2/7] Updating TNS bulk CSV..."
        python3 scripts/download_tns_bulk.py
        echo "[2/7] Running TNS crossmatch (with positional fallback)..."
        python3 scripts/crossmatch_tns.py \
            --labels "${LABELS}" \
            --bulk-csv data/tns_public_objects.csv \
            --positional-fallback \
            --positional-radius 3.0
    else
        echo "[2/7] Skipping TNS update (use --update-tns to enable)"
    fi

    # Step 3: Normalize bronze → silver (picks up new ALeRCE classifiers)
    echo "[3/7] Normalizing bronze → silver..."
    python3 scripts/normalize.py

    # Step 4: Rebuild gold tables (51 LC features + 13 experts + survey flag)
    echo "[4/7] Rebuilding gold tables..."
    python3 scripts/build_object_epoch_snapshots.py \
        --objects-csv "${LABELS}" \
        --allow-unsafe-latest-snapshot
    python3 scripts/build_expert_helpfulness.py

    # Step 5: Train expert trust heads
    echo "[5/7] Training expert trust..."
    python3 scripts/train_expert_trust.py \
        --snapshots data/gold/object_epoch_snapshots.parquet \
        --helpfulness data/gold/expert_helpfulness.parquet \
        --models-dir models/trust \
        --output-snapshots data/gold/object_epoch_snapshots_trust.parquet

    # Step 6: Train followup model
    echo "[6/7] Training followup model..."
    python3 scripts/train_followup.py \
        --snapshots data/gold/object_epoch_snapshots_trust.parquet \
        --trust-models-dir models/trust \
        --model-dir models/followup

    # Step 7: Score and analyze
    echo "[7/7] Scoring (n_det=3,5,10)..."
    mkdir -p reports/scores
    for N in 3 5 10; do
        python3 scripts/score_nightly.py \
            --from-labels "${LABELS}" \
            --n-det "${N}" \
            --snapshots data/gold/object_epoch_snapshots_trust.parquet \
            --trust-models-dir models/trust \
            --followup-model-dir models/followup \
            --output reports/scores/scores_ndet${N}.jsonl
    done
    python3 scripts/analyze_results.py \
        --scores reports/scores/scores_ndet3.jsonl

    echo ""
    echo "=== Done ==="
    echo "Models:  models/trust/ + models/followup/"
    echo "Scores:  reports/scores/"
    echo "Summary: reports/results_summary.txt"
    exit 0
fi

# -----------------------------------------------------------------------
# SGE mode: submit job chain with dependencies
# -----------------------------------------------------------------------
COMMON="-V -P ${SGE_PROJECT:-pi-brout}"
PYSETUP="module load \${DEBASS_PYTHON_MODULE:-python3/3.10.12} && source \${DEBASS_VENV:-\$HOME/debass_meta_env}/bin/activate"

if ${DRY_RUN}; then
    echo "[dry-run] Would submit SGE chain. Commands below:"
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

# Step 1: (Optional) Re-fetch ALeRCE
HOLD_PREV=""
if ${UPDATE_ALERCE}; then
    JID1=$(submit_or_echo "alerce_refetch" $COMMON \
        -pe omp 1 -l mem_per_core=4G -l h_rt=01:00:00 \
        -N debass_v2_alerce_refetch -j y -o "logs/v2_alerce_refetch.\$JOB_ID.log" \
        -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
              python3 scripts/backfill.py --broker alerce --from-labels ${LABELS} --parallel ${PARALLEL}")
    echo "Step 1: ALeRCE re-fetch       → Job ${JID1}"
    HOLD_PREV="-hold_jid ${JID1}"
fi

# Step 2: (Optional) TNS update
if ${UPDATE_TNS}; then
    JID2=$(submit_or_echo "tns_update" $COMMON ${HOLD_PREV} \
        -pe omp 1 -l mem_per_core=4G -l h_rt=01:00:00 \
        -N debass_v2_tns_update -j y -o "logs/v2_tns_update.\$JOB_ID.log" \
        -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
              python3 scripts/download_tns_bulk.py && \
              python3 scripts/crossmatch_tns.py \
                  --labels ${LABELS} \
                  --bulk-csv data/tns_public_objects.csv \
                  --positional-fallback --positional-radius 3.0")
    echo "Step 2: TNS update            → Job ${JID2}"
    HOLD_PREV="-hold_jid ${JID2}"
fi

# Step 3: Normalize
JID3=$(submit_or_echo "normalize" $COMMON ${HOLD_PREV} \
    -pe omp 1 -l mem_per_core=4G -l h_rt=00:30:00 \
    -N debass_v2_normalize -j y -o "logs/v2_normalize.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          python3 scripts/normalize.py")
echo "Step 3: Normalize             → Job ${JID3}"

# Step 4: Rebuild gold tables
JID4=$(submit_or_echo "rebuild_gold" $COMMON -hold_jid "${JID3}" \
    -pe omp 1 -l mem_per_core=8G -l h_rt=02:00:00 \
    -N debass_v2_rebuild_gold -j y -o "logs/v2_rebuild_gold.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          python3 scripts/build_object_epoch_snapshots.py \
              --objects-csv ${LABELS} --allow-unsafe-latest-snapshot && \
          python3 scripts/build_expert_helpfulness.py")
echo "Step 4: Rebuild gold tables   → Job ${JID4}  (hold on ${JID3})"

# Step 5: Train expert trust
JID5=$(submit_or_echo "train_trust" $COMMON -hold_jid "${JID4}" \
    -pe omp 4 -l mem_per_core=8G -l h_rt=04:00:00 \
    -N debass_v2_train_trust -j y -o "logs/v2_train_trust.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          export OMP_NUM_THREADS=\${NSLOTS:-4} && \
          export OPENBLAS_NUM_THREADS=\${NSLOTS:-4} && \
          export MKL_NUM_THREADS=\${NSLOTS:-4} && \
          python3 scripts/train_expert_trust.py \
              --snapshots data/gold/object_epoch_snapshots.parquet \
              --helpfulness data/gold/expert_helpfulness.parquet \
              --models-dir models/trust \
              --output-snapshots data/gold/object_epoch_snapshots_trust.parquet \
              --n-jobs \${NSLOTS:-4}")
echo "Step 5: Train expert trust    → Job ${JID5}  (hold on ${JID4})"

# Step 6: Train followup
JID6=$(submit_or_echo "train_followup" $COMMON -hold_jid "${JID5}" \
    -pe omp 4 -l mem_per_core=8G -l h_rt=02:00:00 \
    -N debass_v2_train_followup -j y -o "logs/v2_train_followup.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          export OMP_NUM_THREADS=\${NSLOTS:-4} && \
          export OPENBLAS_NUM_THREADS=\${NSLOTS:-4} && \
          export MKL_NUM_THREADS=\${NSLOTS:-4} && \
          python3 scripts/train_followup.py \
              --snapshots data/gold/object_epoch_snapshots_trust.parquet \
              --trust-models-dir models/trust \
              --model-dir models/followup \
              --n-jobs \${NSLOTS:-4}")
echo "Step 6: Train followup        → Job ${JID6}  (hold on ${JID5})"

# Step 7: Score + analyze
JID7=$(submit_or_echo "score_analyze" $COMMON -hold_jid "${JID6}" \
    -pe omp 1 -l mem_per_core=8G -l h_rt=01:00:00 \
    -N debass_v2_score -j y -o "logs/v2_score.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          mkdir -p reports/scores && \
          for N in 3 5 10; do \
              echo \"Scoring n_det=\${N}...\"; \
              python3 scripts/score_nightly.py \
                  --from-labels ${LABELS} \
                  --n-det \${N} \
                  --snapshots data/gold/object_epoch_snapshots_trust.parquet \
                  --trust-models-dir models/trust \
                  --followup-model-dir models/followup \
                  --output reports/scores/scores_ndet\${N}.jsonl; \
          done && \
          python3 scripts/analyze_results.py --scores reports/scores/scores_ndet3.jsonl && \
          echo '' && echo '=== RESULTS SUMMARY ===' && \
          cat reports/results_summary.txt")
echo "Step 7: Score + analyze       → Job ${JID7}  (hold on ${JID6})"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  qstat -u \$USER"
echo ""
echo "When complete, check:"
echo "  cat \$DEBASS_ROOT/reports/results_summary.txt"
echo "  ls \$DEBASS_ROOT/models/trust/"
echo "  head \$DEBASS_ROOT/reports/scores/scores_ndet3.jsonl"
