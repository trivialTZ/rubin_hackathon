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
# Env vars (in $DEBASS_ROOT/.env):
#   LASAIR_TOKEN             — shared Lasair token (used for both ZTF + LSST)
#   LASAIR_ZTF_TOKEN         — (optional) separate ZTF Lasair token
#   LASAIR_LSST_TOKEN        — (optional) separate LSST Lasair token
#   LASAIR_ZTF_ENDPOINT      — (optional) default: https://lasair-ztf.lsst.ac.uk
#   LASAIR_LSST_ENDPOINT     — (optional) default: https://lasair.lsst.ac.uk
#   TNS_API_KEY              — TNS API key (only if --update-tns)
#   TNS_TNS_ID               — TNS bot ID (only if --update-tns)
#   TNS_MARKER_NAME          — TNS marker name (only if --update-tns)
#
# Usage:
#   bash jobs/submit_retrain_v2.sh                         # retrain only
#   bash jobs/submit_retrain_v2.sh --update-alerce         # re-fetch ALeRCE
#   bash jobs/submit_retrain_v2.sh --update-tns            # refresh TNS
#   bash jobs/submit_retrain_v2.sh --discover-lsst 200     # discover 200 LSST objects
#   bash jobs/submit_retrain_v2.sh --update-alerce --update-tns --discover-lsst 200
#   bash jobs/submit_retrain_v2.sh --dry-run               # print commands only
#   bash jobs/submit_retrain_v2.sh --local                 # run locally (no SGE)
#
# How to get LASAIR_TOKEN (same token works for both ZTF and LSST):
#   1. Go to https://lasair.lsst.ac.uk
#   2. Create an account / log in
#   3. Click your username → "My Profile" → "API Token"
#   4. Add to $DEBASS_ROOT/.env:
#        LASAIR_TOKEN=your_token_here
#
# How to get TNS credentials:
#   1. Go to https://www.wis-tns.org
#   2. Register as a bot user (or use your group's existing bot)
#   3. Under "My Bots" → copy API Key, Bot ID, Marker Name
#   4. Add to $DEBASS_ROOT/.env:
#        TNS_API_KEY=your_key_here
#        TNS_TNS_ID=12345
#        TNS_MARKER_NAME=your_marker_name

set -euo pipefail

: "${DEBASS_ROOT:?DEBASS_ROOT must be set before running this job}"

# Defaults
UPDATE_ALERCE=false
UPDATE_TNS=false
DISCOVER_LSST=0
DRY_RUN=false
LOCAL_MODE=false
PARALLEL=8
LABELS="${DEBASS_ROOT}/data/labels.csv"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --update-alerce)  UPDATE_ALERCE=true; shift ;;
        --update-tns)     UPDATE_TNS=true; shift ;;
        --discover-lsst)  DISCOVER_LSST="$2"; shift 2 ;;
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
echo "Discover LSST:  ${DISCOVER_LSST} objects (0=skip)"
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
    echo "[0/8] Checking Python environment..."
    python3 --version
    pip install -q -r env/requirements.txt

    # Step 1: (Optional) Discover LSST transient candidates + merge into labels
    if [ "${DISCOVER_LSST}" -gt 0 ] 2>/dev/null; then
        echo "[1/8] Discovering ${DISCOVER_LSST} LSST candidates from ALeRCE..."
        LSST_CSV="${DEBASS_ROOT}/data/lsst_candidates.csv"
        python3 scripts/discover_lsst_training.py \
            --max-candidates "${DISCOVER_LSST}" \
            --min-detections 2 \
            --output "${LSST_CSV}"

        echo "[1/8] Merging LSST candidates into labels.csv..."
        python3 scripts/merge_labels.py \
            --new "${LSST_CSV}" \
            --labels "${LABELS}"
    else
        echo "[1/8] Skipping LSST discovery (use --discover-lsst N to enable)"
    fi

    # Always merge lsst_candidates.csv if it exists (from a previous run)
    LSST_CSV="${DEBASS_ROOT}/data/lsst_candidates.csv"
    if [ -f "${LSST_CSV}" ] && [ "${DISCOVER_LSST}" -eq 0 ] 2>/dev/null; then
        echo "[1/8] Found existing lsst_candidates.csv — merging into labels..."
        python3 scripts/merge_labels.py \
            --new "${LSST_CSV}" \
            --labels "${LABELS}"
    fi

    # Step 2: (Optional) Re-fetch ALeRCE to get BHRF/ATAT/Rubin classifiers
    if ${UPDATE_ALERCE}; then
        echo "[2/8] Re-fetching ALeRCE data (parallel=${PARALLEL})..."
        python3 scripts/backfill.py \
            --broker alerce \
            --from-labels "${LABELS}" \
            --parallel "${PARALLEL}"
    else
        echo "[2/8] Skipping ALeRCE re-fetch (use --update-alerce to enable)"
    fi

    # Step 3: (Optional) Update TNS CSV and re-run crossmatch
    if ${UPDATE_TNS}; then
        echo "[3/8] Updating TNS bulk CSV..."
        python3 scripts/download_tns_bulk.py
        echo "[3/8] Running TNS crossmatch (with positional fallback)..."
        python3 scripts/crossmatch_tns.py \
            --labels "${LABELS}" \
            --bulk-csv data/tns_public_objects.csv \
            --positional-fallback \
            --positional-radius 3.0
    else
        echo "[3/8] Skipping TNS update (use --update-tns to enable)"
    fi

    # Step 4: Fetch lightcurves + backfill all brokers for new objects
    echo "[4/8] Fetching lightcurves + broker scores..."
    python3 scripts/fetch_lightcurves.py --from-labels "${LABELS}"
    python3 scripts/backfill.py --broker all --from-labels "${LABELS}" --parallel "${PARALLEL}"

    # Step 5a: Normalize bronze → silver (picks up new ALeRCE classifiers)
    echo "[5a/9] Normalizing bronze → silver..."
    python3 scripts/normalize.py

    # Step 5b: Build multi-source truth (TNS + Fink xm + host context + labels)
    echo "[5b/9] Building multi-source truth table..."
    python3 scripts/build_truth_multisource.py \
        --labels "${LABELS}" \
        --silver-dir data/silver

    # Step 5c: Collect per-epoch classification history from local experts
    #          This is the key to learning epoch-dependent trust for LSST objects
    #          where broker APIs only give static snapshots.
    echo "[5b/9] Collecting per-epoch local expert history..."
    python3 scripts/collect_epoch_history.py \
        --from-labels "${LABELS}" \
        --lc-dir data/lightcurves \
        --max-n-det 20

    # Step 6: Rebuild gold tables (51 LC features + 13 experts + survey flag)
    echo "[6/9] Rebuilding gold tables..."
    python3 scripts/build_object_epoch_snapshots.py \
        --objects-csv "${LABELS}" \
        --allow-unsafe-latest-snapshot
    python3 scripts/build_expert_helpfulness.py

    # Step 7: Train expert trust heads
    echo "[7/8] Training expert trust..."
    python3 scripts/train_expert_trust.py \
        --snapshots data/gold/object_epoch_snapshots.parquet \
        --helpfulness data/gold/expert_helpfulness.parquet \
        --models-dir models/trust \
        --output-snapshots data/gold/object_epoch_snapshots_trust.parquet

    # Step 7b: Train followup model
    echo "[7b/8] Training followup model..."
    python3 scripts/train_followup.py \
        --snapshots data/gold/object_epoch_snapshots_trust.parquet \
        --trust-models-dir models/trust \
        --model-dir models/followup

    # Step 8: Score and analyze
    echo "[8/8] Scoring (n_det=3,5,10)..."
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

# Step 0: Discover LSST + merge into labels (always runs if lsst_candidates.csv exists)
HOLD_PREV=""
LSST_CSV="${DEBASS_ROOT}/data/lsst_candidates.csv"
DISCOVER_CMD=""
if [ "${DISCOVER_LSST}" -gt 0 ] 2>/dev/null; then
    DISCOVER_CMD="python3 scripts/discover_lsst_training.py \
        --max-candidates ${DISCOVER_LSST} --min-detections 2 \
        --output ${LSST_CSV} && "
fi
# Always merge if lsst_candidates.csv exists (idempotent)
JID0=$(submit_or_echo "discover_merge" $COMMON \
    -pe omp 1 -l mem_per_core=4G -l h_rt=01:00:00 \
    -N debass_v2_discover -j y -o "logs/v2_discover.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          ${DISCOVER_CMD} \
          if [ -f ${LSST_CSV} ]; then \
              python3 scripts/merge_labels.py --new ${LSST_CSV} --labels ${LABELS}; \
          else \
              echo 'No lsst_candidates.csv — skipping merge'; \
          fi")
echo "Step 0: Discover + merge      → Job ${JID0}"
HOLD_PREV="-hold_jid ${JID0}"

# Step 1: Fetch lightcurves + backfill all brokers (for any new objects from merge)
JID1=$(submit_or_echo "fetch_backfill" $COMMON ${HOLD_PREV} \
    -pe omp 1 -l mem_per_core=4G -l h_rt=02:00:00 \
    -N debass_v2_fetch -j y -o "logs/v2_fetch_backfill.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          python3 scripts/fetch_lightcurves.py --from-labels ${LABELS} && \
          python3 scripts/backfill.py --broker all --from-labels ${LABELS} --parallel ${PARALLEL}")
echo "Step 1: Fetch + backfill      → Job ${JID1}  (hold on ${JID0})"
HOLD_PREV="-hold_jid ${JID1}"

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

# Step 3: Normalize + multi-source truth + collect local expert epochs
JID3=$(submit_or_echo "normalize_truth" $COMMON ${HOLD_PREV} \
    -pe omp 1 -l mem_per_core=4G -l h_rt=01:00:00 \
    -N debass_v2_normalize -j y -o "logs/v2_normalize.\$JOB_ID.log" \
    -b y "cd $DEBASS_ROOT && ${PYSETUP} && \
          echo 'Normalizing...' && \
          python3 scripts/normalize.py && \
          echo 'Building multi-source truth...' && \
          python3 scripts/build_truth_multisource.py --labels ${LABELS} --silver-dir data/silver && \
          echo 'Collecting local expert epoch history...' && \
          python3 scripts/collect_epoch_history.py --from-labels ${LABELS} --max-n-det 20")
echo "Step 3: Normalize + truth     → Job ${JID3}"

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
