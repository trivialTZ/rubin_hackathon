#!/bin/bash
#$ -N debass_v11_expert
#$ -cwd -V
#$ -l h_rt=10:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v11_expert.qsub.out
#$ -e logs/fusion_v11_expert.qsub.err
# metaDEBASS fusion_v11 — integration stage: fold the seq_v11 expert in with
# HONEST OOF train-row projections, then run the full v11 train → score → eval,
# then the locked-benchmark re-score (v11 AND v10 on the SAME cleaned truth, for
# apples-to-apples).
#
#   1. local_infer seq_v11 over every labeled object → silver per-epoch scores
#      with K-FOLD OOF ROUTING (fold_map.json). Single sequential process
#      (never concurrent local_infer against one silver dir — v6e.2 race).
#      DEBASS_SEQ_V11_MODEL pins the v11 OOF artifact; DEBASS_SEQ_V9_MODEL is
#      NEVER pointed at a v11 artifact.
#   2. FULL gold rebuild WITH seq_v11 (--truth object_truth_v11; same v11 split;
#      seq-train guard armed; --lsst-live-locked). proj__seq_v11__* auto-flows
#      into helpfulness/trust/anchor.
#   3. helpfulness → train_fusion_v11 (Stage A → hierarchical heads (G7) →
#      anchored blend (α, G3) → post-blend conformal → guards G2/G3/G6 → gates).
#   4. score_fusion_v11 (gold + DP1) → eval_fusion_v8 (G1 lives here, EVAL-only).
#   5. locked LSST-live benchmark re-score: v11 + v10 on the cleaned cohort truth
#      (G4 lives here, EVAL-only).
#
# Locked test stays value-identical (G5b in the build job); all new artifacts
# use the _v11 suffix. Idempotent (FUSION_V11_FORCE=1 rebuilds).
# Submit via jobs/submit_fusion_v11_chain.sh (holds on the GPU pretrain job).
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
mkdir -p logs data/scores reports/fusion_v11 reports/metrics

SNAP="data/gold/object_epoch_snapshots_fusion_v11.parquet"
SPLIT="data/gold/split_fusion_v11.json"
TRUTH_V11="data/truth/object_truth_v11.parquet"
TRUTH_MERGED="data/truth/object_truth_v11_merged.parquet"
# Rebuild gold from the MERGED truth (LSST-live spec + ztf_assoc_spec + B5
# catalog negatives) the build job produced; fall back to ZTF-only if absent.
TRUTH_BUILD="${TRUTH_V11}"
[[ -f "${TRUTH_MERGED}" ]] && TRUTH_BUILD="${TRUTH_MERGED}"
LOCKED="data/gold/lsst_live_locked_test.json"
HELP="data/gold/expert_helpfulness_fusion_v11.parquet"
SNAP_TRUST="${SNAP%.parquet}_trust.parquet"
DP1_SNAP="data/gold/dp1_snapshots_fusion_v11.parquet"
OOF_MARKER="data/silver/local_expert_outputs/seq_v11/.fusion_v11_oof_done"
FORCE="${FUSION_V11_FORCE:-0}"

# seq_v11 arm may be absent (gated / pretrain skipped): fold it in only if the
# OOF artifact exists. NEVER export DEBASS_SEQ_V9_MODEL at a v11 artifact.
HAVE_SEQ_V11=0
if [[ -f models/seq_classifier_v11/fold_map.json ]]; then
    export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"
    HAVE_SEQ_V11=1
fi

echo "$(ts) v11 expert — START (NSLOTS=${NSLOTS}, seq_v11=${HAVE_SEQ_V11})"
test -f "${SPLIT}" || { echo "missing ${SPLIT} (v11 build not finished?)"; exit 2; }
test -f "${TRUTH_V11}" || { echo "missing ${TRUTH_V11} (v11 build not finished?)"; exit 2; }

# 1. seq_v11 OOF inference into silver (sequential; marker idempotency)
if [[ "${HAVE_SEQ_V11}" == "1" ]]; then
    if [[ "${FORCE}" == "1" || ! -f "${OOF_MARKER}" ]]; then
        python3 -u scripts/local_infer.py --expert seq_v11 \
            --from-labels data/labels.csv \
            --lc-dir data/lightcurves --silver-dir data/silver --max-n-det 20
        touch "${OOF_MARKER}"
    else
        echo "$(ts) [skip] v11 OOF seq_v11 silver exists (marker ${OOF_MARKER})"
    fi

    # 2. FULL gold rebuild WITH seq_v11 (same manifest; guard armed)
    LOCKED_ARGS=(); [[ -f "${LOCKED}" ]] && LOCKED_ARGS=(--lsst-live-locked "${LOCKED}")
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --truth "${TRUTH_BUILD}" \
        --output "${SNAP}" --split-manifest "${SPLIT}" \
        --association-csv data/crossmatch/lsst_to_ztf.csv \
        --seq-train-ids models/seq_classifier_v11/fold_map.json \
        "${LOCKED_ARGS[@]}" \
        --dp1 --dp1-output "${DP1_SNAP}"
else
    echo "$(ts) [skip] seq_v11 arm absent — gold from build job kept (seq_v11 OUT of blend)"
    if [[ "${FORCE}" == "1" || ! -f "${DP1_SNAP}" ]]; then
        LOCKED_ARGS=(); [[ -f "${LOCKED}" ]] && LOCKED_ARGS=(--lsst-live-locked "${LOCKED}")
        python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
            --truth "${TRUTH_BUILD}" --dp1-only \
            "${LOCKED_ARGS[@]}" --dp1-output "${DP1_SNAP}"
    fi
fi

# 3. helpfulness
if [[ "${FORCE}" == "1" || ! -f "${HELP}" ]]; then
    python3 -u scripts/build_helpfulness_fusion.py \
        --snapshots "${SNAP}" --output "${HELP}"
else
    echo "$(ts) [skip] helpfulness exists: ${HELP}"
fi

# 4. full train (Stage A → hierarchical heads → anchored blend → conformal →
#    guards/gates). --truth pins the B0 ordering guard.
#    --acknowledge-g2-unevaluable: the current LSST spec-Ia pool is near-empty
#    (B1), so G2 starts G2_UNEVALUABLE (n<10). Acknowledging lets the chain
#    complete and STAMPS the report headline — never a silent pass. A genuine G2
#    FAIL (n>=10, level regressed) still hard-fails the job. Drop this flag once
#    the LSST spec pool clears n>=10 to make G2 a hard gate.
python3 -u scripts/train_fusion_v11.py --n-jobs "${NSLOTS}" \
    --snapshots "${SNAP}" --helpfulness "${HELP}" --split "${SPLIT}" \
    --truth "${TRUTH_V11}" --lsst-locked "${LOCKED}" \
    --output-snapshots "${SNAP_TRUST}" \
    --trust-dir models/trust_fusion_v11 \
    --followup-dir models/followup_fusion_v11 \
    --blend-dir models/anchor_blend_v11 \
    --conformal-dir models/conformal_fusion_v11 \
    --fdr-gamma 0.1 --fdr-n-det-max 5 \
    --acknowledge-g2-unevaluable \
    --build-report reports/metrics/fusion_v11_build.json \
    --metrics-out reports/metrics/fusion_v11_train.json

# 5. score (gold + DP1) with the anchored blend
python3 -u scripts/score_fusion_v11.py --tag fusion_v11 \
    --snapshots "${SNAP_TRUST}" \
    --split "${SPLIT}" --fdr-gamma 0.1 --fdr-n-det-max 5 \
    --trust-dir models/trust_fusion_v11 \
    --followup-dir models/followup_fusion_v11 \
    --blend-dir models/anchor_blend_v11 \
    --conformal models/conformal_fusion_v11/mondrian_aps.pkl
python3 -u scripts/score_fusion_v11.py --tag fusion_v11 --dp1 \
    --snapshots "${DP1_SNAP}" \
    --split "${SPLIT}" \
    --trust-dir models/trust_fusion_v11 \
    --followup-dir models/followup_fusion_v11 \
    --blend-dir models/anchor_blend_v11 \
    --conformal models/conformal_fusion_v11/mondrian_aps.pkl

# 6. eval (G1 ZTF locked-test macro AUC vs v10 CI lives HERE — EVAL-only)
python3 -u scripts/eval_fusion_v8.py \
    --pred data/scores/predictions_fusion_v11.parquet \
    --pred-dp1 data/scores/predictions_fusion_v11_dp1.parquet \
    --snapshots "${SNAP_TRUST}" --split "${SPLIT}" \
    --train-metrics reports/metrics/fusion_v11_train.json \
    --out-dir reports/fusion_v11

# 7. locked LSST-live benchmark re-score — v11 AND v10 on the SAME cleaned truth
#    (RUNBOOK scoring flow; G4 SN-vs-other blend>=anchor lives at eval time).
if [[ -f "${LOCKED}" ]]; then
    echo "$(ts) v11 expert — locked LSST-live benchmark re-score (v11 + v10)"
    CLEANED="data/truth/object_truth_20260704_cleaned.parquet"
    COHORT_LABELS="data/live_eval_20260704/cohort/labels.csv"
    LC_DIR="data/live_eval_20260704/lightcurves"
    SILVER_DIR="data/live_eval_20260704/silver"
    BENCH="reports/lsst_live_bench"
    mkdir -p "${BENCH}/scores"
    EMPTY_SPLIT="${BENCH}/empty_trust_metadata.json"
    printf '{"train_ids": [], "cal_ids": [], "test_ids": []}\n' > "${EMPTY_SPLIT}"
    BSNAP="${BENCH}/snapshots_fusion_v11.parquet"
    if [[ -f "${CLEANED}" && -d "${LC_DIR}" ]]; then
        python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
            --lc-dir "${LC_DIR}" --silver-dir "${SILVER_DIR}" \
            --truth "${CLEANED}" --bts "" --labels "${COHORT_LABELS}" \
            --trust-metadata "${EMPTY_SPLIT}" --no-lsst-weak \
            --lsst-live-locked "${LOCKED}" \
            --output "${BSNAP}" --split-manifest "${BENCH}/split_fusion_v11.json" \
            --skip-traj
        # v11 (anchored blend)
        python3 -u scripts/score_fusion_v11.py --tag lsst_live_v11 \
            --snapshots "${BSNAP}" \
            --trust-dir models/trust_fusion_v11 \
            --followup-dir models/followup_fusion_v11 \
            --blend-dir models/anchor_blend_v11 \
            --conformal models/conformal_fusion_v11/mondrian_aps.pkl \
            --scores-dir "${BENCH}/scores" --no-priority
        # v10 (re-scored on the SAME cleaned truth — apples-to-apples)
        python3 -u scripts/score_fusion_v8.py --tag lsst_live_v10 \
            --snapshots "${BSNAP}" \
            --trust-dir models/trust_fusion_v10 \
            --followup-dir models/followup_fusion_v10 \
            --conformal models/conformal_fusion_v10/mondrian_aps.pkl \
            --scores-dir "${BENCH}/scores" --no-priority
    else
        echo "$(ts) WARN: cleaned cohort truth / lightcurves absent — benchmark re-score skipped" >&2
    fi
fi

echo "$(ts) v11 expert — DONE"
python3 - <<'EOF'
import json, pathlib
tr = pathlib.Path("reports/metrics/fusion_v11_train.json")
if tr.exists():
    payload = json.loads(tr.read_text())
    print("V11 GUARD STATUSES:", json.dumps(payload.get("guard_statuses", {}), indent=0))
    for e in payload.get("gates", []) or []:
        if isinstance(e, dict):
            print("GATE:", e.get("gate", e.get("component")), "→", e.get("decision"))
    g2 = payload.get("guards", {}).get("G2", {})
    print("G2:", json.dumps({k: g2.get(k) for k in
          ("status", "n_rows", "median_p_snia", "max_p_snia", "n_lc_all_nan")}))
hg = pathlib.Path("reports/fusion_v11/headline_guards.json")
if hg.exists():
    print("V11 HEADLINE:", json.dumps(json.loads(hg.read_text()).get("headline", {}), indent=0)[:900])
EOF
