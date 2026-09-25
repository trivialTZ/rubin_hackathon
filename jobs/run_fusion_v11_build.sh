#!/bin/bash
#$ -N debass_v11_build
#$ -cwd -V
#$ -l h_rt=06:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v11_build.qsub.out
#$ -e logs/fusion_v11_build.qsub.err
# metaDEBASS fusion_v11 — stage 1 on SCC: land the v11 truth table (B0), build
# the FULL v11 gold (positives-only epochs + 5 negative-flux features, B2/B3),
# emit the FINAL split manifest, then run G5b (locked ZTF gold value-identity)
# right after the gold build — the order that keeps every downstream training
# job honest (the v10 chain-ordering fix carries over: nothing trains before the
# manifest it obeys).
#
# ORDER OF OPERATIONS (spec §3, P6): P1's rederive (B0) + P2's is_positive fix
# (B2) land HERE, first. seq_v11 (P5) is trained in the pretrain job and folded
# in by the expert job — G5b only checks v10-era columns (base-51+EXT+traj+
# proj__/avail__/exact__), so seq_v11's absence from THIS gold is irrelevant to
# the identity check.
#
# All builds take --truth data/truth/object_truth_v11.parquet (B0).
# --lsst-live-locked forces the frozen benchmark ids out of train∪cal at BUILD
# (G6-at-build); --association-csv groups LSST↔ZTF counterparts.
#
# Idempotent: completed artifacts are reused (FUSION_V11_FORCE=1 rebuilds).
# Submit via jobs/submit_fusion_v11_chain.sh.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
mkdir -p logs data/gold data/truth data/scores reports/metrics

TNS_PARQUET="data/truth/tns_public.parquet"
TRUTH_V11="data/truth/object_truth_v11.parquet"
TRUTH_MERGED="data/truth/object_truth_v11_merged.parquet"
LIVE_TRUTH="data/truth/lsst_live_truth.parquet"
CLEANED_TRUTH="data/truth/object_truth_20260704_cleaned.parquet"
COHORT_CSV="data/live_eval_20260704/cohort/merged.csv"
ASSOC_CSV="data/crossmatch/lsst_to_ztf.csv"
SNAP="data/gold/object_epoch_snapshots_fusion_v11.parquet"
SPLIT="data/gold/split_fusion_v11.json"
LOCKED="data/gold/lsst_live_locked_test.json"
V10_GOLD="data/gold/object_epoch_snapshots_fusion_v10.parquet"
V10_HASH="data/gold/object_epoch_snapshots_fusion_v10.hashmanifest.json"
FORCE="${FUSION_V11_FORCE:-0}"

echo "$(ts) v11 build — START (NSLOTS=${NSLOTS})"

# 0. TNS bulk (authoritative type/discoverydate; rederive + live truth need it)
if [[ "${FORCE}" == "1" || ! -f "${TNS_PARQUET}" ]]; then
    python3 -u scripts/download_tns_bulk.py --out "${TNS_PARQUET}" --mode auto || {
        echo "$(ts) WARN: TNS bulk fetch failed — rederive falls back to BTS+existing truth" >&2; }
else
    echo "$(ts) [skip] TNS bulk exists: ${TNS_PARQUET}"
fi

# 1. P1 rederive: object_truth_v11.parquet (+ label_delta_v11.csv). NEVER
#    overwrites the original truth table (B0). Head-2 refuses to fit without it.
if [[ "${FORCE}" == "1" || ! -f "${TRUTH_V11}" ]]; then
    python3 -u scripts/rederive_spec_truth.py \
        --truth data/truth/object_truth.parquet \
        --bts data/truth/ztf_bts.parquet \
        --tns-bulk "${TNS_PARQUET}" \
        --locked-split data/gold/split_fusion_v10.json \
        --out "${TRUTH_V11}" --delta-out data/truth/label_delta_v11.csv
else
    echo "$(ts) [skip] v11 truth exists: ${TRUTH_V11}"
fi
test -f "${TRUTH_V11}" || { echo "missing ${TRUTH_V11} — rederive must succeed"; exit 2; }

# Belt-and-braces seq-train guard on rebuilds (mirror v10): a v11 classifier's
# train objects must never drift into cal in a rebuilt manifest.
SEQ_TRAIN_ARGS=()
if [[ -f models/seq_classifier_v11/fold_map.json ]]; then
    SEQ_TRAIN_ARGS=(--seq-train-ids models/seq_classifier_v11/fold_map.json)
    echo "$(ts) seq-train guard armed with models/seq_classifier_v11/fold_map.json"
fi

# 1e. P1 LSST-live data steps (spec §3 P1). The training chain — not just the
#     weekly refresh — must produce these, or the counterpart quarantine
#     (needs the association CSV), G6-at-build (needs the frozen manifest), and
#     G2's assoc-spec term / LSST head-1 negatives (need the merged truth) are
#     all inert on a fresh clone (findings: dead association output; G6 no-op;
#     truth-plumbing pin). Each needs Lasair creds/cohort data — degrade LOUDLY,
#     never silently, so the gold build below still runs ZTF-only if they fail.
#
# 1e-i. Association harvest -> lsst_to_ztf.csv (+ ztf_assoc_spec.csv). The
#       canonical path every consumer reads (--association-csv default).
if [[ "${FORCE}" == "1" || ! -f "${ASSOC_CSV}" ]]; then
    python3 -u scripts/harvest_ztf_lsst_associations.py --out "${ASSOC_CSV}" || {
        echo "$(ts) WARN: association harvest failed (no Lasair creds/network) — "\
             "counterpart quarantine inert this build" >&2; }
else
    echo "$(ts) [skip] association CSV exists: ${ASSOC_CSV}"
fi
ASSOC_ARGS=()
[[ -f "${ASSOC_CSV}" ]] && ASSOC_ARGS=(--association-csv "${ASSOC_CSV}")

# 1e-ii. Frozen benchmark manifest + cleaned cohort truth (cohort-clean;
#        APPEND-ONLY — frozen ids never move). Creates ${LOCKED} so G6-at-build
#        is actually armed (the manifest is gitignored and does NOT ship via git).
if [[ -f "${COHORT_CSV}" ]]; then
    if [[ "${FORCE}" == "1" || ! -f "${LOCKED}" ]]; then
        python3 -u scripts/build_truth_lsst_live.py --mode cohort-clean \
            --tns-bulk "${TNS_PARQUET}" --cohort "${COHORT_CSV}" \
            --cleaned-out "${CLEANED_TRUTH}" --manifest-out "${LOCKED}" || {
              echo "$(ts) WARN: cohort-clean failed — benchmark manifest not (re)built" >&2; }
    else
        echo "$(ts) [skip] benchmark manifest exists: ${LOCKED}"
    fi
else
    echo "$(ts) WARN: cohort CSV ${COHORT_CSV} absent — benchmark manifest not built "\
         "(G6-at-build disarmed; train will hard-fail without the manifest)" >&2
fi

# 1e-iii. LSST-live truth seed (new spec arrivals + ztf_assoc_spec rows +
#         catalog negatives, B5). Needs Lasair — degrade to ZTF-only on failure.
if [[ "${FORCE}" == "1" || ! -f "${LIVE_TRUTH}" ]]; then
    python3 -u scripts/build_truth_lsst_live.py --mode live \
        --tns-bulk "${TNS_PARQUET}" --live-out "${LIVE_TRUTH}" \
        --manifest-in "${LOCKED}" || {
          echo "$(ts) WARN: LSST-live seed skipped (no Lasair creds/network) — "\
               "merged truth == rederived ZTF-only truth" >&2; }
fi

# 1e-iv. TRUTH PLUMBING PIN (spec §3, P1 FINAL step): fold lsst_live_truth into
#        the rederived ZTF v11 truth (LSST-live rows win). The MERGED file is the
#        training-build --truth so LSST-live spec, ztf_assoc_spec (G2's union
#        term) and B5 catalog negatives reach gold. Idempotent; merged==base when
#        no live seed exists.
python3 -u scripts/build_truth_lsst_live.py --mode merge \
    --base-truth "${TRUTH_V11}" --merge-live "${LIVE_TRUTH}" \
    --merged-out "${TRUTH_MERGED}" || {
      echo "$(ts) WARN: truth merge failed — gold build falls back to ZTF-only truth" >&2; }
TRUTH_BUILD="${TRUTH_V11}"
[[ -f "${TRUTH_MERGED}" ]] && TRUTH_BUILD="${TRUTH_MERGED}"
echo "$(ts) gold build truth: ${TRUTH_BUILD}"

# 2. FULL v11 gold + FINAL split (positives-only epochs + neg features; frozen
#    experts + traj; seq_v11 folded in later by the expert job). G6-at-build via
#    --lsst-live-locked; missing manifest -> no-op (mirrors --association-csv).
LOCKED_ARGS=()
[[ -f "${LOCKED}" ]] && LOCKED_ARGS=(--lsst-live-locked "${LOCKED}")
if [[ "${FORCE}" == "1" || ! -f "${SNAP}" || ! -f "${SPLIT}" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --truth "${TRUTH_BUILD}" \
        --output "${SNAP}" --split-manifest "${SPLIT}" \
        "${ASSOC_ARGS[@]}" \
        "${LOCKED_ARGS[@]}" "${SEQ_TRAIN_ARGS[@]}"
else
    echo "$(ts) [skip] gold snapshot + split exist: ${SNAP}, ${SPLIT}"
fi

# 3. G5b: value-identity of the rebuilt v11 gold's ZTF rows vs locked v10 gold
#    (v10-era columns only). SCC has the v10 parquet; locally use the hash
#    manifest. Never compares file bytes.
echo "$(ts) v11 build — G5b (locked ZTF gold value-identity)"
if [[ -f "${V10_GOLD}" ]]; then
    python3 -u scripts/assert_locked_gold_identity.py \
        --reference "${V10_GOLD}" --rebuilt "${SNAP}"
elif [[ -f "${V10_HASH}" ]]; then
    python3 -u scripts/assert_locked_gold_identity.py \
        --reference-hash-manifest "${V10_HASH}" --rebuilt "${SNAP}"
else
    echo "$(ts) WARN: no v10 reference gold or hash manifest — G5b SKIPPED (record in report)" >&2
fi

echo "$(ts) v11 build — DONE"
python3 - <<'EOF'
import json
m = json.load(open("data/gold/split_fusion_v11.json"))
print("V11 SPLIT:", {k: m[k] for k in (
    "counts", "n_new", "n_new_test_reassigned_to_train",
    "seq_train_guard", "association_grouped_split",
    "n_new_quarantined_test_counterparts",
    "lsst_live_locked", "n_lsst_live_locked_quarantined") if k in m})
EOF
