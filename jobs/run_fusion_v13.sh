#!/bin/bash
#$ -N debass_v13
#$ -cwd -V
#$ -l h_rt=10:00:00
#$ -l mem_per_core=8G
#$ -pe omp 8
#$ -o logs/fusion_v13.qsub.out
#$ -e logs/fusion_v13.qsub.err
# metaDEBASS fusion_v13 (docs/fusion_v13_plan.md) — a retrain on the fusion_v12w gold (v12 labels + the ALeRCE-stamp
# weak LSST rows), with:
#   * Stage A: stamp classifiers (+ BHRF top) trained on is_sn; LSST weak rows only for is_sn trust heads, never ALeRCE
#     (ZTF weak rows as in v12); q_prior only for experts with a trust head (what scoring emits);
#     q_prior out-of-fold on train rows; no q__ columns for experts without a trust head.
#   * ParSNIP dropped (a constant stub whose availability marked new LSST spectroscopic objects).
#   * Heads: LSST weak rows out of head 1, its calibrators and alpha; no class-pure LSST equalization; the context
#     family (Sherlock, Babamul) masked on all LSST rows at fit and serve time; guard G8 on
#     availability-vs-class correlation; availability dropout (no-broker / no-local / no-expert regimes + random drops);
#     head 1 cross-fitted so its calibrators, alpha and G2 use out-of-fold train rows plus cal.
# No gold rebuild: reuses data/gold/*_fusion_v12w.* (gold, helpfulness, split, DP1). Afterwards scores the frozen LSST
# benchmark (plus the availability ablation) and the DP2 lightcurve+local-expert gold.
#
# FUSION_V13_ARM=nodrop: control arm *_v13nd — identical but without availability dropout, to isolate it.
# FUSION_V13_ARM=v13b: *_v13b — v13 plus (docs/fusion_v13_plan.md, Results) head-1 calibrators weighted by object mix
#   (--head1-cal-weights object) and local SuperNNova masked on LSST (--head1-survey-mask lsst:supernnova: its LSST
#   training rows are uniform stubs marked available, real outputs only on re-run SNe and at serving). Stage A is
#   v13's (unchanged), copied in unless FUSION_V13_STAGE_A_FROM is set to another arm.
# FUSION_V13_ARM=v13c: *_v13c — the v13b recipe (Stage A retrained) on the rebuilt fusion_v13c golds
#   (jobs/run_v13c_salt3_array.sh, jobs/run_fusion_v13c_gold.sh): LSST salt3_chi2 / alerce_lc training rows timed
#   (no more averaging over every epoch), SALT3 re-fitted with the per-point Birge-rescaled mapping everywhere. The
#   benchmark, DP2 and explorer-cohort golds are the v13c ones; only v13c is scored on them (older stacks were trained
#   on the old SALT3 outputs; their benchmark predictions stay as they are).
# FUSION_V13_ARM=v13d: *_v13d — v13c's golds and Stage A (copied in), heads and blend refitted with the v13d fixes
#   (docs/fusion_v13_plan.md, v13d): event_count__ / exact__ kept out of both heads; the anchor weights the is_sn
#   trust-head experts by the trust of their call; alpha fitted on the SN-vs-other log-loss with an object-clustered
#   SE; the anchor's P(Ia|SN) fallback per object. v13c is scored on the same benchmark ablation for a paired table.
# FUSION_V13_ARM=v13e: *_v13e — v13d with alpha the grid minimum per cell (--alpha-rule best): v13d's
#   object-clustered SE (0.05 to 0.08 in the LSST cells, few SN objects) made the 1-SE rule pick alpha 0.25 / 0 where
#   the out-of-fold loss is lowest at 0.75 / 0.5. Stage A from v13c; v13c and v13d scored on the same ablation.
# FUSION_V13_SMOKE=1: --smoke, outputs *_v13_smoke.
# FUSION_V13_REUSE_STAGE_A=1: reuse this arm's Stage-A snapshot + trust dir from an earlier run (--skip-stage-a).
# FUSION_V13_STAGE_A_FROM=<sfx>: copy that arm's Stage-A trust dir (+ link its snapshot) and reuse it.
# G8_MAX (default 0.25): the full-gold dry run (2026-09-25) gave LSST max 0.17 (lc_features_bv) and ZTF 0.21
#   (salt3_chi2, lc_features_bv), both genuine availability gaps that recur at scoring.
#   qsub -P pi-brout jobs/run_fusion_v13.sh
#   qsub -P pi-brout -v FUSION_V13_ARM=nodrop jobs/run_fusion_v13.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-8}
ARM="${FUSION_V13_ARM:-}"
SMOKE="${FUSION_V13_SMOKE:-0}"
SFX=v13; SMOKE_ARGS=(); ACK_ARGS=(); G8_ARGS=()
if [[ "${G8_ACK:-0}" == "1" ]]; then G8_ARGS=(--acknowledge-g8); fi   # record a G8 violation and continue
DROP_ARGS=(--head1-dropout); ARM_ARGS=(); STAGE_A_FROM="${FUSION_V13_STAGE_A_FROM:-}"
BASE=v12w; GOLD_TAG=v12; DP2_TAG=v12loc; ABL_MODELS=(v12 v12w)
if [[ "${ARM}" == "nodrop" ]]; then SFX=v13nd; DROP_ARGS=(); fi
if [[ "${ARM}" == "v13b" ]]; then
    SFX=v13b; STAGE_A_FROM="${STAGE_A_FROM:-v13}"
    ARM_ARGS=(--head1-cal-weights object --head1-survey-mask lsst:supernnova)
fi
if [[ "${ARM}" == "v13c" ]]; then
    SFX=v13c; BASE=v13c; GOLD_TAG=v13c; DP2_TAG=v13cloc; ABL_MODELS=()
    ARM_ARGS=(--head1-cal-weights object --head1-survey-mask lsst:supernnova)
fi
if [[ "${ARM}" == "v13d" ]]; then
    SFX=v13d; BASE=v13c; GOLD_TAG=v13c; DP2_TAG=v13cloc; ABL_MODELS=(v13c); STAGE_A_FROM="${STAGE_A_FROM:-v13c}"
    ARM_ARGS=(--head1-cal-weights object --head1-survey-mask lsst:supernnova
              --head-drop-feature-prefix event_count__ --head-drop-feature-prefix exact__
              --alpha-objective sn_binary --alpha-se object --anchor-base-rate-unit object
              --anchor-call-weight-sn-filter)
fi
if [[ "${ARM}" == "v13e" ]]; then
    SFX=v13e; BASE=v13c; GOLD_TAG=v13c; DP2_TAG=v13cloc; ABL_MODELS=(v13c v13d); STAGE_A_FROM="${STAGE_A_FROM:-v13c}"
    ARM_ARGS=(--head1-cal-weights object --head1-survey-mask lsst:supernnova
              --head-drop-feature-prefix event_count__ --head-drop-feature-prefix exact__
              --alpha-objective sn_binary --alpha-se object --anchor-base-rate-unit object
              --anchor-call-weight-sn-filter --alpha-rule best)
fi
if [[ "${SMOKE}" == "1" ]]; then
    SFX="${SFX}_smoke"; SMOKE_ARGS=(--smoke)
    ACK_ARGS=(--acknowledge-g7-not-evaluable)
fi
mkdir -p logs data/scores "reports/fusion_${SFX}" reports/metrics

SNAP="data/gold/object_epoch_snapshots_fusion_${BASE}.parquet"
HELP="data/gold/expert_helpfulness_fusion_${BASE}.parquet"
SPLIT="data/gold/split_fusion_${BASE}.json"
DP1_SNAP="data/gold/dp1_snapshots_fusion_${BASE}.parquet"
TRUTH_V11="data/truth/object_truth_v11.parquet"
LOCKED="data/gold/lsst_live_locked_test.json"
SNAP_TRUST="data/gold/object_epoch_snapshots_fusion_${SFX}_trust.parquet"
TRUST="models/trust_fusion_${SFX}"; FOLLOW="models/followup_fusion_${SFX}"
BLEND="models/anchor_blend_${SFX}"; CONF="models/conformal_fusion_${SFX}"
BENCH="data/label_refresh_20260924/bench"
BENCH_GOLD="${BENCH}/gold/bench_${GOLD_TAG}.parquet"
DP2_GOLD="data/edp2_train/gold/snapshots_${DP2_TAG}.parquet"
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"

REUSE_ARGS=()
if [[ -n "${STAGE_A_FROM}" && "${SMOKE}" != "1" ]]; then
    SRC_SNAP="data/gold/object_epoch_snapshots_fusion_${STAGE_A_FROM}_trust.parquet"
    test -f "${SRC_SNAP}" && test -d "models/trust_fusion_${STAGE_A_FROM}" || { echo "no Stage A for ${STAGE_A_FROM}"; exit 2; }
    [[ -e "${SNAP_TRUST}" ]] || ln -s "$(basename "${SRC_SNAP}")" "${SNAP_TRUST}"
    [[ -d "${TRUST}" ]] || cp -r "models/trust_fusion_${STAGE_A_FROM}" "${TRUST}"
    FUSION_V13_REUSE_STAGE_A=1
fi
if [[ "${FUSION_V13_REUSE_STAGE_A:-0}" == "1" && -f "${SNAP_TRUST}" && -d "${TRUST}" ]]; then REUSE_ARGS=(--skip-stage-a); fi
echo "$(ts) fusion ${SFX} — START (NSLOTS=${NSLOTS}; reuse Stage A: ${#REUSE_ARGS[@]})"
for f in "${SNAP}" "${HELP}" "${SPLIT}" "${DP1_SNAP}" "${TRUTH_V11}" "${LOCKED}" "${BENCH_GOLD}" \
         "${BENCH}/truth.parquet" "${DP2_GOLD}"; do
    test -f "$f" || { echo "missing input $f"; exit 2; }
done

# 1. train
TRAIN_RC=0
python3 -u scripts/train_fusion_v11.py --n-jobs "${NSLOTS}" \
    --snapshots "${SNAP}" --helpfulness "${HELP}" --split "${SPLIT}" \
    --truth "${TRUTH_V11}" --lsst-locked "${LOCKED}" \
    --output-snapshots "${SNAP_TRUST}" \
    --trust-dir "${TRUST}" --followup-dir "${FOLLOW}" --blend-dir "${BLEND}" --conformal-dir "${CONF}" \
    --fdr-gamma 0.1 --fdr-n-det-max 5 \
    --stage-a-weak-policy lsst_is_sn_only --stage-a-q-prior-experts trained \
    --head1-exclude-quality lsst:weak --no-lsst-equalization --head1-context-mask survey --drop-expert parsnip \
    --g8-max-corr "${G8_MAX:-0.25}" "${G8_ARGS[@]}" --cross-fit-folds 5 "${REUSE_ARGS[@]}" \
    "${DROP_ARGS[@]}" ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} "${SMOKE_ARGS[@]}" "${ACK_ARGS[@]}" \
    --build-report "reports/metrics/fusion_${SFX}_build.json" \
    --metrics-out "reports/metrics/fusion_${SFX}_train.json" || TRAIN_RC=$?
echo "$(ts) train exit code ${TRAIN_RC}"
test -f "${FOLLOW}/model.pkl" || { echo "no follow-up model written — stopping"; exit 3; }

# 2. score (gold + DP1) and eval, as for v12
python3 -u scripts/score_fusion_v11.py --tag "fusion_${SFX}" --snapshots "${SNAP_TRUST}" \
    --split "${SPLIT}" --fdr-gamma 0.1 --fdr-n-det-max 5 \
    --trust-dir "${TRUST}" --followup-dir "${FOLLOW}" --blend-dir "${BLEND}" --conformal "${CONF}/mondrian_aps.pkl"
python3 -u scripts/score_fusion_v11.py --tag "fusion_${SFX}" --dp1 --snapshots "${DP1_SNAP}" --split "${SPLIT}" \
    --trust-dir "${TRUST}" --followup-dir "${FOLLOW}" --blend-dir "${BLEND}" --conformal "${CONF}/mondrian_aps.pkl"
python3 -u scripts/eval_fusion_v8.py \
    --pred "data/scores/predictions_fusion_${SFX}.parquet" \
    --pred-dp1 "data/scores/predictions_fusion_${SFX}_dp1.parquet" \
    --snapshots "${SNAP_TRUST}" --split "${SPLIT}" \
    --train-metrics "reports/metrics/fusion_${SFX}_train.json" \
    --out-dir "reports/fusion_${SFX}"

# 3. frozen LSST benchmark: full inputs + availability ablation (v12 and v12w re-scored alongside for a paired table,
#    except on the v13c gold)
ABL_ARGS=(--model "${SFX}")
for m in ${ABL_MODELS[@]+"${ABL_MODELS[@]}"}; do ABL_ARGS+=(--model "${m}"); done
python3 -u scripts/eval_input_ablation.py --gold "${BENCH_GOLD}" --truth "${BENCH}/truth.parquet" \
    "${ABL_ARGS[@]}" --out-dir "${BENCH}/ablate_${SFX}"
cp "${BENCH}/ablate_${SFX}/predictions_abl_full_${SFX}.parquet" "${BENCH}/scores/predictions_bench_${SFX}.parquet"

# 4. DP2 typed objects: lightcurve + local experts, no brokers
python3 -u scripts/score_fusion_v11.py --tag "dp2_${SFX}" --snapshots "${DP2_GOLD}" \
    --trust-dir "${TRUST}" --followup-dir "${FOLLOW}" --blend-dir "${BLEND}" --conformal "${CONF}/mondrian_aps.pkl" \
    --scores-dir data/edp2_train/scores --no-priority --require-local-experts

# 5. the TNS×EDP2 explorer cohort (golds built by jobs/run_tnsx_v12_score.sh; lightcurve + brokers + local experts)
TNSX="data/tnsx_eval_20260924"
for sv in lsst ztf; do
    if [[ -f "${TNSX}/gold/snapshots_${sv}_${GOLD_TAG}.parquet" ]]; then
        python3 -u scripts/score_fusion_v11.py --tag "tnsx_${sv}_${SFX}" --snapshots "${TNSX}/gold/snapshots_${sv}_${GOLD_TAG}.parquet" \
            --trust-dir "${TRUST}" --followup-dir "${FOLLOW}" --blend-dir "${BLEND}" --conformal "${CONF}/mondrian_aps.pkl" \
            --scores-dir "${TNSX}/scores" --no-priority
    fi
done

echo "$(ts) fusion ${SFX} — DONE (train exit ${TRAIN_RC})"
python3 - "${SFX}" <<'EOF'
import json, pathlib, sys
sfx = sys.argv[1]
tr = pathlib.Path(f"reports/metrics/fusion_{sfx}_train.json")
if tr.exists():
    p = json.loads(tr.read_text())
    print("GUARD STATUSES:", json.dumps(p.get("guard_statuses", {})))
    for e in p.get("gates", []) or []:
        if isinstance(e, dict):
            print("GATE:", e.get("gate", e.get("component")), "->", e.get("decision"), "|", str(e.get("detail", ""))[:160])
hg = pathlib.Path(f"reports/fusion_{sfx}/headline_guards.json")
if hg.exists():
    print("HEADLINE:", json.dumps(json.loads(hg.read_text()).get("headline", {}))[:900])
EOF
cat "data/label_refresh_20260924/bench/ablate_${SFX}/ablation_metrics.md"
exit "${TRAIN_RC}"
