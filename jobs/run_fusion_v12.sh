#!/bin/bash
#$ -N debass_v12
#$ -cwd -V
#$ -l h_rt=12:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v12.qsub.out
#$ -e logs/fusion_v12.qsub.err
# metaDEBASS fusion_v12 — the v11 stack retrained with rebuilt Rubin/LSST labels (2026-09-24;
# docs/metadebass_rubin_review_20260924.md):
#   * CATS projector fixed (docs/metadebass_cats_bug.md): code 21 = Periodic is no longer a non-Ia SN vote.
#   * LSST labels = TNS spectroscopic types + Gaia/SIMBAD catalogue context (scripts/build_truth_fusion_v12.py,
#     scripts/build_lsst_catalog_context.py). No ALeRCE-stamp weak labels (--no-lsst-weak) and no Sherlock-only
#     context: both copied a broker that is also a model input.
#   * LSST training inputs match what scoring sees: Pitt-Google events are stripped (never fetched at scoring,
#     BigQuery bills); Babamul and the ALeRCE Rubin stamp are backfilled for every labelled LSST object; local
#     experts are run for the new objects.
# The frozen LSST benchmark (data/gold/lsst_live_locked_test.json) stays quarantined from the gold, as in v11, and
# is scored on the Mac afterwards. Same seq_v11 GRU artifact as v11 (no pretrain stage).
#
# Staged from the Mac before submitting (data/label_refresh_20260924/):
#   lsst_candidates_catalog.parquet  backlog_train_truth.parquet  backlog_holdout_truth.parquet
#   silver_lsst/broker_events.parquet (99 new objects)  babamul/silver/broker_events.parquet (Babamul + ALeRCE backfill)
#   v12_new_lsst_ids.csv (objects whose local experts must be run)
#   and the new lightcurves rsynced into data/lightcurves_v12/ (a hard-link copy of data/lightcurves).
#
# FUSION_V12_SMOKE=1: --smoke gold (~200 objects) into *_v12_smoke outputs, to check the plumbing.
# FUSION_V12_WEAK=1: control arm *_v12w — identical, but the ALeRCE-stamp weak LSST labels are kept (v11 policy), so
#   v12 vs v12w isolates the label change. Run it only after the v12 job has finished steps 1-3 (shared silver).
# FUSION_V12_FORCE=1: redo the silver / local-expert / gold / helpfulness steps.
#   qsub -P pi-brout jobs/run_fusion_v12.sh
#   qsub -P pi-brout -v FUSION_V12_SMOKE=1 -l h_rt=03:00:00 jobs/run_fusion_v12.sh
#   qsub -P pi-brout -v FUSION_V12_WEAK=1 -hold_jid <v12 job id> jobs/run_fusion_v12.sh
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
SMOKE="${FUSION_V12_SMOKE:-0}"
FORCE="${FUSION_V12_FORCE:-0}"
WEAK="${FUSION_V12_WEAK:-0}"
SFX=v12; SMOKE_ARGS=(); ACK_ARGS=(); WEAK_ARGS=(--no-lsst-weak)
if [[ "${WEAK}" == "1" ]]; then
    SFX=v12w; WEAK_ARGS=()
fi
if [[ "${SMOKE}" == "1" ]]; then
    SFX=v12_smoke; SMOKE_ARGS=(--smoke)
    ACK_ARGS=(--acknowledge-g7-not-evaluable --acknowledge-split-predates-manifest)
fi
mkdir -p logs data/scores "reports/fusion_${SFX}" reports/metrics

STAGE="data/label_refresh_20260924"
LC="data/lightcurves_v12"
SILVER="data/silver_v12"
TRUTH_V11="data/truth/object_truth_v11.parquet"
TRUTH="data/truth/object_truth_v12_merged.parquet"
LOCKED="data/gold/lsst_live_locked_test.json"
SNAP="data/gold/object_epoch_snapshots_fusion_${SFX}.parquet"
SPLIT="data/gold/split_fusion_${SFX}.json"
HELP="data/gold/expert_helpfulness_fusion_${SFX}.parquet"
SNAP_TRUST="${SNAP%.parquet}_trust.parquet"
DP1_SNAP="data/gold/dp1_snapshots_fusion_${SFX}.parquet"
TRUST="models/trust_fusion_${SFX}"; FOLLOW="models/followup_fusion_${SFX}"
BLEND="models/anchor_blend_${SFX}"; CONF="models/conformal_fusion_${SFX}"
export DEBASS_SEQ_V11_MODEL="models/seq_classifier_v11"   # same OOF artifact as v11; seq_v9 resolves to seq_classifier_v10

echo "$(ts) fusion ${SFX} — START (NSLOTS=${NSLOTS})"
for f in "${STAGE}/lsst_candidates_catalog.parquet" "${STAGE}/backlog_train_truth.parquet" \
         "${STAGE}/backlog_holdout_truth.parquet" "${STAGE}/silver_lsst/broker_events.parquet" \
         "${STAGE}/babamul/silver/broker_events.parquet" "${STAGE}/v12_new_lsst_ids.csv" \
         data/truth/object_truth_v11_merged.parquet "${LOCKED}" models/seq_classifier_v11/fold_map.json; do
    test -f "$f" || { echo "missing staged input $f"; exit 2; }
done
test -d "${LC}" || { echo "missing ${LC} (cp -al data/lightcurves ${LC}, then rsync the new lightcurves)"; exit 2; }

# 1. truth: v11 truth with the LSST labels rebuilt (the v12w arm reuses the v12 job's file)
[[ "${WEAK}" == "1" && -f "${TRUTH}" ]] || python3 -u scripts/build_truth_fusion_v12.py --base data/truth/object_truth_v11_merged.parquet \
    --lsst-catalog "${STAGE}/lsst_candidates_catalog.parquet" \
    --lsst-spec "${STAGE}/backlog_train_truth.parquet" \
    --exclude "${STAGE}/backlog_holdout_truth.parquet" \
    --out "${TRUTH}"

# 2. silver_v12 = v11 silver − Pitt-Google + the staged new/backfilled events. A real copy, not hard links:
#    local_infer rewrites local_expert_outputs in place.
if [[ "${WEAK}" == "1" && ! -f "${SILVER}/.v12_local_experts_done" ]]; then
    echo "the v12w arm needs the v12 job's silver (steps 1-3) first"; exit 2
fi
if [[ "${WEAK}" != "1" ]] && [[ "${FORCE}" == "1" || ! -f "${SILVER}/.v12_ready" ]]; then
    if [[ "${FORCE}" == "1" || ! -f "${SILVER}/.v12_copied" ]]; then
        if [[ -e "${SILVER}" ]]; then mv "${SILVER}" "${SILVER}.old_$(date +%Y%m%d%H%M%S)"; fi
        echo "$(ts) copying data/silver -> ${SILVER}"
        cp -a data/silver "${SILVER}"
        touch "${SILVER}/.v12_copied"
    fi
    python3 - "${SILVER}" "${STAGE}" <<'EOF'
import sys
import pandas as pd
silver, stage = sys.argv[1], sys.argv[2]
ev = pd.read_parquet(f"{silver}/broker_events.parquet")
ev["object_id"] = ev["object_id"].astype(str)
n0 = len(ev)
pgb = ev["expert_key"].astype(str).str.startswith("pittgoogle/")
ev = ev[~pgb]
adds = []
for p in (f"{stage}/silver_lsst/broker_events.parquet", f"{stage}/babamul/silver/broker_events.parquet"):
    a = pd.read_parquet(p)
    a["object_id"] = a["object_id"].astype(str)
    a = a[[c for c in ev.columns]]
    for c in ev.columns:            # match the v11 silver's column types (e.g. alert_id is text there)
        if ev[c].dtype == object and a[c].dtype != object:
            a[c] = a[c].map(lambda v: None if pd.isna(v) else str(v))
        elif ev[c].dtype != a[c].dtype and ev[c].dtype != object:
            try:
                a[c] = a[c].astype(ev[c].dtype)
            except (TypeError, ValueError):
                pass
    adds.append(a)
add = pd.concat(adds, ignore_index=True)
# a fresh fetch supersedes older events of the same (object, expert)
key = ev["object_id"] + "|" + ev["expert_key"].astype(str)
new_keys = set(add["object_id"] + "|" + add["expert_key"].astype(str))
old = key.isin(new_keys)
out = pd.concat([ev[~old], add], ignore_index=True)
out.to_parquet(f"{silver}/broker_events.parquet.tmp", index=False)
import os
os.replace(f"{silver}/broker_events.parquet.tmp", f"{silver}/broker_events.parquet")
print(f"silver_v12: {n0:,} events; dropped {int(pgb.sum()):,} Pitt-Google and {int(old.sum()):,} superseded; "
      f"added {len(add):,} for {add['object_id'].nunique():,} objects -> {len(out):,}")
print(out[out["object_id"].str.match(r"^\d{15,20}$")].groupby("expert_key")["object_id"].nunique().to_string())
EOF
    touch "${SILVER}/.v12_ready"
else
    echo "$(ts) [skip] ${SILVER} ready"
fi

# 3. local experts for the new LSST objects (sequential; upserts into the v12 silver)
MARK="${SILVER}/.v12_local_experts_done"
if [[ "${WEAK}" != "1" ]] && [[ "${FORCE}" == "1" || ! -f "${MARK}" ]]; then
    python3 -u scripts/local_infer.py --expert all --from-labels "${STAGE}/v12_new_lsst_ids.csv" \
        --lc-dir "${LC}" --silver-dir "${SILVER}" --max-n-det 20
    touch "${MARK}"
else
    echo "$(ts) [skip] local experts for new objects done"
fi

# 4. gold (+ DP1) with the v12 truth; LSST weak labels off; benchmark quarantined; seq-train guard armed
if [[ "${FORCE}" == "1" || ! -f "${SNAP}" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --lc-dir "${LC}" --silver-dir "${SILVER}" --truth "${TRUTH}" "${WEAK_ARGS[@]}" \
        --output "${SNAP}" --split-manifest "${SPLIT}" \
        --association-csv data/crossmatch/lsst_to_ztf.csv \
        --seq-train-ids models/seq_classifier_v11/fold_map.json \
        --lsst-live-locked "${LOCKED}" \
        --dp1 --dp1-output "${DP1_SNAP}" "${SMOKE_ARGS[@]}"
else
    echo "$(ts) [skip] gold exists: ${SNAP}"
fi

# 5. helpfulness
if [[ "${FORCE}" == "1" || ! -f "${HELP}" ]]; then
    python3 -u scripts/build_helpfulness_fusion.py --snapshots "${SNAP}" --output "${HELP}"
fi

# 6. train. Models are written before the guards run; a guard failure is reported, and scoring/eval still run
#    so the failure can be inspected (the stack is not deployable until it is resolved).
TRAIN_RC=0
python3 -u scripts/train_fusion_v11.py --n-jobs "${NSLOTS}" \
    --snapshots "${SNAP}" --helpfulness "${HELP}" --split "${SPLIT}" \
    --truth "${TRUTH_V11}" --lsst-locked "${LOCKED}" \
    --output-snapshots "${SNAP_TRUST}" \
    --trust-dir "${TRUST}" --followup-dir "${FOLLOW}" --blend-dir "${BLEND}" --conformal-dir "${CONF}" \
    --fdr-gamma 0.1 --fdr-n-det-max 5 \
    --acknowledge-g2-unevaluable "${ACK_ARGS[@]}" \
    --build-report "reports/metrics/fusion_${SFX}_build.json" \
    --metrics-out "reports/metrics/fusion_${SFX}_train.json" || TRAIN_RC=$?
echo "$(ts) train exit code ${TRAIN_RC}"
test -f "${FOLLOW}/model.pkl" || { echo "no follow-up model written — stopping"; exit 3; }

# 7. score (gold + DP1) and eval
python3 -u scripts/score_fusion_v11.py --tag "fusion_${SFX}" --snapshots "${SNAP_TRUST}" \
    --split "${SPLIT}" --fdr-gamma 0.1 --fdr-n-det-max 5 \
    --trust-dir "${TRUST}" --followup-dir "${FOLLOW}" --blend-dir "${BLEND}" \
    --conformal "${CONF}/mondrian_aps.pkl"
python3 -u scripts/score_fusion_v11.py --tag "fusion_${SFX}" --dp1 --snapshots "${DP1_SNAP}" \
    --split "${SPLIT}" \
    --trust-dir "${TRUST}" --followup-dir "${FOLLOW}" --blend-dir "${BLEND}" \
    --conformal "${CONF}/mondrian_aps.pkl"
python3 -u scripts/eval_fusion_v8.py \
    --pred "data/scores/predictions_fusion_${SFX}.parquet" \
    --pred-dp1 "data/scores/predictions_fusion_${SFX}_dp1.parquet" \
    --snapshots "${SNAP_TRUST}" --split "${SPLIT}" \
    --train-metrics "reports/metrics/fusion_${SFX}_train.json" \
    --out-dir "reports/fusion_${SFX}"

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
exit "${TRAIN_RC}"
