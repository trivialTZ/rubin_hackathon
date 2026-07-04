#!/bin/bash
#$ -N debass_v10_expert
#$ -cwd -V
#$ -l h_rt=10:00:00
#$ -l mem_per_core=8G
#$ -pe omp 16
#$ -o logs/fusion_v10_expert.qsub.out
#$ -e logs/fusion_v10_expert.qsub.err
# metaDEBASS fusion_v10 — the seq_v9 expert integrated with HONEST train-row
# projections, end to end:
#
#   1. local_infer seq_v9 over every labeled object → silver per-epoch scores
#      with K-FOLD OOF ROUTING: fold_map.json sends each train object to the
#      fold model that never saw it (the v10 fix for the v9c stacking trap).
#      Single sequential process — never run concurrent local_infer against
#      one silver dir (v6e.2 race postmortem).  The v9c silver for seq_v9 is
#      in-sample on train rows, so it is ALWAYS overwritten on the first v10
#      run (idempotency via a v10 marker file, not silver existence).
#   2. FULL gold rebuild _v10 (experts + traj + DP1) against the SAME
#      split_fusion_v10.json the classifier was trained on, with the
#      seq-train guard armed (fold_map.json): any drift that would put a
#      classifier-seen object into cal is diverted to train, locked-test
#      hits hard-fail.
#   3. helpfulness → Stage A retrains (seq_v9 trust head on OOF projections)
#      → component gates → Stage B → score → eval (reports/fusion_v10).
#   4. Prints the seq_v9_expert gate verdict NEXT TO the v9c verdict — the
#      v9c gate (-0.0009, CI spanning 0) was uninterpretable because train
#      rows were in-sample and part of cal was contaminated; this rerun is
#      the fair test.
#
# Locked test stays byte-identical; all new artifacts use the _v10 suffix.
# Idempotent: completed artifacts are reused (FUSION_V10_FORCE=1 rebuilds).
# Submit via jobs/submit_fusion_v10_chain.sh (holds on the GPU pretrain job).
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
NSLOTS=${NSLOTS:-16}
mkdir -p logs data/scores reports/fusion_v10 reports/metrics

SNAP="data/gold/object_epoch_snapshots_fusion_v10.parquet"
SPLIT="data/gold/split_fusion_v10.json"
HELP="data/gold/expert_helpfulness_fusion_v10.parquet"
SNAP_TRUST="${SNAP%.parquet}_trust.parquet"
DP1_SNAP="data/gold/dp1_snapshots_fusion_v10.parquet"
OOF_MARKER="data/silver/local_expert_outputs/seq_v9/.fusion_v10_oof_done"
FORCE="${FUSION_V10_FORCE:-0}"

# Pin the expert to the v10 OOF artifact (fold routing lives in the expert).
export DEBASS_SEQ_V9_MODEL="models/seq_classifier_v10"

echo "$(ts) v10 expert — START (NSLOTS=${NSLOTS})"
test -f "${SPLIT}" || { echo "missing ${SPLIT} (v10 build not finished?)"; exit 2; }
test -f models/seq_classifier_v10/classifier.pt || { echo "missing models/seq_classifier_v10"; exit 2; }
test -f models/seq_classifier_v10/fold_map.json || {
    echo "missing models/seq_classifier_v10/fold_map.json — v10 REQUIRES OOF routing"; exit 2; }

# 1. seq_v9 OOF inference into silver (sequential). Marker-based idempotency:
#    pre-v10 seq_v9 silver is in-sample on train rows and must be replaced.
if [[ "${FORCE}" == "1" || ! -f "${OOF_MARKER}" ]]; then
    python3 -u scripts/local_infer.py --expert seq_v9 \
        --from-labels data/labels.csv \
        --lc-dir data/lightcurves --silver-dir data/silver --max-n-det 20
    touch "${OOF_MARKER}"
else
    echo "$(ts) [skip] v10 OOF seq_v9 silver exists (marker ${OOF_MARKER})"
fi

# 2. FULL gold + DP1 rebuild (same manifest; seq-train guard armed)
if [[ "${FORCE}" == "1" || ! -f "${SNAP}" ]]; then
    python3 -u scripts/build_snapshots_fusion.py --n-jobs "${NSLOTS}" \
        --output "${SNAP}" --split-manifest "${SPLIT}" \
        --association-csv data/crossmatch/lsst_to_ztf.csv \
        --seq-train-ids models/seq_classifier_v10/fold_map.json \
        --dp1 --dp1-output "${DP1_SNAP}"
else
    echo "$(ts) [skip] gold snapshot exists: ${SNAP}"
fi

# 3. helpfulness
if [[ "${FORCE}" == "1" || ! -f "${HELP}" ]]; then
    python3 -u scripts/build_helpfulness_fusion.py \
        --snapshots "${SNAP}" --output "${HELP}"
else
    echo "$(ts) [skip] helpfulness exists: ${HELP}"
fi

# 4. full train (Stage A retrains — seq_v9 trust head now fits on OOF
#    train-row projections; gates decide every component on clean cal)
python3 -u scripts/train_fusion_v8.py --n-jobs "${NSLOTS}" \
    --snapshots "${SNAP}" --helpfulness "${HELP}" --split "${SPLIT}" \
    --output-snapshots "${SNAP_TRUST}" \
    --trust-dir models/trust_fusion_v10 \
    --followup-dir models/followup_fusion_v10 \
    --conformal-dir models/conformal_fusion_v10 \
    --metrics-out reports/metrics/fusion_v10_train.json

# 5. score (gold + DP1; FDR-controlled selection wired in v10).
#    --fdr-n-det-max 5: fit the FDR thresholds in the 3-5-detection regime
#    the spectroscopic-TAC selection gates in deployment — a tau fit at the
#    cal objects' final (n_det~20, well-separated) epochs does not transfer.
python3 -u scripts/score_fusion_v8.py --tag fusion_v10 \
    --snapshots "${SNAP_TRUST}" \
    --split "${SPLIT}" --fdr-gamma 0.1 --fdr-n-det-max 5 \
    --trust-dir models/trust_fusion_v10 \
    --followup-dir models/followup_fusion_v10 \
    --conformal models/conformal_fusion_v10/mondrian_aps.pkl
python3 -u scripts/score_fusion_v8.py --tag fusion_v10 --dp1 \
    --snapshots "${DP1_SNAP}" \
    --split "${SPLIT}" \
    --trust-dir models/trust_fusion_v10 \
    --followup-dir models/followup_fusion_v10 \
    --conformal models/conformal_fusion_v10/mondrian_aps.pkl

# 6. eval
python3 -u scripts/eval_fusion_v8.py \
    --pred data/scores/predictions_fusion_v10.parquet \
    --pred-dp1 data/scores/predictions_fusion_v10_dp1.parquet \
    --snapshots "${SNAP_TRUST}" --split "${SPLIT}" \
    --train-metrics reports/metrics/fusion_v10_train.json \
    --out-dir reports/fusion_v10

echo "$(ts) v10 expert — DONE"
python3 - <<'EOF'
import json, pathlib

def gate(path, component="seq_v9_expert"):
    p = pathlib.Path(path)
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    for e in payload.get("component_gates", payload.get("gates", [])) or []:
        if isinstance(e, dict) and e.get("component") == component:
            return e
    return None

def fmt(e):
    if e is None:
        return "not found"
    d = e.get("delta_macro_auc")
    ci = e.get("delta_macro_auc_ci95")
    d_s = f"{d:+.4f}" if isinstance(d, (int, float)) else str(d)
    ci_s = f" CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}]" if isinstance(ci, list) and len(ci) == 2 else ""
    return f"{e.get('decision', '?').upper()} (d_macro_auc {d_s}{ci_s})"

v10 = gate("reports/metrics/fusion_v10_train.json")
v9c = gate("reports/metrics/fusion_v9c_train.json") or gate(
    "reports_from_scc/fusion_v9c/fusion_v9c_train.json")
print("SEQ_V9 GATE COMPARISON — v9c (in-sample train rows, contaminated cal) "
      "vs v10 (OOF train rows, ordered chain):")
print(f"  v9c: {fmt(v9c)}")
print(f"  v10: {fmt(v10)}")

hg = pathlib.Path("reports/fusion_v10/headline_guards.json")
if hg.exists():
    payload = json.loads(hg.read_text())
    print("V10 HEADLINE:", json.dumps(payload.get("headline", {}), indent=2)[:900])
tr = pathlib.Path("reports/metrics/fusion_v10_train.json")
if tr.exists():
    payload = json.loads(tr.read_text())
    for e in payload.get("component_gates", payload.get("gates", [])) or []:
        if isinstance(e, dict):
            print("GATE:", e.get("component"), "→", e.get("decision"), e.get("delta_macro_auc"))
    sa = payload.get("stage_a", {})
    if isinstance(sa, dict) and "seq_v9" in sa:
        print("seq_v9 trust:", json.dumps(sa["seq_v9"])[:300])
EOF
