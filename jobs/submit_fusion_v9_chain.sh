#!/bin/bash
# Submit the fusion_v9 chain on SCC, chained onto an already-running (or
# completed) v8 pipeline job:
#
#   bash jobs/submit_fusion_v9_chain.sh [V8_JOB_ID]
#
# Chain: [v8 pipeline (yours)] → GPU SSL pretrain (l40s) → v9 arm (CPU).
# If V8_JOB_ID is omitted or already finished, the pretrain starts immediately
# (it only needs data/gold/split_fusion_v8.json + lightcurves).
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
mkdir -p logs

V8_JID="${1:-}"
HOLD_PRETRAIN=()
if [[ -n "${V8_JID}" ]] && qstat -j "${V8_JID}" >/dev/null 2>&1; then
    HOLD_PRETRAIN=(-hold_jid "${V8_JID}")
    echo "pretrain will hold on v8 job ${V8_JID}"
fi

PRE_OUT=$(qsub -terse "${HOLD_PRETRAIN[@]}" -P pi-brout jobs/run_fusion_v9_pretrain.sh)
PRE_JID="${PRE_OUT%%.*}"
echo "submitted pretrain (SSL + classifier fine-tune, GPU unpinned): ${PRE_JID}"

HOLD_ARM=(-hold_jid "${PRE_JID}")
if [[ -n "${V8_JID}" ]] && qstat -j "${V8_JID}" >/dev/null 2>&1; then
    HOLD_ARM=(-hold_jid "${V8_JID},${PRE_JID}")
fi
ARM_OUT=$(qsub -terse "${HOLD_ARM[@]}" -P pi-brout jobs/run_fusion_v9_arm.sh)
ARM_JID="${ARM_OUT%%.*}"
echo "submitted v9 arm (frozen-embedding gate re-test): ${ARM_JID}"

# v9c: seq classifier as a registered expert — full pipeline rerun with
# seq_v9 silver + LSST weak labels.  Serialized after the arm job to avoid
# concurrent heavy CPU stages and any silver write overlap.
V9C_OUT=$(qsub -terse -hold_jid "${ARM_JID}" -P pi-brout jobs/run_fusion_v9c_expert.sh)
echo "submitted v9c expert-integration: ${V9C_OUT%%.*} (holds: ${ARM_JID})"
qstat -u "$(whoami)" | head -14
