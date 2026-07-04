#!/bin/bash
# Submit the fusion_v10 chain on SCC:
#
#   bash jobs/submit_fusion_v10_chain.sh [HOLD_JOB_ID]
#
# Chain (ORDER IS THE POINT — the v10 P0 fix):
#   build (CPU, 16-core): gold pre-table + the FINAL split_fusion_v10.json
#     → pretrain (GPU, gpu_c=8.0): SSL + OOF classifier, BOTH against that
#       manifest (no chain-ordering cal leak: training never precedes the
#       manifest it obeys)
#       → expert (CPU, 16-core): OOF seq_v9 silver → full gold rebuild →
#         Stage A/B → gates (incl. the seq_v9 re-gate vs v9c) → reports/fusion_v10
#
# If HOLD_JOB_ID is given and still running, the build holds on it first.
# Idempotent resubmission: each job skips completed artifacts
# (FUSION_V10_FORCE=1 rebuilds everything).
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
mkdir -p logs

EXT_JID="${1:-}"
HOLD_BUILD=()
if [[ -n "${EXT_JID}" ]] && qstat -j "${EXT_JID}" >/dev/null 2>&1; then
    HOLD_BUILD=(-hold_jid "${EXT_JID}")
    echo "build will hold on job ${EXT_JID}"
fi

BUILD_OUT=$(qsub -terse "${HOLD_BUILD[@]}" -P pi-brout jobs/run_fusion_v10_build.sh)
BUILD_JID="${BUILD_OUT%%.*}"
echo "submitted v10 build (gold pre-table + FINAL split manifest): ${BUILD_JID}"

PRE_OUT=$(qsub -terse -hold_jid "${BUILD_JID}" -P pi-brout jobs/run_fusion_v10_pretrain.sh)
PRE_JID="${PRE_OUT%%.*}"
echo "submitted v10 pretrain (SSL + OOF classifier + staged ZTF→LSST arm, GPU): ${PRE_JID} (holds: ${BUILD_JID})"

EXPERT_OUT=$(qsub -terse -hold_jid "${PRE_JID}" -P pi-brout jobs/run_fusion_v10_expert.sh)
echo "submitted v10 expert-integration (OOF silver → gold → gates → eval): ${EXPERT_OUT%%.*} (holds: ${PRE_JID})"
qstat -u "$(whoami)" | head -14
