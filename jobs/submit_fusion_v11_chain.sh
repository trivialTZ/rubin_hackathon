#!/bin/bash
# Submit the fusion_v11 chain on SCC:
#
#   bash jobs/submit_fusion_v11_chain.sh [HOLD_JOB_ID]
#
# Chain (ORDER IS THE POINT — B0/B2 land first, P6 integrates last):
#   build (CPU, 16-core): TNS bulk → P1 rederive object_truth_v11 → FULL v11
#     gold (positives-only epochs + neg features) + FINAL split_fusion_v11.json
#     → G5b (locked ZTF gold value-identity, right after the gold build)
#     → pretrain (GPU, gpu_c=8.0): seq_v11 SSL encoder + OOF classifier
#         (--seq-schema v11), BOTH against that manifest (no chain-ordering cal
#         leak: training never precedes the manifest it obeys)
#         → expert (CPU, 16-core): seq_v11 OOF silver → gold rebuild WITH
#           seq_v11 → Stage A/B (hierarchical heads) → anchored blend →
#           post-blend conformal → guards/gates → score → eval → locked
#           LSST-live benchmark re-score (v11 + v10 on cleaned truth)
#
# If HOLD_JOB_ID is given and still running, the build holds on it first.
# Idempotent resubmission: each job skips completed artifacts
# (FUSION_V11_FORCE=1 rebuilds everything). Enable the ELAsTiCC2 pretrain corpus
# with FUSION_V11_ELASTICC2=1 (exported into the pretrain job's environment).
set -euo pipefail
cd /project/pi-brout/rubin_hackathon
mkdir -p logs

EXT_JID="${1:-}"
HOLD_BUILD=()
if [[ -n "${EXT_JID}" ]] && qstat -j "${EXT_JID}" >/dev/null 2>&1; then
    HOLD_BUILD=(-hold_jid "${EXT_JID}")
    echo "build will hold on job ${EXT_JID}"
fi

BUILD_OUT=$(qsub -terse "${HOLD_BUILD[@]}" -P pi-brout jobs/run_fusion_v11_build.sh)
BUILD_JID="${BUILD_OUT%%.*}"
echo "submitted v11 build (rederive truth_v11 + gold + FINAL split + G5b): ${BUILD_JID}"

PRE_OUT=$(qsub -terse -hold_jid "${BUILD_JID}" -P pi-brout jobs/run_fusion_v11_pretrain.sh)
PRE_JID="${PRE_OUT%%.*}"
echo "submitted v11 pretrain (seq_v11 SSL + OOF classifier, GPU): ${PRE_JID} (holds: ${BUILD_JID})"

EXPERT_OUT=$(qsub -terse -hold_jid "${PRE_JID}" -P pi-brout jobs/run_fusion_v11_expert.sh)
echo "submitted v11 expert-integration (OOF silver → gold → heads → blend → conformal → eval): ${EXPERT_OUT%%.*} (holds: ${PRE_JID})"
qstat -u "$(whoami)" | head -14
