#!/bin/bash
#$ -N debass_v6e2_infer
#$ -l h_rt=03:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v6e2_infer.array.$TASK_ID.out
#
# v6e.2 — populate silver/local_expert_outputs/{salt3_chi2,alerce_lc,
# lc_features_bv,supernnova}/ with LSST training rows.
#
# Submit:
#   qsub -t 1-4 jobs/run_v6e2_local_infer_lsst_array.sh
#
# Pre-reqs:
#   - data/labels_lsst.csv exists (created by this script if missing)
#   - v6e source code (registry includes salt3 + lc_features_bv)
#
# Each task runs one expert sequentially over all LSST training objects.
# local_infer.py uses _merge_records to upsert, so existing ZTF rows in
# silver are untouched.

set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python

# Generate labels_lsst.csv on demand. Numeric IDs only (LSST diaObjectId).
if [[ ! -f data/labels_lsst.csv ]]; then
    echo "=== generating data/labels_lsst.csv ==="
    $PY - <<'PYEOF'
import pandas as pd
d = pd.read_csv("data/labels.csv")
oids = d["object_id"].astype(str)
lsst_mask = oids.str.match(r"^[0-9]{10,}$")
out = d.loc[lsst_mask].copy()
out.to_csv("data/labels_lsst.csv", index=False)
print(f"wrote data/labels_lsst.csv with {len(out):,} rows")
PYEOF
fi

# Map SGE_TASK_ID → expert name
EXPERTS=(salt3_chi2 alerce_lc lc_features_bv supernnova)
EXPERT="${EXPERTS[$((SGE_TASK_ID - 1))]}"

echo "=== v6e2 local-infer expert=$EXPERT (task ${SGE_TASK_ID}/4) @ $(date) on $(hostname) ==="
$PY -u scripts/local_infer.py \
    --expert "$EXPERT" \
    --from-labels data/labels_lsst.csv \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --max-n-det 20

echo "=== task ${SGE_TASK_ID} ($EXPERT) done @ $(date) ==="
