#!/bin/bash
#$ -N debass_v6e2_seq
#$ -l h_rt=04:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v6e2_infer_seq.qsub.out
#
# v6e.2 — sequential local_infer for salt3_chi2 (LSST) + lc_features_bv (full).
# REPLACES the parallel array (4631742) which raced on local_expert_outputs.json
# and lost ZTF rows for alerce_lc + lc_features_bv. The race-loss was salvaged
# by scripts/_repair_silver_v6e2.py (alerce_lc rebuilt from part-allobj backup).
# This job restores the lost lc_features_bv ZTF rows and adds the missing salt3
# LSST rows. Sequential = no race.
set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) v6e.2 sequential local_infer start"

echo "$(ts) [1/2] salt3_chi2 on LSST training labels (3,998 objects)"
$PY -u scripts/local_infer.py \
    --expert salt3_chi2 \
    --from-labels data/labels_lsst.csv \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --max-n-det 20

echo
echo "$(ts) [2/2] lc_features_bv on FULL labels (12,772 objects: 8,774 ZTF + 3,998 LSST)"
$PY -u scripts/local_infer.py \
    --expert lc_features_bv \
    --from-labels data/labels.csv \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --max-n-det 20

echo
echo "$(ts) v6e.2 sequential local_infer DONE"
echo
echo "=== final silver state ==="
$PY - <<'PYEOF'
import pandas as pd
for ek in ("salt3_chi2", "alerce_lc", "supernnova", "lc_features_bv"):
    p = f"data/silver/local_expert_outputs/{ek}/part-latest.parquet"
    df = pd.read_parquet(p)
    oids = df["object_id"].astype(str)
    z = int(oids.str.startswith("ZTF").sum())
    l = int(oids.str.match(r"^[0-9]{10,}$").sum())
    print(f"  {ek}: {len(df):,} ({z:,} ZTF, {l:,} LSST)")
PYEOF
