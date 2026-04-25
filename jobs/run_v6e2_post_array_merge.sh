#!/bin/bash
#$ -N debass_v6e2_merge
#$ -l h_rt=01:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v6e2_merge.qsub.out
#
# v6e.2 — merge salt3 shard outputs into main silver, then run
# lc_features_bv on FULL labels.csv to restore lost ZTF rows + add LSST.

set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) [1/3] merge salt3 shards"
$PY scripts/_merge_silver_v6e2.py \
    --shard-glob 'data/silver_v6e2_salt3_shard_*/local_expert_outputs/salt3_chi2/part-latest.parquet' \
    --target-silver data/silver \
    --expert salt3_chi2

echo
echo "$(ts) [2/3] lc_features_bv on FULL labels (12,772 = 8,774 ZTF + 3,998 LSST)"
$PY -u scripts/local_infer.py \
    --expert lc_features_bv \
    --from-labels data/labels.csv \
    --lc-dir data/lightcurves \
    --silver-dir data/silver \
    --max-n-det 20

echo
echo "$(ts) [3/3] final silver counts"
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
