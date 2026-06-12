#!/bin/bash
#$ -N debass_v6e
#$ -l h_rt=12:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/v6e_dp1.qsub.out

# v6e DP1 build + score — Tier A
#
# Fixes salt3_chi2 LSST band-map bug (was 0% avail) + wires ORACLE local
# expert (Tier B dormant on SCC: no oracle_lsst trust head, no ORACLE pkg).
# Activates a 4th local expert (salt3_chi2 ≥ 60% avail expected).
#
# Pre-reqs (already done):
#   - git pull v6e source on SCC
#   - pip install --user iminuit
#
# Pipeline:
#   1) build v6e gold (~30-60 min for 41K with active salt3 fits)
#   2) re-score with v5d trust + followup (≈60 s)
set -euo pipefail
cd /project/pi-brout/rubin_hackathon

PY=/usr3/graduate/tztang/debass_meta_env/bin/python

echo "=== v6e DP1 build + score @ $(date) ==="
$PY -c "import iminuit, sncosmo; print('iminuit', iminuit.__version__, 'sncosmo', sncosmo.__version__)"

echo
echo "=== step 1: build v6e gold (full 41K) ==="
$PY scripts/build_dp1_snapshots.py \
    --truth data/truth/dp1_truth_50k.parquet \
    --out   data/gold/dp1_snapshots_50k_v6e.parquet

echo
echo "=== step 2: rescore with v5d trust + followup ==="
$PY scripts/score_dp1_v5d.py \
    --snapshots    data/gold/dp1_snapshots_50k_v6e.parquet \
    --trust-dir    models/trust_safe_v5d \
    --followup-dir models/followup_safe_v5d \
    --out-dir      reports/v6e_dp1_50k

echo
echo "=== step 3: quick summary ==="
$PY -c "
import pandas as pd
g = pd.read_parquet('data/gold/dp1_snapshots_50k_v6e.parquet')
p = pd.read_parquet('reports/v6e_dp1_50k/predictions.parquet')
print(f'snapshots: {len(g):,}')
print(f'predictions: {len(p):,}')
for k in ['salt3_chi2','oracle_lsst','alerce_lc','supernnova','lc_features_bv']:
    col = f'avail__{k}'
    if col in g.columns:
        n = int(g[col].fillna(0).astype(float).gt(0.5).sum())
        print(f'  {k}: avail = {n} / {len(g)} ({100*n/len(g):.1f}%)')
print('ensemble_n_trusted_experts distribution:')
print(p['ensemble_n_trusted_experts'].value_counts().sort_index())
"

echo
echo "=== v6e done @ $(date) ==="
