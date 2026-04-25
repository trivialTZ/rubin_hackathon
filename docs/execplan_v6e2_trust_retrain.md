# Exec plan — v6e.2: Retrain trust + followup with LSST-aware local experts

**Status:** Drafted 2026-04-25 (after v6e Tier A analysis pass).
**Driver:** v6e DP1 production showed `ensemble_n_trusted_experts` is
always 1 — the v5d trust head ignores salt3 even though salt3 is now
populated on 25.7% of DP1 snapshots. Investigation revealed the root
cause is upstream of the trust head.

## Why this plan exists

The v5c gold (`data/gold/object_epoch_snapshots_safe_v5c.parquet`,
211,392 snapshots / 12,772 objects) has a **structural blind spot**:
all 4 local experts are 0% available on the 72,331 LSST training rows.

| Local expert | ZTF avail | LSST avail |
|---|---|---|
| alerce_lc      | 100.0% (139,061) | **0.0% (0/72,331)** |
| supernnova     |  80.3% (111,597) | **0.0% (0/72,331)** |
| salt3_chi2     |  99.5% (138,351) | **0.0% (0/72,331)** |
| lc_features_bv |  99.2% (137,898) | **0.0% (0/72,331)** |

The v5d trust heads for each of these 4 experts were therefore trained
on **ZTF-only data**. When v6e activates these experts on LSST DP1
snapshots, the trust heads have no LSST training signal to draw on, so
they give them low weight regardless of how reliable the predictions
actually are.

This was masked in the v6e Tier A run because:
- The v5d **followup head** was trained on a v5c trust-scored gold
  where these experts had 0% LSST avail. It learned a v5c-shaped
  prior. The followup score (`p_follow_proxy`) is downstream of the
  trust-weighted ensemble and is robust to expert-availability shifts.
- The v6e analysis showed `p_follow_proxy` is unchanged within CI95.
- But the underlying `ensemble_p_snia` shifts on the 24% of v6e
  snapshots where salt3 is now active — these shifts are net-negative
  on classes salt3 doesn't disambiguate (Gaia variables, EclBin),
  because the trust head can't down-weight bad salt3 evidence on LSST.

To get the multi-expert ensemble actually working on LSST, the trust
heads must train on real LSST-active data. That requires rebuilding
gold from scratch with v6e source code, then retraining helpfulness,
trust, and followup downstream.

## Scope (v6e.2)

**In scope:**
- Rebuild gold with v6e source on v5c labels (`data/labels.csv`).
- Re-apply v5b/v5c truth patches (LSST-truth circularity fix).
- Rebuild expert helpfulness.
- Retrain all trust heads (auto-discovered; same 11 experts as v5d).
- Train followup head with `--weak-weight 0.25` (matches v5d).
- Rescore DP1 v6e snapshots with the new heads.
- Compare paper-grade to v6e baseline.

**Out of scope:**
- ORACLE wiring into the training-set ingest pipeline. (v6e.3 task.)
- ParSNIP wrapper. (v6f task.)
- New broker queries.
- New labels / truth-source changes.

## Expected wall-clock on SCC

| Step | Time | Notes |
|---|---|---|
| Gold rebuild (8-way array)  | 30-60 min | salt3 + SNN + alerce_lc + lc_features_bv on 12,772 objects |
| Apply v5b/v5c truth patches | < 2 min   | Same scripts as v5c retrain |
| Helpfulness rebuild         | 2-5 min   | Single-threaded read+write |
| Trust + followup retrain    | 8-12 min  | 11 experts × LightGBM, n_jobs=4 |
| DP1 rescore                 | ~1 min    | Reuses v6e snapshots |
| **Total**                   | ~50-90 min| Mostly gold-rebuild dominated |

## Strategy: sharded SGE array (mirrors v6e infra)

`build_object_epoch_snapshots.py` is single-threaded by design.
Adding `--shard-id N --n-shards K` (this commit) selects every Kth
object_id from `data/labels.csv`, allowing 8-way parallelism via SGE
array.

After all shards complete:
1. Merge with `scripts/merge_gold_shards.py` (dedup on (object_id, n_det))
2. Apply v5b/v5c truth patches in sequence
3. Run helpfulness → trust → followup pipeline
4. Rescore DP1

## File-by-file plan

### Code changes (this commit)

| Path | Change |
|---|---|
| `scripts/build_object_epoch_snapshots.py` | Add `--shard-id`/`--n-shards` |
| `scripts/merge_dp1_snapshot_shards.py` | Auto-detect `object_id` vs `diaObjectId` for dedup |
| `jobs/run_v6e2_gold_array.sh` | New SGE array job (8 × 1-core × 16G × 6h h_rt) |
| `jobs/run_v6e2_pipeline.sh` | New post-merge pipeline (truth patches → helpfulness → trust → followup → rescore) |
| `docs/execplan_v6e2_trust_retrain.md` | This file |

### Output artifacts on SCC

```
data/gold/
  object_epoch_snapshots_v6e2.shard-{0..7}-of-8.parquet  # array shard outputs
  object_epoch_snapshots_v6e2_raw.parquet                # merged
  object_epoch_snapshots_safe_v6e2.parquet               # after v5b/v5c truth patches
  expert_helpfulness_safe_v6e2.parquet
  object_epoch_snapshots_trust_safe_v6e2.parquet
models/
  trust_safe_v6e2/<expert>/model.pkl + metadata.json     # 11 trust heads
  followup_safe_v6e2/{model.pkl, metadata.json}
reports/
  metrics/expert_trust_metrics_safe_v6e2.json
  metrics/followup_metrics_safe_v6e2.json
  v6e2_dp1_50k/{predictions.parquet, metrics.json,
                v6e_vs_v6e2_comparison.md, ...analysis pass...}
```

v5c, v5d, v6e artifacts are preserved unchanged.

## Pass / fail criteria

| Criterion | Target |
|---|---|
| Gold rebuild has all 4 local experts > 50% avail on LSST | LSST salt3 ≥ 50%, SNN ≥ 50%, alerce_lc ≥ 50%, lc_features_bv ≥ 50% |
| Trust head AUC for salt3 (calibrated) | Within 0.05 of v5d salt3 AUC (no regression) |
| `ensemble_n_trusted_experts` distribution shifts to higher modes on DP1 | Mode > 1 on at least 25% of snapshots |
| `p_follow_proxy` Gaia rejection EF doesn't regress vs v6e | Within CI95 |
| `p_follow_proxy` published-SN recall doesn't regress vs v6e | Within CI95 |
| `ensemble_p_snia` EclBin EF improves vs v6e | < 8 (was 11.72 in v6e) |

## Rollback plan

If any pass criterion fails, revert to v6e: predictions and trust heads
remain at `reports/v6e_dp1_50k/` and `models/trust_safe_v5d/`. v6e2
artifacts are isolated under `*_v6e2` suffixes.

## Out-of-scope follow-ons

- **v6e.3 (ORACLE in training)**: wire `oracle_lsst` into
  `build_object_epoch_snapshots.py` (currently only in DP1 driver),
  install ORACLE on SCC (`--no-deps + transformers`), rebuild gold
  with ORACLE column populated, retrain trust head for `oracle_lsst`.
- **v6f (ParSNIP)**: debug `ParSNIPExpert` segfault.
