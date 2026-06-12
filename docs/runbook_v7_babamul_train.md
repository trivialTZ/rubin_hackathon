# Runbook — v7 Babamul retrain on SCC

Phases 1-3 verified locally on 2026-04-26 (`scripts/_exp_babamul_cohort_pull.py`,
`scripts/_exp_babamul_probe.py`). Phases 4-5 below run on SCC, against the
existing v6e.2 baseline, and stay frozen-baseline-safe by writing to a `_v7`
suffix.

## Pre-flight

1. **Sync code** — push the four touched files to SCC:
   - `src/debass_meta/access/babamul.py` (new)
   - `src/debass_meta/access/__init__.py` (registered BabamulAdapter)
   - `src/debass_meta/projectors/babamul.py` (new)
   - `src/debass_meta/projectors/base.py` (added `babamul` to EXPERT_REGISTRY + dispatch)
   - `src/debass_meta/ingest/silver.py` (default exactness/scope tables)
   - `scripts/_exp_babamul_cohort_pull.py` (new — used as the SCC backfill driver)
   - `docs/execplan_v7_more_brokers.md` (Tier 2D updated)

2. **Set Babamul creds on SCC**. The local `.env` already has them; copy or
   re-run the read-and-write block from the local CLI session. Verify with:

```bash
ssh scc
cd /project/pi-brout/rubin_hackathon
PY=/usr3/graduate/tztang/debass_meta_env/bin/python
$PY scripts/_exp_babamul_probe.py | tail -20    # all 6 steps PASS
```

## Phase 4a — bulk pull on the existing ZTF cohort (~1 min wall)

```bash
$PY scripts/_exp_babamul_cohort_pull.py --bronze-dir data/bronze --n-threads 24
# expect: ~74% hit rate on 1,920 ZTF labels
```

This writes one new bronze parquet under `data/bronze/babamul_<ts>.parquet`
alongside the existing fink/alerce/lasair files. Bronze is append-only so
this never disturbs the v6e.2 frozen pipeline.

## Phase 4b — silver rebuild (~1 min wall)

```bash
$PY scripts/normalize.py --bronze-dir data/bronze --silver-dir data/silver
# adds ~8.6K rows (1,427 hits × 6 features) to silver/broker_events.parquet
```

Verify:

```bash
$PY -c "
import pandas as pd
s = pd.read_parquet('data/silver/broker_events.parquet')
print('total silver rows:', len(s))
print('babamul rows:', (s['expert_key']=='babamul').sum())
print('babamul fields:', s[s['expert_key']=='babamul']['field'].unique())
"
```

## Phase 4c — gold snapshot rebuild, sharded 8-way (~7-8 min wall)

Use the same SGE array pattern as v6e.2 to avoid the silver-race bug:

```bash
qsub jobs/run_v6e_dp1_build_score.sh    # already shards 8-way
# or directly:
for i in 0 1 2 3 4 5 6 7; do
    $PY scripts/build_object_epoch_snapshots.py \
        --shard-id $i --n-shards 8 \
        --silver-dir data/silver \
        --output data/gold/object_epoch_snapshots_safe_v7.shard-$i-of-8.parquet \
        --truth data/truth/object_truth.parquet \
        --objects-csv data/labels.csv &
done
wait

# Concat shards
$PY -c "
import pandas as pd, glob
df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob('data/gold/object_epoch_snapshots_safe_v7.shard-*'))])
df.to_parquet('data/gold/object_epoch_snapshots_safe_v7.parquet', index=False)
print('rows:', len(df), 'avail__babamul rate:', df['avail__babamul'].mean())
"
```

Expected: ~211K rows (matches v6e.2), `avail__babamul.mean() ≈ 0.74` on the
1,920 ZTF cohort, `0.0` on the 3,998 LSST cohort (DP1 not backfilled by Babamul).

## Phase 4d — followup head retrain ONLY (no new trust head)

Critical architecture decision (see `project_babamul_verification.md`):
**Babamul is a feature on the followup head, NOT a trust head.** Skip
`train_expert_trust.py` for `babamul`.

```bash
$PY scripts/train_followup.py \
    --gold data/gold/object_epoch_snapshots_safe_v7.parquet \
    --trust-dir models/trust_safe_v6e2 \
    --out-dir models/followup_safe_v7 \
    --include-broker-features babamul     # NEW flag if not present; otherwise
                                          # the gold columns starting with
                                          # proj__babamul__ are auto-detected
```

If `train_followup.py` doesn't have `--include-broker-features`, the simpler
path is: the followup head already auto-discovers `proj__*` columns by
scanning the gold parquet. Verify by checking the model's feature_importance
includes `proj__babamul__babamul_star_flag` and friends.

## Phase 4e — score DP1 (sanity check) and ZTF test split

```bash
$PY scripts/score_dp1_v5d.py \
    --snapshots data/gold/dp1_snapshots_50k.parquet \
    --trust-dir models/trust_safe_v6e2 \
    --followup-dir models/followup_safe_v7 \
    --out-dir reports/v7_dp1_50k
```

Note: DP1 snapshots have `avail__babamul=0` everywhere (live-stream-only).
The DP1 numbers should be **statistically identical** to v6e.2 on DP1 — that
is the **honest test that v7 doesn't regress** when Babamul is unavailable.
If DP1 numbers move, the followup head learned to depend on Babamul in a way
that breaks when it's missing — investigate.

The real v7 win lives in the ZTF test-split metrics (where Babamul has
74% coverage). Compare:

```bash
# v6e.2 baseline
cat reports/metrics/followup_metrics_safe_v6e2.json | jq .test
# v7 with Babamul
cat models/followup_safe_v7/metrics.json | jq .test
```

Pass criterion (per `docs/execplan_v7_more_brokers.md` Tier 2D):
- At least one of the 6 `proj__babamul__*` features has feature_importance ≥ 1%
- AND DP1 numbers (Gaia-rej, EclBin/RRLyrae EF) within ± 0.5 of v6e.2

## Phase 5 — paper paragraph

When v7 metrics land, add to `paper/metaDEBASS_aas/sections/discussion.tex`:

> **Multi-broker fusion: Babamul integration.** We integrate the Babamul
> broker (Caltech/UMN) as a context expert, exposing four alert-time
> categorical flags (``star``, ``near\_brightstar``, ``rock``,
> ``stationary``) computed by Babamul's BOOM backend. On the 1,920-object
> ZTF training cohort, Babamul covers 74.3\% of objects after retro-ingest,
> and the ``star`` flag alone separates SN-confirmed sources (0.3\% star=1)
> from the heterogeneous ``other'' class (68\% star=1) with effectively
> zero false-positive rate. Babamul does not currently retro-index DP1
> commissioning data, so the DP1 demonstration metrics in
> Section~\ref{subsec:results-dp1} are unchanged; we report Babamul-aware
> followup-head metrics on the ZTF test split as forward-looking evidence
> that broker-level dual-survey alert routing strengthens the prioritisation
> ranker once Rubin live alerts begin flowing.

## Files NOT to touch

- `models/trust_safe_v6e2/` — frozen (no new trust head for Babamul)
- `models/followup_safe_v6e2/` — frozen
- `data/gold/object_epoch_snapshots_safe_v6e2.parquet` — frozen
- `reports/v6e2_dp1_50k/` — frozen DP1 baseline

All v7 outputs use the `_v7` suffix.
