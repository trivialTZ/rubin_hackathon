# DEBASS Light-up Runbook — exact SCC commands

**Companion to `plan_lightup_campaign.md`.** Every phase has copy-pasteable commands. Run from `/project/pi-brout/rubin_hackathon/` with `source .venv/bin/activate` active.

---

## Preamble — SSH + env

```bash
# SSH (ControlMaster 8h persist)
ssh scc

# On SCC
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate

# Environment check
df -h . | head -5                              # >150 GB free on /project
python3 -c "import pandas, lightgbm, sncosmo"  # smoke test deps
```

Expected free space post-campaign: ~3–4 GB used (up from 2.5 GB).

---

## F0 — Register AMPEL + pre-flight

### Code change (local first, then push to SCC)

File: `src/debass_meta/access/__init__.py`

Find `ALL_ADAPTERS = [...]`, add `AmpelAdapter`:

```python
from .ampel import AmpelAdapter  # new import

ALL_ADAPTERS = [
    AlerceAdapter(),
    FinkAdapter(),
    LasairAdapter(),
    PittGoogleAdapter(),
    AntaresAdapter(),
    AmpelAdapter(),   # new
]
```

### Push to SCC + verify

```bash
# Local
git add src/debass_meta/access/__init__.py
git commit -m "feat: register AMPEL adapter in ALL_ADAPTERS"

# Or rsync if not committing
rsync -avz src/debass_meta/access/ scc:/project/pi-brout/rubin_hackathon/src/debass_meta/access/

# On SCC — dry-run
python3 -c "
from debass_meta.access import ALL_ADAPTERS
print('Adapters:', [type(a).__name__ for a in ALL_ADAPTERS])
"
# Expect 6 adapters including AmpelAdapter

# Single-object reachability test for each new broker
python3 -c "
from debass_meta.access.antares import AntaresAdapter
a = AntaresAdapter()
events = a.fetch_object('ZTF21abfmbix')
print(f'ANTARES events: {len(events)}')
"
# Repeat for Ampel, Pitt-Google UPSILoN
```

### Exit criteria

- ALL_ADAPTERS has 6 entries
- Each adapter responds to at least one test query (0 events is OK — means API alive, just no match)

---

## F1 — Backfill 5 idle remote brokers

Run as background SGE jobs — each broker's backfill is independent.

### Launch backfill jobs (3 parallel)

```bash
# From SCC login node
cd /project/pi-brout/rubin_hackathon
mkdir -p logs/backfill

# ANTARES
nohup python3 scripts/backfill.py \
    --broker antares \
    --labels data/labels.csv \
    --parallel 4 \
    --output-dir data/bronze \
    > logs/backfill/antares_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "antares PID: $!"

# AMPEL — needs AMPEL_ARCHIVE_TOKEN in .env
nohup python3 scripts/backfill.py \
    --broker ampel \
    --labels data/labels.csv \
    --parallel 4 \
    --output-dir data/bronze \
    > logs/backfill/ampel_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "ampel PID: $!"

# Pitt-Google (includes UPSILoN via pitt.py _fetch_upsilon_lsst)
nohup python3 scripts/backfill.py \
    --broker pitt_google \
    --labels data/labels.csv \
    --parallel 4 \
    --output-dir data/bronze \
    > logs/backfill/pitt_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "pitt PID: $!"

# Monitor
tail -f logs/backfill/*.log
```

### Expected output

- `data/bronze/antares_*.parquet` (~50–200 MB — coverage low due to indexing window)
- `data/bronze/ampel_*.parquet` (~100–300 MB)
- `data/bronze/pitt_google_*.parquet` (~200–500 MB — includes SuperNNova and UPSILoN fields)

### Completion check

```bash
python3 << 'PY'
from pathlib import Path
import pandas as pd
for broker in ['antares', 'ampel', 'pitt_google']:
    files = list(Path('data/bronze').glob(f'{broker}_*.parquet'))
    total_rows = sum(pd.read_parquet(f).shape[0] for f in files)
    print(f'{broker}: {len(files)} files, {total_rows:,} rows')
PY
```

---

## F2 — Silver + gold rebuild

```bash
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate

# 1. Normalize bronze → silver (auto-discovers new brokers)
python3 scripts/normalize.py \
    --bronze-dir data/bronze \
    --output data/silver/broker_events.parquet \
    2>&1 | tee logs/normalize_$(date +%Y%m%d).log

# Verify new expert keys showed up
python3 -c "
import pandas as pd
df = pd.read_parquet('data/silver/broker_events.parquet')
vc = df.groupby(['broker','classifier']).size().sort_values(ascending=False)
print(vc.to_string())
"

# 2. Collect per-epoch local expert history (existing: supernnova, alerce_lc)
python3 scripts/collect_epoch_history.py \
    --labels data/labels.csv \
    --lightcurves-dir data/lightcurves \
    --output-dir data/silver/local_expert_outputs \
    2>&1 | tee logs/collect_local_$(date +%Y%m%d).log

# 3. Build gold snapshot (attaches ALL_EXPERT_KEYS projections)
python3 scripts/build_object_epoch_snapshots.py \
    --silver-dir data/silver \
    --truth data/truth/object_truth.parquet \
    --lightcurves-dir data/lightcurves \
    --output data/gold/object_epoch_snapshots_trust.parquet \
    2>&1 | tee logs/gold_$(date +%Y%m%d).log
```

### Completion check

```bash
python3 << 'PY'
import pandas as pd
df = pd.read_parquet('data/gold/object_epoch_snapshots_trust.parquet')
proj_cols = sorted({c.rsplit('__', 1)[0].replace('proj__', '') for c in df.columns if c.endswith('__p_snia')})
print(f'Experts with projections in gold: {len(proj_cols)}')
for e in proj_cols:
    n = df[f'proj__{e}__p_snia'].notna().sum()
    print(f'  {e}: {n:,} rows non-null')
PY
```

Target: ≥ 15 experts with > 100 non-null rows each.

---

## F3 — Train trust heads + followup

```bash
# 1. Rebuild helpfulness table
python3 scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --output data/gold/expert_helpfulness.parquet \
    2>&1 | tee logs/helpfulness_$(date +%Y%m%d).log

# 2. Train per-expert trust heads
python3 scripts/train_expert_trust.py \
    --helpfulness data/gold/expert_helpfulness.parquet \
    --output-dir models/trust \
    --min-rows 300 \
    --metrics-out reports/metrics/expert_trust_metrics.json \
    2>&1 | tee logs/train_trust_$(date +%Y%m%d).log

# If --min-rows flag not supported, it's still auto-discovery —
# check train_expert_trust.py for filter logic and add if missing.

# 3. Retrain followup head
python3 scripts/train_followup.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --output-dir models/followup \
    --metrics-out reports/metrics/followup_metrics.json \
    2>&1 | tee logs/train_followup_$(date +%Y%m%d).log
```

### Completion check

```bash
ls -d models/trust/*/     # Count how many heads
python3 -c "
import json
m = json.load(open('reports/metrics/expert_trust_metrics.json'))
for k,v in sorted(m.items()):
    raw = v.get('raw', {}).get('roc_auc', 'N/A')
    cal = v.get('calibrated', {}).get('roc_auc', 'N/A')
    print(f'{k:50s}  raw_auc={raw}  cal_auc={cal}')
"
```

---

## F4 — SALT3 + Bazin/Villar batch runs

### Write runners (code task — local repo)

**File: `scripts/run_salt3_batch.py`** — follow `collect_epoch_history.py` pattern:

```python
# Pseudocode outline:
# 1. Load labels + iterate objects
# 2. For each object: load lightcurve, run SALT3 Ia fit + Nugent-IIP fit
# 3. Compute delta_chi2, write row with {object_id, expert='salt3_chi2',
#    model_name, n_det, alert_mjd, class_probabilities=JSON(dummy),
#    raw_output={delta_chi2, chi2_ia, chi2_iip}, available, temporal_exactness='rerun_exact'}
# 4. Write to data/silver/local_expert_outputs/salt3_chi2/part-latest.parquet
```

**File: `scripts/run_lc_features_batch.py`** — similar, uses `LcFeaturesExpert.predict_epoch()` which already exists.

### Run on SCC as array job

```bash
# Bazin/Villar — fast, run single-threaded
nohup python3 scripts/run_lc_features_batch.py \
    --labels data/labels.csv \
    --lightcurves-dir data/lightcurves \
    --output-dir data/silver/local_expert_outputs/lc_features_bv \
    > logs/lc_features_$(date +%Y%m%d).log 2>&1 &

# SALT3 — slow, array job with 20 workers
# First: scripts/run_salt3_batch.py must support --shard N/M
for i in {0..19}; do
    qsub -N salt3_shard_$i -cwd -V \
        -o logs/salt3_shard_$i.out -e logs/salt3_shard_$i.err \
        -b y -l h_rt=12:00:00 \
        "source .venv/bin/activate && python3 scripts/run_salt3_batch.py --shard $i/20 --labels data/labels.csv --lightcurves-dir data/lightcurves --output-dir data/silver/local_expert_outputs/salt3_chi2_shards"
done

# After all shards complete
python3 scripts/run_salt3_batch.py --merge-shards data/silver/local_expert_outputs/salt3_chi2_shards --output data/silver/local_expert_outputs/salt3_chi2/part-latest.parquet
```

### Then re-run F2 + F3

```bash
python3 scripts/build_object_epoch_snapshots.py ...   # rebuild gold with new local experts
python3 scripts/build_expert_helpfulness.py ...
python3 scripts/train_expert_trust.py ...             # trains salt3_chi2 + lc_features_bv heads
```

---

## F5 — ORACLE local inference wiring

### Code task

File: `src/debass_meta/experts/local/oracle_local.py` lines 104–154 (the `predict_epoch` method).

Replace the placeholder block:

```python
# REPLACE THIS BLOCK (lines 144-153):
try:
    out.available = False
    out.raw_output["reason"] = "inference wiring TODO after repo clone"
```

With actual inference:

```python
try:
    import pandas as pd
    # Build SNANA-style dataframe expected by ORACLE
    snana_rows = []
    for det in truncated:
        snana_rows.append({
            "MJD": float(det.get("mjd") or (det.get("jd") - 2400000.5)),
            "BAND": _band_for_oracle(det),  # maps ZTF g/r → LSST g/r
            "FLUXCAL": float(det.get("magpsf", det.get("mag", 0))),
            "FLUXCALERR": float(det.get("sigmapsf", det.get("magerr", 0))),
        })
    lc_df = pd.DataFrame(snana_rows)

    # Lazy-instantiate model on first call
    if self._model is None:
        self._model = self._ORACLE1_ELAsTiCC.load_from_checkpoint(str(self._weights_pth))
        self._model.eval()

    probs = self._model.classify(lc_df)  # dict: {class_name: prob}
    out.class_probabilities = {k: float(v) for k, v in probs.items()}
    out.raw_output["raw_probs_json"] = json.dumps(probs)
    out.available = True
except Exception as exc:
    out.available = False
    out.raw_output["reason"] = f"inference error: {exc}"
```

### Add to `collect_epoch_history.py` dispatch

Find the expert-registration block and add:

```python
try:
    from debass_meta.experts.local.oracle_local import OracleLocalExpert
    experts["oracle_lsst"] = OracleLocalExpert()
except Exception as e:
    print(f"[warn] oracle_lsst unavailable: {e}")
```

### Test single-object then batch

```bash
python3 -c "
from debass_meta.experts.local.oracle_local import OracleLocalExpert
import json
e = OracleLocalExpert()
print('available:', e._available, 'reason:', e._reason)
lc = json.load(open('data/lightcurves/ZTF21abnujlh.json'))
out = e.predict_epoch('ZTF21abnujlh', lc, epoch_jd=2460000)
print(out)
"
```

Then `scripts/collect_epoch_history.py` picks it up automatically.

---

## F6 — Analyze + figures

```bash
python3 scripts/analyze_results.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --trust-models-dir models/trust \
    --followup-model models/followup/model.pkl \
    --output-dir reports/metrics \
    2>&1 | tee logs/analyze_$(date +%Y%m%d).log

python3 scripts/make_paper_figures.py \
    --latency reports/metrics/latency_curves.parquet \
    --trust-metrics reports/metrics/expert_trust_metrics.json \
    --out-dir paper/debass_aas/figures \
    2>&1 | tee logs/figures_$(date +%Y%m%d).log
```

### New figures to add

Write `scripts/plot_ablation.py` — sweep top-K in ensemble, plot AUC vs K.
Write `scripts/plot_per_class_confusion.py` — 3x3 heatmap at n_det = 3,5,10,15.

---

## F7 — Cross-val + honesty pass

### 5-fold grouped CV

Edit `src/debass_meta/models/expert_trust.py` to add `cv_train_expert_trust()` alongside `train_expert_trust()`:

```python
from sklearn.model_selection import GroupKFold

def cv_train_expert_trust(df, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    for train_idx, test_idx in gkf.split(df, groups=df['object_id']):
        ...  # train LightGBM on train_idx, eval on test_idx
        fold_metrics.append({...})
    return {
        'mean_auc': np.mean([m['auc'] for m in fold_metrics]),
        'std_auc': np.std([m['auc'] for m in fold_metrics]),
        'folds': fold_metrics,
    }
```

Add `--cv` flag to `scripts/train_expert_trust.py`.

### Per-class confusion + failure modes

- `scripts/plot_per_class_confusion.py` — 3x3 confusion using argmax of trust-weighted probabilities at each n_det.
- `scripts/failure_modes.py` — extract misclassified objects at n_det=10, join with host context + redshift + discovery_broker, write CSV to `reports/failure_modes.csv`.

---

## F8 — ELAsTiCC2 (deferred)

Only if paper timeline allows.

```bash
# Selective DUMP download
mkdir -p data/truth/elasticc2
wget -r -l1 -np -A "DUMP*.csv.gz" \
    https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ \
    -P data/truth/elasticc2

# Then extend build_truth_multisource.py with --elasticc2 path
# Retrain LSST-exclusive heads on combined ZTF + simulated truth
```

---

## Monitoring helpers

```bash
# Watch disk usage during campaign
watch -n 60 'df -h /project | head -5; du -sh data/bronze data/silver data/gold'

# Watch SGE queue
qstat -u $USER

# Tail all logs
tail -f logs/*.log
```

---

## Rollback

All phases write to new files; nothing destructive. To reset:

```bash
# Roll trust models back
rsync -avz _backup_pre_lightup/models/ models/

# Roll gold back
rsync -avz _backup_pre_lightup/data/gold/ data/gold/
```

Take the backup **before F2**:

```bash
cp -r data/gold _backup_pre_lightup/data/gold
cp -r models _backup_pre_lightup/models
cp -r reports _backup_pre_lightup/reports
```

---

**Last updated**: 2026-04-20 initial draft. Mark completion timestamps as each phase finishes.
