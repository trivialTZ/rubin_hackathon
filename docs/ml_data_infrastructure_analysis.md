# ML Data Infrastructure Analysis — DEBASS Phase-1

**Date**: 2026-03-31
**Purpose**: Verify data input infrastructure is complete for ML training on BU SCC

## Scientific Intent Summary

**Primary Goal**: Model which expert/classifier is trustworthy at early epochs (3-5 detections)

**Primary Output**: `expert_confidence` — calibrated per-expert trust at `(object_id, n_det, alert_jd)`

**Secondary Output**: `p_follow_proxy` — trust-weighted follow-up score

**Phase-1 Experts** (7 total):
- `fink/snn` — alert-level SNN posterior (exact)
- `fink/rf_ia` — alert-level RF Ia score (exact)
- `parsnip` — local rerun posterior (exact when weights exist)
- `supernnova` — local rerun posterior (exact when weights exist)
- `alerce/lc_classifier_transient` — object snapshot (live-only, unsafe historical)
- `alerce/stamp_classifier` — object snapshot (live-only, unsafe historical)
- `lasair/sherlock` — context expert (static-safe)

## ML Model Architecture

All models use **LightGBM** gradient-boosted trees (Ke et al. 2017, NeurIPS).
Chosen over RandomForest for:
- **Native NaN handling** — Fink scores are ~100% NaN on most ZTF objects;
  LightGBM learns optimal split direction for missing values (no imputation).
- **Leaf-wise growth** — captures broker-score x lightcurve interactions
  more effectively than depth-wise RF.
- **Early stopping** — prevents overfitting on small training sets.

Shared hyperparameters: `learning_rate=0.05, num_leaves=31, min_child_samples=5,
feature_fraction=0.8, bagging_fraction=0.8, reg_alpha=0.1, reg_lambda=0.1`.

### Trust Heads (per expert)
**Algorithm**: LightGBM binary classifier (`is_unbalance=True`, n_estimators=500)
**Input**: `gold/object_epoch_snapshots.parquet`
- Lightcurve features at n_det (no future leakage)
- Expert evidence available at that epoch
- Temporal exactness flags

**Target**: `gold/expert_helpfulness.parquet`
- Binary: was expert correct/helpful at this epoch?
- Derived from truth + expert prediction

**Training**: GroupKFold (5-fold, grouped by object_id) for OOF predictions.
Deploy model trained on train+cal data.

**Output**: `models/trust/<expert>/model.pkl` + metadata

### Follow-up Head
**Algorithm**: LightGBM binary classifier (`is_unbalance=True`, n_estimators=500)
**Input**:
- `gold/object_epoch_snapshots.parquet`
- Out-of-fold trust predictions from trust heads

**Target**: `truth/object_truth.parquet`
- `follow_proxy = 1{final_class_ternary == 'snia'}`

**Training**: Trains on train+cal objects, evaluates on test.

**Output**: `models/followup/model.pkl`

### Baseline (EarlyMetaClassifier)
**Algorithm**: LightGBM multiclass (3-class, n_estimators=1000 with early stopping)
**Calibration**: Temperature scaling on held-out cal set; metrics on separate test set.
**Feature importances**: Logged and saved to metrics JSON.

## Data Pipeline — Pre-ML Phase

### Step 1: Lightcurves
**Script**: `scripts/download_alerce_training.py`
**Output**: `data/lightcurves/*.json`
**Status**: ✓ Working
**SCC Ready**: Yes (CPU, no credentials needed for cached LCs)

### Step 2: Broker Payloads → Bronze
**Script**: `scripts/backfill.py`
**Output**: `data/bronze/*.parquet`
**Inputs Required**:
- Lightcurves (from Step 1)
- Broker API credentials (Fink, ALeRCE, Lasair)

**Status by Broker**:
- **Fink**: ✓ Adapter exists (`src/debass_meta/access/fink.py`)
- **ALeRCE**: ✓ Adapter exists (`src/debass_meta/access/alerce.py`)
- **Lasair**: ✓ Adapter exists (`src/debass_meta/access/lasair.py`)

**SCC Ready**: Needs verification
- Fink client requires `fink-client>=7.0`
- ALeRCE client requires `alerce>=2.3.0`
- Lasair client requires `lasair>=0.1.0`
- All in requirements.txt ✓

### Step 3: Bronze → Silver Events
**Script**: `scripts/normalize.py`
**Output**: `data/silver/broker_events.parquet`
**Purpose**: Event-level broker outputs with temporal exactness metadata

**Status**: ✓ Working (v2 architecture)
**SCC Ready**: Yes (CPU only, no external deps)

### Step 4: TNS Truth Crossmatch
**Script**: `scripts/crossmatch_tns.py` (NEW: bulk CSV mode)
**Output**: `data/truth/object_truth.parquet`

**Two Modes**:
1. **API mode** (slow): 2 calls/object, rate limited
2. **Bulk CSV mode** (fast): local crossmatch, no API calls

**Status**: ✓ Just implemented bulk CSV mode
**SCC Ready**: Yes
- Download CSV once: `scripts/download_tns_bulk.py` (requires TNS credentials)
- Crossmatch locally: `--bulk-csv data/tns_public_objects.csv`
- Verified 100% accurate on 14-object test set

### Step 5: Local Expert Inference
**Script**: `scripts/local_infer.py`
**Output**: `data/silver/local_expert_outputs/*.parquet`
**Purpose**: Rerun ParSNIP/SuperNNova on truncated lightcurves (no future leakage)

**Status by Expert**:
- **ParSNIP**: Wrapper exists (`src/debass_meta/experts/local/parsnip.py`)
  - Requires: `astro-parsnip>=1.4.0` (optional in requirements.txt)
  - Requires: Model weights in `artifacts/local_experts/parsnip/`
  - **SCC Ready**: Needs weight files + GPU (optional)

- **SuperNNova**: Wrapper exists (`src/debass_meta/experts/local/supernnova.py`)
  - Requires: `supernnova>=2.0` (optional in requirements.txt)
  - Requires: Model weights in `artifacts/local_experts/supernnova/`
  - **SCC Ready**: Needs weight files + GPU (optional)

**Critical**: Local experts are optional for Phase-1 trust training. Can proceed with broker experts only.

### Step 6: Object-Epoch Snapshots (Gold)
**Script**: `scripts/build_object_epoch_snapshots.py`
**Output**: `data/gold/object_epoch_snapshots.parquet`
**Purpose**: As-of join — features + expert evidence at each (object, n_det) epoch

**Inputs**:
- Lightcurves (for feature extraction)
- Silver broker events (for expert evidence)
- Truth table (for join)

**Status**: ✓ Working (v2 architecture with temporal exactness)
**SCC Ready**: Yes (CPU only)

### Step 7: Expert Helpfulness Targets
**Script**: `scripts/build_expert_helpfulness.py`
**Output**: `data/gold/expert_helpfulness.parquet`
**Purpose**: Binary targets — was expert correct at this epoch?

**Inputs**:
- Gold snapshots (from Step 6)
- Truth table (for ground truth)

**Status**: ✓ Working
**SCC Ready**: Yes (CPU only)

## Critical Dependencies for SCC

### Python Environment
```bash
python3 -m venv ~/debass_meta_env
source ~/debass_meta_env/bin/activate
pip install -r env/requirements.txt
```

**Core ML Stack**:
- ✓ pandas>=2.0
- ✓ pyarrow>=15.0
- ✓ numpy>=1.26
- ✓ scikit-learn>=1.4
- ✓ lightgbm>=4.0
- ✓ scipy>=1.12

**Broker Clients**:
- ✓ alerce>=2.3.0
- ✓ fink-client>=7.0
- ✓ lasair>=0.1.0
- ✓ requests>=2.31

**Validation**:
- ✓ pydantic>=2.0
- ✓ jsonschema>=4.0

### External Credentials Required

**For Broker Backfill** (Step 2):
- Fink: API token (if needed)
- ALeRCE: No auth required for public API
- Lasair: API token

**For TNS Truth** (Step 4):
- TNS_API_KEY
- TNS_TNS_ID
- TNS_MARKER_NAME
- Only needed once to download bulk CSV

**For Local Experts** (Step 5, optional):
- Model weight files (not in repo, need to download separately)

## Data Flow Verification Checklist

### Pre-ML Pipeline (CPU)
- [x] Download lightcurves from ALeRCE
- [x] Backfill broker payloads (Fink, ALeRCE, Lasair)
- [x] Normalize to silver events
- [x] TNS crossmatch (bulk CSV mode)
- [ ] Local expert inference (optional, needs weights)
- [x] Build gold snapshots (as-of joins)
- [x] Build expert helpfulness targets
- [x] Audit data readiness
- [x] Check benchmarks

### ML Training (CPU)
- [ ] Train trust heads (one per expert)
- [ ] Train follow-up head
- [ ] Score epochs with expert_confidence

### Critical Gaps to Address

**High Priority**:
1. **Broker API credentials on SCC** — verify Fink/Lasair tokens work
2. **TNS bulk CSV download** — run once on SCC with credentials
3. **Test full pre-ML pipeline** — run `scripts/run_preml_pipeline.py` on 20 objects

**Medium Priority**:
4. **Local expert weights** — download ParSNIP/SuperNNova weights if GPU available
5. **GPU access on SCC** — verify GPU nodes available for local inference

**Low Priority**:
6. **Benchmark tuning** — adjust `benchmarks/preml_data_engineering.toml` as needed

## Recommended SCC Verification Steps

### Step 1: Environment Setup
```bash
# On SCC login node
cd /projectnb/<yourproject>/rubin_hackathon
python3 -m venv ~/debass_meta_env
source ~/debass_meta_env/bin/activate
pip install -r env/requirements.txt
```

### Step 2: Credentials Setup
```bash
# Copy .env.example to .env and fill in:
# - TNS credentials (for bulk CSV download)
# - Broker API tokens (Fink, Lasair)
cp .env.example .env
nano .env
```

### Step 3: Download TNS Bulk CSV (one-time)
```bash
python3 scripts/download_tns_bulk.py --output-dir data
# Creates: data/tns_public_objects.csv (~160K objects)
```

### Step 4: Test Pre-ML Pipeline (20 objects)
```bash
python3 scripts/run_preml_pipeline.py \
    --root tmp/scc_test \
    --labels data/labels.csv \
    --lightcurves-dir data/lightcurves \
    --skip-local-infer \
    --require-trust-training-ready
```

This will:
- Backfill broker payloads
- Normalize to silver
- Crossmatch TNS (using bulk CSV)
- Build gold snapshots
- Build expert helpfulness
- Validate readiness

**Expected output**: `tmp/scc_test/reports/summary/preml_readiness.json`
- `ready_for_trust_training_now: true`

### Step 5: Test ML Training (smoke test)
```bash
python3 scripts/train_expert_trust.py \
    --snapshots tmp/scc_test/gold/object_epoch_snapshots.parquet \
    --helpfulness tmp/scc_test/gold/expert_helpfulness.parquet \
    --output-dir tmp/scc_test/models/trust

python3 scripts/train_followup.py \
    --snapshots tmp/scc_test/gold/object_epoch_snapshots.parquet \
    --truth tmp/scc_test/truth/object_truth.parquet \
    --trust-dir tmp/scc_test/models/trust \
    --output-dir tmp/scc_test/models/followup
```

## Known Limitations

1. **Fink coverage**: Sparse for ZTF objects (most scores are NaN)
2. **ALeRCE historical snapshots**: Unsafe for trust training (marked in gold)
3. **Local experts**: Require GPU + weight files (optional for Phase-1)
4. **19-object smoke test**: Overfits (F1=1.0) — need 2000+ for real metrics

## Success Criteria for SCC Deployment

**Minimum Viable** (trust training ready):
- ✓ Broker payloads fetched for all 3 brokers
- ✓ TNS truth table built (bulk CSV mode)
- ✓ Gold snapshots built with temporal exactness
- ✓ Expert helpfulness targets built
- ✓ At least 3 experts have sufficient coverage (>50% of epochs)

**Full Phase-1 Goal**:
- All 7 experts have coverage
- Local experts (ParSNIP, SuperNNova) running on GPU
- 2000+ objects with spectroscopic truth
- Trust heads trained and calibrated
- Follow-up head trained
- Scoring pipeline validated

## Next Actions

1. **Verify broker adapters on SCC** — test API calls work
2. **Download TNS bulk CSV** — run `scripts/download_tns_bulk.py` once
3. **Run pre-ML pipeline** — 20-object smoke test on SCC
4. **Verify ML training** — train trust heads on smoke test data
5. **Scale to 2000 objects** — full training run with SGE array jobs
