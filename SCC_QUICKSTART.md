# SCC Deployment Quick Start

**Status**: ✓ Infrastructure validated and ready for deployment

## Prerequisites

- BU SCC account with project allocation
- TNS API credentials (for bulk CSV download)
- Lasair API token (for broker backfill)

## Step 1: Copy Repository to SCC

```bash
# From your local machine
scp -r ~/Documents/GitHub/rubin_hackathon <username>@scc1.bu.edu:/projectnb/<project>/
```

## Step 2: Set Up Python Environment

```bash
# SSH to SCC
ssh <username>@scc1.bu.edu

# Navigate to project
cd /projectnb/<project>/rubin_hackathon

# Create virtual environment
python3 -m venv ~/debass_meta_env
source ~/debass_meta_env/bin/activate

# Install dependencies
pip install -r env/requirements.txt

# Verify installation
python3 scripts/validate_infrastructure.py
```

## Step 3: Configure Credentials

```bash
# Copy template
cp .env.example .env

# Edit with your credentials
nano .env
```

Add:
```bash
# TNS (for bulk CSV download)
TNS_API_KEY=your_key_here
TNS_TNS_ID=your_id_here
TNS_MARKER_NAME=your_name_here

# Lasair (for broker backfill)
LASAIR_TOKEN=your_token_here
```

## Step 4: Download TNS Bulk CSV (One-Time)

```bash
# Download ~160K objects, ~50MB
python3 scripts/download_tns_bulk.py --output-dir data

# Verify
ls -lh data/tns_public_objects.csv
wc -l data/tns_public_objects.csv
```

## Step 5: Run Smoke Test (20 Objects)

```bash
# Full pre-ML pipeline on 20 objects
python3 scripts/run_preml_pipeline.py \
    --root tmp/scc_smoke_test \
    --labels data/labels.csv \
    --lightcurves-dir data/lightcurves \
    --skip-local-infer \
    --require-trust-training-ready

# Check readiness
cat tmp/scc_smoke_test/reports/summary/preml_readiness.json | grep ready_for_trust_training_now
```

Expected output: `"ready_for_trust_training_now": true`

## Step 6: Train ML Models (Smoke Test)

```bash
# Train trust heads (7 experts)
python3 scripts/train_expert_trust.py \
    --snapshots tmp/scc_smoke_test/gold/object_epoch_snapshots.parquet \
    --helpfulness tmp/scc_smoke_test/gold/expert_helpfulness.parquet \
    --models-dir tmp/scc_smoke_test/models/trust

# Train follow-up head
python3 scripts/train_followup.py \
    --snapshots tmp/scc_smoke_test/gold/object_epoch_snapshots_trust.parquet \
    --truth tmp/scc_smoke_test/truth/object_truth.parquet \
    --models-dir tmp/scc_smoke_test/models/followup

# Score epochs
python3 scripts/score_nightly.py \
    --from-labels data/labels.csv \
    --n-det 4 \
    --lc-dir data/lightcurves \
    --silver-dir tmp/scc_smoke_test/silver \
    --truth tmp/scc_smoke_test/truth/object_truth.parquet \
    --trust-dir tmp/scc_smoke_test/models/trust \
    --followup-dir tmp/scc_smoke_test/models/followup \
    --output-dir tmp/scc_smoke_test/reports/scores
```

## Step 7: Scale to 2000 Objects

```bash
# Set environment variable
export DEBASS_ROOT=/projectnb/<project>/rubin_hackathon

# Submit CPU prep job (SGE array)
bash jobs/submit_cpu_prep.sh --limit 2000

# Monitor progress
python3 scripts/watch_cpu_prep.py

# After completion, train models
python3 scripts/train_expert_trust.py \
    --snapshots data/gold/object_epoch_snapshots.parquet \
    --helpfulness data/gold/expert_helpfulness.parquet \
    --models-dir models/trust

python3 scripts/train_followup.py \
    --snapshots data/gold/object_epoch_snapshots_trust.parquet \
    --truth data/truth/object_truth.parquet \
    --models-dir models/followup
```

## Troubleshooting

### Issue: TNS bulk CSV download fails
**Solution**: Verify TNS credentials in `.env`, check network connectivity

### Issue: Broker backfill fails
**Solution**: Check Lasair API token, verify broker client versions

### Issue: Pre-ML pipeline fails
**Solution**: Run `python3 scripts/audit_data_readiness.py --root tmp/scc_smoke_test --labels data/labels.csv --lightcurves-dir data/lightcurves`

### Issue: Trust training fails
**Solution**: Check `tmp/scc_smoke_test/reports/summary/preml_readiness.json` for blockers

## Expected Timeline

- **Environment setup**: 15 minutes
- **TNS bulk CSV download**: 5 minutes
- **Smoke test (20 objects)**: 10-15 minutes
- **ML training (smoke test)**: 5 minutes
- **Full run (2000 objects)**: 2-4 hours (with SGE array jobs)

## Success Criteria

After smoke test:
- ✓ `tmp/scc_smoke_test/gold/object_epoch_snapshots.parquet` exists
- ✓ `tmp/scc_smoke_test/gold/expert_helpfulness.parquet` exists
- ✓ `tmp/scc_smoke_test/models/trust/<expert>/model.pkl` exists (7 experts)
- ✓ `tmp/scc_smoke_test/models/followup/model.pkl` exists
- ✓ `tmp/scc_smoke_test/reports/scores/*.jsonl` exists

## Key Files Generated

**Pre-ML outputs**:
- `data/bronze/*.parquet` — raw broker payloads
- `data/silver/broker_events.parquet` — normalized expert events
- `data/truth/object_truth.parquet` — TNS crossmatch truth
- `data/gold/object_epoch_snapshots.parquet` — ML training features
- `data/gold/expert_helpfulness.parquet` — ML training targets

**ML outputs**:
- `models/trust/<expert>/model.pkl` — trust head per expert
- `models/followup/model.pkl` — follow-up head
- `reports/metrics/expert_trust_metrics.json` — trust head metrics
- `reports/metrics/followup_metrics.json` — follow-up metrics
- `reports/scores/*.jsonl` — expert_confidence payloads

## Contact

For issues, see `troubleshooting.md` or check:
- `docs/scc_deployment_verification.md` — complete infrastructure analysis
- `docs/ml_data_infrastructure_analysis.md` — scientific intent and data flow
- `CLAUDE.md` — project context and design decisions
