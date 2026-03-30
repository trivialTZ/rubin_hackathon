# DEBASS on BU SCC — Setup & Run Guide

## 1. One-time environment setup

```bash
# SSH into SCC
ssh scc1.bu.edu

# Load Python module
module load python3/3.11.4

# Create virtualenv in your home directory
python3 -m venv ~/debass_env
source ~/debass_env/bin/activate

# Clone / copy the repo
cd /projectnb/<yourproject>
git clone <repo-url> rubin_hackathon
cd rubin_hackathon

# Install dependencies
pip install -r env/requirements.txt

# Set the repo root env var (add to ~/.bashrc for persistence)
export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon
```

## 2. Credentials

```bash
# Copy the example env file and fill in your Lasair token
cp .env.example .env
nano .env   # set LASAIR_TOKEN=<your token>
```

ALeRCE and Fink require no credentials — they use public REST APIs.

## 3. Create log directory

```bash
mkdir -p $DEBASS_ROOT/logs
```

## 4. Submit the full early-epoch pipeline (one command)

```bash
cd $DEBASS_ROOT

# Submit everything — jobs chain automatically via hold_jid
bash jobs/submit_all.sh

# With GPU local experts (SuperNNova / ParSNIP on SCC GPU node):
bash jobs/submit_all.sh --gpu

# Smaller run for testing:
bash jobs/submit_all.sh --limit 200
```

This submits 4-5 chained jobs:
```
download_training  →  build_epochs  →  [local_infer_gpu]  →  train_early  →  score_all
```

To submit individual stages manually:
```bash
qsub jobs/download_training.sh   # fetch ALeRCE objects + lightcurves
qsub jobs/build_epochs.sh        # build (object_id, n_det) epoch table
qsub jobs/local_infer_gpu.sh     # GPU: run SuperNNova/ParSNIP (optional)
qsub jobs/train_early.sh         # train LightGBM meta-classifier
qsub jobs/score_all.sh           # score objects, write follow-up priorities
```

## 5. Monitor jobs

```bash
qstat -u $USER
qstat -j <job_id>        # detailed status
tail -f logs/train.<job_id>.log
```

## 6. Tuning job resources

Edit the `#$ -l` directives in `jobs/*.sh` to match your queue limits.
Key variables you can override at submission time:

```bash
export DEBASS_LIMIT=2000     # number of objects to backfill
export DEBASS_METHOD=lgbm    # meta-classifier method (lgbm or logistic)
export DEBASS_LABELS=/path/to/labels.csv  # ground-truth label CSV
```

## 7. Output locations

| Stage | Output |
|-------|--------|
| backfill | `data/bronze/<broker>_<ts>.parquet` |
| normalize | `data/silver/broker_outputs.parquet`, `data/gold/snapshots.parquet` |
| local_infer | `data/silver/local_expert_outputs.json` |
| train | `models/baselines/*.pkl`, `models/meta/` |
| report | `reports/metrics/evaluation.json` |

## 8. Pipeline architecture summary

```
Remote brokers          Local experts
(ALeRCE, Fink,          (SuperNNova,
 Lasair)                 ParSNIP, AMPEL)
     |                        |
     v                        v
  Bronze layer  ──────────────┤
  (raw Parquet)               |
     |                        |
     v                        |
  Silver layer ───────────────┘
  (canonical rows, raw_label_or_score preserved)
     |
     v
  Gold layer
  (object-by-epoch snapshots + availability mask)
     |
     v
  Trust heads  (per-broker q_b(t))
     |
     v
  Meta-classifier  (ternary: SN Ia / non-Ia SN / other)
     |
     v
  Calibration  (temperature scaling)
     |
     v
  reports/metrics/evaluation.json
```

## 9. Semantic type contract

The pipeline enforces that broker outputs are **never treated as
interchangeable probability vectors**:

| Broker | Field | Semantic type |
|--------|-------|---------------|
| ALeRCE | `probabilities` | `probability` — safe to use directly |
| Fink | `rf_snia_vs_nonia`, `snn_*` | `probability` |
| Fink | `finkclass` | `heuristic_class` — do NOT use as posterior |
| Lasair | `sherlock_classification` | `crossmatch_tag` — context only |
| SuperNNova | `prob_class0/1` | `probability` |
| ParSNIP | `class_probabilities` | `probability` |
