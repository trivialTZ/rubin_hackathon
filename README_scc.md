# DEBASS on BU SCC — Trust Pipeline Guide

## 1. One-time environment setup

```bash
# SSH into SCC
ssh scc1.bu.edu

# Pick one Python module version that actually exists on SCC
module avail python3
export DEBASS_PYTHON_MODULE=python3/3.10.12   # replace if your SCC module list differs
module load $DEBASS_PYTHON_MODULE

# Clone / copy the repo
# IMPORTANT: this repository is private, so unauthenticated `git pull`
# or `git clone` will fail on SCC. Configure either:
#   1. SSH access with a GitHub-authorized key:
#        git remote set-url origin git@github.com:<org>/<repo>.git
#        ssh -T git@github.com
#   2. HTTPS access with a GitHub personal access token:
#        git config --global credential.helper store
#        # then authenticate on the next git fetch/pull/clone prompt
#
# The tokens in `.env` are for Lasair/TNS data access only.
# They do NOT grant GitHub access to the private repo.
cd /projectnb/<yourproject>
git clone <repo-url> rubin_hackathon
cd rubin_hackathon

# Create the virtualenv in project space, not $HOME
python3 -m venv /projectnb/<yourproject>/venvs/debass_meta_env
source /projectnb/<yourproject>/venvs/debass_meta_env/bin/activate

# Install dependencies
pip install -r env/requirements.txt

# Optional but required for the GPU expert stage:
# install a GPU-enabled PyTorch build plus the expert packages you plan to use.
# BU recommends installing PyTorch into your own virtualenv rather than relying
# on old module builds. See the BU PyTorch page for the current pip command.
# Example expert installs:
pip install astro-parsnip
# supernnova requires Python 3.11+ on PyPI, so keep it separate from the
# default SCC Python 3.10 environment unless you plan a dedicated 3.11 GPU env.

# ParSNIP inference in this repo requires BOTH:
#   artifacts/local_experts/parsnip/model.pt
#   artifacts/local_experts/parsnip/classifier.pkl
# The model alone is not enough because typed class probabilities come from the
# trained ParSNIP LightGBM classifier.

# Set the repo root env var (add to ~/.bashrc for persistence)
export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon
export DEBASS_VENV=/projectnb/<yourproject>/venvs/debass_meta_env
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

## 4. Recommended workflow: Phase 1 (CPU-only)

The entire Phase 1 pipeline runs on CPU. SuperNNova uses Fink's pre-trained
bidirectional LSTM via `classify_lcs` (batch mode, ~10 min for 7K objects).
All trust/follow-up models use LightGBM. **No GPU queue required.**

### Step 1: Data preparation

```bash
cd $DEBASS_ROOT

# One-command bootstrap: verify checkout, load Python, create venv,
# install requirements, then submit CPU prep
bash -l jobs/scc_bootstrap.sh --limit 2000

# Or if environment is already set up:
bash jobs/submit_cpu_prep.sh --limit 2000
```

This runs:
```text
download_training → backfill (array) → normalize
  → crossmatch_tns → build_object_epoch_snapshots → build_expert_helpfulness
```

### Step 2: SuperNNova inference

After CPU prep completes:

```bash
# Patch SuperNNova for pandas 2.x compatibility (run once):
python3 scripts/patch_supernnova_pandas2.py

# Submit batch SNN inference (~10 min for 7K objects × 20 epochs):
qsub -V -P pi-brout jobs/local_infer_snn.sh
```

SuperNNova artifacts must be placed in `artifacts/local_experts/supernnova/`:
- `model.pt` -- state_dict from `fink-science/snn_snia_vs_nonia`
- `cli_args.json` -- training settings (bidirectional LSTM, 32 hidden, g+r filters)
- `data_norm.json` -- normalization constants

### Step 3: Train, score, and analyze

After SNN inference completes, submit the full pipeline chain with SGE
job dependencies (each step holds on the previous):

```bash
# Submit all 5 steps at once:
bash jobs/submit_phase1_chain.sh

# Or hold on a still-running SNN job:
bash jobs/submit_phase1_chain.sh --after-snn <SNN_JOB_ID>
```

The chain runs:
```text
rebuild_gold → train_expert_trust → train_followup → score (n_det=3,5,10) → analyze_results
```

### Step 4: Read results

```bash
# Human-readable summary of all metrics:
cat reports/results_summary.txt

# Test-set metrics (held-out 20% of objects):
cat reports/metrics/expert_trust_metrics.json   # per-expert trust ROC-AUC
cat reports/metrics/followup_metrics.json       # follow-up ROC-AUC

# Score payloads (JSONL, one line per object):
head -1 reports/scores/scores_ndet3.jsonl | python3 -m json.tool

# Split reproducibility — exact object IDs in each partition:
python3 -c "
import json
m = json.load(open('models/trust/metadata.json'))
print(f'Train: {len(m[\"train_ids\"])}  Cal: {len(m[\"cal_ids\"])}  Test: {len(m[\"test_ids\"])}')
"
```

### Phase 2: GPU experts (when weights available)

GPU is only needed for ParSNIP (no public weights yet). SuperNNova runs on CPU.

```bash
qrsh -l gpus=1 -q l40s
cd $DEBASS_ROOT
bash -l jobs/run_gpu_resume.sh --experts=parsnip
```

### Legacy single-submit flow

```bash
bash jobs/submit_all.sh --gpu --limit 2000
bash jobs/submit_all.sh --gpu --baseline --limit 2000  # baseline benchmark only
```

## 5. Manual stage control

Use `-V` so SCC jobs inherit `DEBASS_ROOT`, `DEBASS_PYTHON_MODULE`, and related env vars:

```bash
qsub -V jobs/download_training.sh      # weakly labelled seed objects + lightcurves
qsub -V jobs/backfill.sh               # array backfill by broker
qsub -V jobs/build_epochs.sh           # normalize + truth + gold prep
qsub -V jobs/local_infer_gpu.sh        # GPU: local experts only
qsub -V jobs/train_expert_trust.sh     # CPU trust-head training
qsub -V jobs/train_followup.sh         # CPU follow-up training
qsub -V jobs/score_all.sh              # score expert_confidence payloads
```

If `qsub` returns `Unable to run job: error: no suitable queues.`, the queue/resource
combination in the job header does not match any queue your account can use.
For CPU stages, first try a smaller shared-memory request and add your project
explicitly if needed:

```bash
qsub -V -P <project> -pe omp 1 -l mem_per_core=8G jobs/train_expert_trust.sh
qsub -V -P <project> -pe omp 1 -l mem_per_core=8G jobs/train_followup.sh
```

The submission wrappers also accept override flags through the environment, so
you do not need to edit `#$` lines for site-specific tuning:

```bash
export DEBASS_QSUB_COMMON_ARGS="-P <project>"
export DEBASS_QSUB_CPU_TRAIN_ARGS="-pe omp 1 -l mem_per_core=8G"
bash jobs/submit_all.sh --limit 2000
```

## 6. Monitor jobs

```bash
qstat -u $USER
qstat -j <job_id>        # detailed status
tail -f logs/build_epochs.<job_id>.log
```

## 7. Tuning job resources

Override resources with `qsub` flags or the submission-wrapper env vars below.
Edit the `#$ -l` directives in `jobs/*.sh` only if you want to change repo defaults.
Key variables you can override at submission time:

```bash
export DEBASS_LIMIT=2000
export DEBASS_MAX_N_DET=20
export DEBASS_N_EST=500
export DEBASS_GPU_EXPERTS="parsnip"
export DEBASS_GPU_QUEUE=l40s
export DEBASS_GPU_COUNT=1
export DEBASS_GPU_MEMORY=40G
export DEBASS_PYTHON_MODULE=python3/3.10.12
export DEBASS_VENV=/projectnb/<yourproject>/venvs/debass_meta_env
export DEBASS_QSUB_COMMON_ARGS="-P <project>"
export DEBASS_QSUB_CPU_ARGS="-q <cpu_queue>"
export DEBASS_QSUB_CPU_TRAIN_ARGS="-pe omp 1 -l mem_per_core=8G"
export DEBASS_QSUB_GPU_ARGS="-q l40s"
```

## 8. Output locations

| Stage | Output | Description |
|-------|--------|-------------|
| backfill | `data/bronze/<broker>_<ts>.parquet` | Raw broker payloads |
| normalize | `data/silver/broker_events.parquet` | Event-level silver with exactness metadata |
| local_infer | `data/silver/local_expert_outputs/<expert>/part-latest.parquet` | Per-epoch local expert scores |
| truth | `data/truth/object_truth.parquet` | TNS spectroscopic crossmatch (3-tier quality) |
| snapshots | `data/gold/object_epoch_snapshots.parquet` | No-leakage epoch features + projected expert probs |
| trust snapshots | `data/gold/object_epoch_snapshots_trust.parquet` | Above + trust scores (q_{e,t}) appended |
| helpfulness | `data/gold/expert_helpfulness.parquet` | Per-expert correctness labels for trust training |
| trust models | `models/trust/<expert>/model.pkl` + `metadata.json` | Per-expert LightGBM trust heads |
| followup model | `models/followup/model.pkl` + `metadata.json` | Binary follow-up recommendation model |
| split | `models/trust/metadata.json` | Train/cal/test object ID lists (reproducible) |
| trust metrics | `reports/metrics/expert_trust_metrics.json` | Per-expert ROC-AUC, Brier on held-out test set |
| followup metrics | `reports/metrics/followup_metrics.json` | Follow-up ROC-AUC, Brier on held-out test set |
| score payloads | `reports/scores/scores_ndet{3,5,10}.jsonl` | Trust-aware expert_confidence + ensemble per object |
| results summary | `reports/results_summary.txt` | Human-readable analysis of all metrics |

## 9. Pipeline architecture summary

```
Remote brokers + local experts
     |
     v
Bronze
     |
     v
Silver v2: broker events
     |
     v
Gold v2: object_epoch_snapshots
     |
     v
expert_helpfulness
     |
     v
Per-expert trust heads
     |
     v
Optional follow-up head
     |
     v
reports/metrics + reports/scores
```

## 10. Semantic type contract

The pipeline enforces that broker outputs are **never treated as
interchangeable probability vectors**:

| Broker | Field | Semantic type |
|--------|-------|---------------|
| ALeRCE | object snapshot probabilities | `probability`, but historical backfill is `latest_object_unsafe` |
| Fink | `rf_snia_vs_nonia`, `snn_*` | `probability` |
| Fink | `finkclass` | `heuristic_class` — do NOT use as posterior |
| Lasair | `sherlock_classification` | `crossmatch_tag` — context only |
| SuperNNova | `prob_class0/1` | `probability` |
| ParSNIP | `class_probabilities` | `probability` |
