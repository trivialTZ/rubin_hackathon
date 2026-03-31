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
cd /projectnb/<yourproject>
git clone <repo-url> rubin_hackathon
cd rubin_hackathon

# Create the virtualenv in project space, not $HOME
python3 -m venv /projectnb/<yourproject>/venvs/debass_meta_meta_env
source /projectnb/<yourproject>/venvs/debass_meta_meta_env/bin/activate

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
export DEBASS_VENV=/projectnb/<yourproject>/venvs/debass_meta_meta_env
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

## 4. Recommended workflow

Run the CPU-safe stages first from a login node:

```bash
cd $DEBASS_ROOT

# One-command bootstrap: verify checkout, load Python, create venv,
# install requirements, then submit CPU prep
bash -l jobs/scc_bootstrap.sh --limit 2000
```

Recommended CPU-first order:

```text
download_training
  → backfill (array)
  → normalize
  → build_truth_table
  → build_object_epoch_snapshots
  → build_expert_helpfulness
```

To watch both jobs with live progress bars:

```bash
$DEBASS_VENV/bin/python scripts/watch_cpu_prep.py
```

After CPU prep finishes, write a summary of what was downloaded and how the
CPU-side artifacts are structured:

```bash
$DEBASS_VENV/bin/python scripts/summarize_cpu_prep.py \
  --json-out reports/summary/cpu_prep_summary.json
```

That report should summarize:

- `data/labels.csv` (weak self-label seed set)
- `data/lightcurves/*.json`
- `data/bronze/*.parquet`
- `data/silver/broker_events.parquet`
- `data/truth/object_truth.parquet`
- `data/gold/object_epoch_snapshots.parquet`
- `data/gold/expert_helpfulness.parquet`

Use `--strict` if you want the command to exit nonzero when labelled objects
are missing from lightcurves, silver, or the epoch table.

If you already have the SCC environment set up and only want to resubmit the
CPU stage, you can still run:

```bash
bash jobs/submit_cpu_prep.sh --limit 2000
```

After those finish, request an interactive GPU session and resume from the prepared checkpoint:

```bash
# Check currently available GPU queues
qgpus -v -s

# Request an interactive GPU shell
# Example matching your current SCC usage:
qrsh -l gpus=1 -q l40s

# Once the qrsh session starts:
cd $DEBASS_ROOT

# Default GPU expert list is ParSNIP
bash -l jobs/run_gpu_resume.sh

# If you later enable more experts, override the list explicitly
bash -l jobs/run_gpu_resume.sh --experts=parsnip,supernnova

# Baseline benchmark only
bash -l jobs/run_gpu_resume.sh --baseline
```

GPU should be used only for local rerunnable experts. CPU is the default for:

- silver/gold building
- expert trust training
- follow-up training
- evaluation
- scoring

The default expert list is `parsnip` because the repo now supports real
ParSNIP inference when both `model.pt` and `classifier.pkl` are present in
`artifacts/local_experts/parsnip/`. SuperNNova is still a stub path unless you
complete its model loader. The GPU stage fails fast if a requested expert is
missing its package, model, or classifier.

Note: on SCC, `nvidia-smi` may show all physical GPUs on the node even when your
job only owns one GPU. The DEBASS resume script now checks `torch.cuda.is_available()`
and `torch.cuda.device_count()` instead of trusting the raw `nvidia-smi` count.

If you still want the single-submit batch flow, it now defaults to the trust-aware chain:

```bash
bash jobs/submit_all.sh --gpu --limit 2000

# Baseline benchmark only
bash jobs/submit_all.sh --gpu --baseline --limit 2000
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

## 6. Monitor jobs

```bash
qstat -u $USER
qstat -j <job_id>        # detailed status
tail -f logs/build_epochs.<job_id>.log
```

## 7. Tuning job resources

Edit the `#$ -l` directives in `jobs/*.sh` to match your queue limits.
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
export DEBASS_VENV=/projectnb/<yourproject>/venvs/debass_meta_meta_env
```

## 8. Output locations

| Stage | Output |
|-------|--------|
| backfill | `data/bronze/<broker>_<ts>.parquet` |
| normalize | `data/silver/broker_events.parquet` |
| truth | `data/truth/object_truth.parquet` |
| snapshots | `data/gold/object_epoch_snapshots.parquet` |
| helpfulness | `data/gold/expert_helpfulness.parquet` |
| local_infer | `data/silver/local_expert_outputs/<expert>/part-*.parquet` |
| trust | `models/trust/<expert>/model.pkl` |
| followup | `models/followup/model.pkl` |
| report | `reports/metrics/*.json`, `reports/scores/*.jsonl` |

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
