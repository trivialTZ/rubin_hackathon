# DEBASS_meta — Trust-Aware Early-Epoch Transient Meta-Classifier

DEBASS estimates **which expert/classifier is trustworthy at a given early epoch**
and then optionally turns those trust estimates into a follow-up recommendation.

The primary science product is not a single fused class posterior. It is
`expert_confidence` for each expert at `(object_id, n_det, alert_jd)`.

## Science Motivation

Rubin/LSST will generate ~10 million alerts per night. Spectroscopic follow-up
resources are scarce — you need to decide *which transients are worth observing*
within hours, often before a full lightcurve is available.

Existing brokers and local experts (ALeRCE, Fink, Lasair, ParSNIP, SuperNNova)
provide heterogeneous signals with different semantics and different historical
trustworthiness. DEBASS models those experts separately, preserves temporal
exactness, and produces calibrated per-expert trust before any follow-up score.

## Architecture

```
Raw alerts / cached lightcurves
    │
    ├── Bronze: raw broker payloads + provenance
    ├── Silver v2: broker events with exactness metadata
    ├── Gold v2: object_epoch_snapshots via as-of joins
    │
    ├── Projectors
    │     ├── Fink SNN / RF
    │     ├── ALeRCE classifier snapshots
    │     ├── Lasair Sherlock context
    │     └── local experts (ParSNIP / SuperNNova)
    │
    ├── Expert trust heads  (LightGBM, binary, is_unbalance=True)
    │     └── calibrated q_e,t = P(expert e is trustworthy now)
    │
    ├── Optional follow-up head  (LightGBM, binary)
    │     └── trust-weighted p_follow_proxy
    │
    └── Baseline benchmark  (LightGBM, 3-class, early stopping)
          └── EarlyMetaClassifier with temperature-scaled calibration
```

## ML Algorithm

All classifiers use **LightGBM** gradient-boosted trees (Ke et al. 2017).
Key design choices:

- **Native NaN handling** — broker scores with sparse coverage (Fink ~100% NaN
  on most ZTF objects) are passed through directly; LightGBM learns optimal
  split direction for missing values. No median imputation.
- **Regularisation** — `learning_rate=0.05`, `num_leaves=31`, `reg_alpha=0.1`,
  `reg_lambda=0.1`, `feature_fraction=0.8`, `bagging_fraction=0.8`.
- **Class balance** — multiclass models use per-object inverse-frequency weights;
  binary models use `is_unbalance=True`.
- **Early stopping** — EarlyMetaClassifier trains up to 1000 rounds with early
  stopping (patience=50) on a held-out calibration set.
- **Calibration** — temperature scaling (multiclass) and isotonic regression
  (binary) available in `src/debass_meta/models/calibrate.py`.
- **Three-way split** — train/cal/test with no data leakage: calibration is fit
  on the cal set, metrics are reported on the separate test set.
- **Grouped object splits** — all epochs of the same object stay in the same
  fold (GroupKFold / GroupSplit) to prevent train/test contamination.

## No-Leakage Epoch Design

For each object with N total detections, the pipeline builds N training rows.
Row at n_det=k uses **only** detections 1..k. This means the model learns how
classification confidence evolves over time — and can be applied at any stage
of a new transient's lightcurve.

The same no-leakage rule also applies to expert evidence:

- exact expert events must satisfy `event_time_jd <= alert_jd`
- unsafe historical object snapshots are marked and excluded from trust training by default

## Science Contract

See [docs/science_contract.md](docs/science_contract.md).

Core rules:

- `expert_confidence` is the primary deliverable
- truth must be external or explicitly marked weak
- ALeRCE object-snapshot history is unsafe for historical trust unless archived or rerun epoch-safely
- `EarlyMetaClassifier` remains a baseline, not the final science architecture

## Trust-Aware Workflow (Local)

```bash
python3.11 -m pip install -r env/requirements.txt

# 1. Download weakly labelled seed objects + cached lightcurves
python3.11 scripts/download_alerce_training.py --limit 20

# 2. Fetch broker payloads and normalize to event-level silver
python3.11 scripts/backfill.py --broker all --from-labels data/labels.csv
python3.11 scripts/normalize.py

# 3. Build truth and exact object-epoch snapshots
python3.11 scripts/build_truth_table.py --labels data/labels.csv
python3.11 scripts/build_object_epoch_snapshots.py
python3.11 scripts/build_expert_helpfulness.py

# 4. Train trust heads and optional follow-up model
python3.11 scripts/train_expert_trust.py
python3.11 scripts/train_followup.py

# 5. Score epochs with the expert_confidence payload
python3.11 scripts/score_nightly.py --from-labels data/labels.csv --n-det 4
```

## Baseline Workflow

The legacy fused baseline remains available:

```bash
# NOTE: labels.csv is a weak/self-label seed set, not canonical science truth.
python3.11 scripts/build_epoch_table_from_lc.py
python3.11 scripts/train_early.py --n-estimators 500
python3.11 scripts/score_early.py --from-labels data/labels.csv --n-det 4
```

This baseline is useful for benchmarking, but it is not the primary science path.

## Full Training Run (BU SCC)

See [README_scc.md](README_scc.md) for environment setup.

### Phase 1: CPU-only pipeline (SuperNNova + LightGBM)

The entire Phase 1 pipeline runs on CPU. SuperNNova uses Fink's pre-trained
bidirectional LSTM weights via `classify_lcs`; all trust/follow-up models use
LightGBM. No GPU queue is required.

```bash
export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon

# 1. CPU prep: download + backfill + normalize + truth + gold tables
bash jobs/submit_cpu_prep.sh --limit 2000

# 2. SuperNNova batch inference (~10 min for 7K objects)
qsub -V -P pi-brout jobs/local_infer_snn.sh

# 3. After SNN finishes, submit the full chain with SGE job dependencies:
#    rebuild gold → train trust → train followup → score → analyze
bash jobs/submit_phase1_chain.sh

# Or hold on the SNN job if it's still running:
bash jobs/submit_phase1_chain.sh --after-snn <SNN_JOB_ID>
```

Monitor and read results:
```bash
qstat -u $USER
cat $DEBASS_ROOT/reports/results_summary.txt
cat $DEBASS_ROOT/reports/metrics/expert_trust_metrics.json
```

### Phase 2: GPU experts (ParSNIP — when weights available)

```bash
# Request an interactive GPU session
qrsh -l gpus=1 -q l40s

cd $DEBASS_ROOT
bash -l jobs/run_gpu_resume.sh --experts=parsnip
```

### Pipeline stages

1. CPU prep: `download_training → backfill(array) → normalize → crossmatch_tns → build_object_epoch_snapshots → build_expert_helpfulness`
2. Local expert inference: `local_infer_snn.sh` (CPU), `local_infer_gpu.sh` (GPU, Phase 2)
3. CPU training chain: `rebuild_gold → train_expert_trust → train_followup → score_nightly → analyze_results`

## Data Engineering QA

Use the readiness audit and benchmark check after changing ingest, normalization, snapshot building, or training inputs:

```bash
python3.11 -m pytest -q
python3.11 scripts/audit_data_readiness.py --root <run-root> --labels data/labels.csv --lightcurves-dir data/lightcurves
python3.11 scripts/check_data_benchmarks.py --root <run-root> --labels data/labels.csv --lightcurves-dir data/lightcurves
```

Edit `benchmarks/data_engineering.toml` as the project benchmark tightens. Recurring failures and fixes are tracked in `troubleshooting.md`.

For the full pre-ML build order, use:

```bash
python3.11 scripts/run_preml_pipeline.py --root tmp/preml_run --labels data/labels.csv --lightcurves-dir data/lightcurves
python3.11 scripts/check_preml_readiness.py --root tmp/preml_run --labels data/labels.csv --lightcurves-dir data/lightcurves
```

If you have curated truth, validate it before building the canonical truth layer:

```bash
python3.11 scripts/validate_truth_table.py path/to/curated_truth.parquet --require-strong
```

## Output

### Primary Science Product: `expert_confidence` Payload

The scoring pipeline (`score_nightly.py`) emits one JSONL record per
(object, n_det) pair. Each record contains:

```json
{
  "object_id": "ZTF21abcdef",
  "n_det": 3,
  "alert_jd": 2460170.91,
  "expert_confidence": {
    "fink/snn":    {"available": true,  "trust": 0.92, "projected": {"p_snia": 0.87, "p_nonIa_snlike": 0.08, "p_other": 0.05, ...}},
    "supernnova":  {"available": true,  "trust": 0.85, "projected": {"p_snia": 0.91, ...}},
    "fink/rf_ia":  {"available": true,  "trust": 0.78, "projected": {"p_snia_scalar": 0.83, ...}},
    "lasair/sherlock": {"available": true, "trust": null, "context_tag": "SN", "reason": "phase-1 context expert"},
    "parsnip":     {"available": false, "reason": "expert unavailable at this epoch"}
  },
  "ensemble": {
    "p_follow_proxy": 0.94,
    "recommended": true
  }
}
```

Per-expert fields:
- `available` -- whether the expert produced output at this epoch
- `trust` -- calibrated P(expert is correct) at this epoch (`q_{e,t}`)
- `projected` -- full ternary probability distribution (p_snia, p_nonIa_snlike, p_other, top1_prob, margin, entropy)
- `exactness` -- temporal correctness: `exact_alert`, `static_safe`, or `latest_object_unsafe`

Ensemble fields:
- `p_follow_proxy` -- trust-weighted probability that this object deserves spectroscopic follow-up
- `recommended` -- binary decision at configurable threshold (default 0.5)

### Evaluation and Test-Set Metrics

The pipeline uses a **three-way grouped split** (60% train / 20% calibration / 20% test)
at the object level. All epochs of the same object stay together. The split is
deterministic (seed=42) and saved to `models/trust/metadata.json` for reproducibility.

Test-set results are written to:

| File | Contents |
|------|----------|
| `reports/metrics/expert_trust_metrics.json` | Per-expert ROC-AUC, Brier score, n_test_rows on held-out test set |
| `reports/metrics/followup_metrics.json` | Follow-up model ROC-AUC, Brier, positive_rate on held-out test set |
| `reports/results_summary.json` | Machine-readable digest of all metrics, expert availability, temporal evolution |
| `reports/results_summary.txt` | Human-readable summary |
| `models/trust/metadata.json` | Exact train/cal/test object ID lists |

### File Inventory

Trust-aware pipeline outputs:

| Stage | Output |
|-------|--------|
| Silver events | `data/silver/broker_events.parquet` |
| Local expert scores | `data/silver/local_expert_outputs/<expert>/part-latest.parquet` |
| TNS truth | `data/truth/object_truth.parquet` |
| Epoch snapshots | `data/gold/object_epoch_snapshots.parquet` |
| Trust-augmented snapshots | `data/gold/object_epoch_snapshots_trust.parquet` |
| Expert helpfulness | `data/gold/expert_helpfulness.parquet` |
| Trust models | `models/trust/<expert>/model.pkl` + `metadata.json` |
| Follow-up model | `models/followup/model.pkl` + `metadata.json` |
| Score payloads | `reports/scores/scores_ndet{3,5,10}.jsonl` |
| Metrics | `reports/metrics/expert_trust_metrics.json`, `reports/metrics/followup_metrics.json` |
| Summary | `reports/results_summary.txt`, `reports/results_summary.json` |

Baseline outputs:

- `models/early_meta/early_meta.pkl`
- `reports/metrics/early_meta_metrics.json`

## Brokers

| Expert key | Type | Phase-1 trust head | Notes |
|------------|------|--------------------|-------|
| `fink/snn` | alert-level posterior-like | yes | exact alert history |
| `fink/rf_ia` | alert-level scalar Ia score | yes | exact alert history |
| `parsnip` | local rerun posterior | yes | rerun exact when weights exist |
| `supernnova` | local rerun posterior | yes | Fink-trained LSTM; batch CPU inference via classify_lcs |
| `alerce/lc_classifier_transient` | object snapshot | live-only | historical backfill is unsafe |
| `alerce/stamp_classifier` | object snapshot | live-only | historical backfill is unsafe |
| `lasair/sherlock` | context expert | context-only | static-safe, not calibrated posterior |

## Repository Structure

```
src/debass_meta/
  access/           Broker adapters
  features/         No-leakage lightcurve feature extraction
  ingest/           Bronze / broker-events / gold snapshot builders
  models/           Trust heads, follow-up head, baseline model
  projectors/       Expert-specific projections
  experts/local/    Local rerunnable experts
scripts/            Training and scoring entry points
jobs/               SCC scripts
schemas/            JSON schemas
inventory/          Source and expert metadata
docs/               Science contract
```

## Ternary Labels

| Label | Meaning |
|-------|---------|
| `snia` | Type Ia supernova |
| `nonIa_snlike` | Core-collapse SN (II, Ibc, SLSN, IIn, IIb) |
| `other` | AGN, variable star, asteroid, bogus, TDE |

## Follow-Up Proxy

Phase-1 uses a trust-weighted follow-up proxy head. If no operational follow-up
truth is available yet, the default proxy target is:

```text
follow_proxy = 1{final_class_ternary == "snia"}
```

That target should be described explicitly as a proxy, not as final operational truth.

## Citation

If you use this code, please cite the relevant broker papers:
- ALeRCE: Förster et al. 2021, AJ 161 242
- Fink: Möller & de Boissière 2020, MNRAS 491 4277
- SuperNNova: Möller & de Boissière 2020, MNRAS 491 4277
- ParSNIP: Boone 2021, AJ 162 275
