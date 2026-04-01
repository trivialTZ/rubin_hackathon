# DEBASS — Trust-Aware Early-Epoch Transient Meta-Classifier

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
    ├── Expert trust heads
    │     └── calibrated q_e,t = P(expert e is trustworthy now)
    │
    └── Optional follow-up head
          └── trust-weighted p_follow_proxy
```

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
python3.11 scripts/train_early.py --n-estimators 100
python3.11 scripts/score_early.py --from-labels data/labels.csv --n-det 4
```

This baseline is useful for benchmarking, but it is not the primary science path.

## Full Training Run (BU SCC)

See [README_scc.md](README_scc.md) for environment setup.

```bash
export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon

# 1. CPU prep from a login node
bash jobs/submit_cpu_prep.sh --limit 2000

# 2. After CPU prep finishes, resume from your GPU session/node
bash -l jobs/run_gpu_resume.sh
```

Recommended split:
1. CPU prep: `download_training → backfill(array) → normalize → build_truth_table → build_object_epoch_snapshots → build_expert_helpfulness`
2. GPU inference when enabled: `local_infer`
3. CPU training: `train_expert_trust → train_followup → score_nightly`

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

Trust-aware outputs:

- `data/silver/broker_events.parquet`
- `data/truth/object_truth.parquet`
- `data/gold/object_epoch_snapshots.parquet`
- `data/gold/expert_helpfulness.parquet`
- `models/trust/<expert>/model.pkl`
- `models/trust/<expert>/calibrator.pkl`
- `models/followup/model.pkl`
- `reports/metrics/expert_trust_metrics.json`
- `reports/metrics/followup_metrics.json`
- `reports/scores/*.jsonl`

Baseline outputs:

- `models/early_meta/early_meta.pkl`
- `reports/metrics/early_meta_metrics.json`

## Brokers

| Expert key | Type | Phase-1 trust head | Notes |
|------------|------|--------------------|-------|
| `fink/snn` | alert-level posterior-like | yes | exact alert history |
| `fink/rf_ia` | alert-level scalar Ia score | yes | exact alert history |
| `parsnip` | local rerun posterior | yes | rerun exact when weights exist |
| `supernnova` | local rerun posterior | pending | wrapper incomplete |
| `alerce/lc_classifier_transient` | object snapshot | live-only | historical backfill is unsafe |
| `alerce/stamp_classifier` | object snapshot | live-only | historical backfill is unsafe |
| `lasair/sherlock` | context expert | context-only | static-safe, not calibrated posterior |

## Repository Structure

```
src/debass_meta_meta/
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
