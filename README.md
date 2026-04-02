# DEBASS_meta — Trust-Aware Early-Epoch Transient Meta-Classifier

DEBASS estimates **which expert/classifier is trustworthy at a given early epoch**
and then turns those trust estimates into a follow-up recommendation.

The primary science product is not a single fused class posterior. It is
`expert_confidence` for each expert at `(object_id, n_det, alert_jd)`,
plus a **trust-weighted ensemble** probability for spectroscopic follow-up.

## Science Motivation

Rubin/LSST generates ~10 million alerts per night. Spectroscopic follow-up
resources are scarce — you need to decide *which transients are worth observing*
within hours, often before a full lightcurve is available.

Existing brokers (ALeRCE, Fink, Lasair) and local experts (ParSNIP, SuperNNova)
provide heterogeneous signals with different failure modes. DEBASS models those
experts separately, preserves temporal exactness, and produces calibrated
per-expert trust before any follow-up score.

**Supports both ZTF and LSST data.** Mixed-survey training uses a union feature
set (51 lightcurve features across ugrizy bands) with LightGBM's native NaN
handling — ZTF objects have NaN for i/z/y bands, LSST objects have NaN for
ZTF-only expert scores. The model learns survey-specific trust patterns.

## Data Flow & Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                              │
│                                                                     │
│  labels.csv ──────────────────────────────────────────────────────► │
│  (object_id, ra, dec, broker, ...)                                  │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐              │
│  │   ALeRCE    │  │    Fink      │  │    Lasair     │              │
│  │ (ZTF+LSST)  │  │  (ZTF only)  │  │ (ZTF+LSST)   │              │
│  │ v2.1+ API   │  │  alerts API  │  │  Sherlock     │              │
│  │ --parallel 8│  │              │  │  +LSST API    │              │
│  └──────┬──────┘  └──────┬───────┘  └──────┬────────┘              │
│         │                │                  │                       │
│         ▼                ▼                  ▼                       │
│  ┌──────────────────────────────────────────────────┐               │
│  │              Bronze (raw payloads)                │               │
│  │              data/bronze/*.parquet                │               │
│  └──────────────────────┬───────────────────────────┘               │
│                         │  normalize.py                             │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────┐               │
│  │     Silver (event-level, exactness-tagged)        │               │
│  │     data/silver/broker_events.parquet              │               │
│  │     + data/silver/local_expert_outputs/            │               │
│  └──────────────────────┬───────────────────────────┘               │
└─────────────────────────┼───────────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────────┐
│                    TRUTH LABELS                                     │
│                         │                                           │
│  TNS bulk CSV ──► crossmatch_tns.py ──► data/truth/object_truth    │
│  (160K objects)    (name match first,    .parquet                   │
│                     positional fallback   (three-tier:              │
│                     for LSST RA/Dec)       spectroscopic /          │
│                                            consensus / weak)        │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────────┐
│                    FEATURE EXTRACTION                               │
│                         │                                           │
│  Lightcurves ──► detection.py ──► lightcurve.py                    │
│  (*.json)        (normalize        (51 features:                   │
│                   ZTF fid/mag       7 band counts,                 │
│                   or LSST           8 mag stats,                   │
│                   flux/band)        24 per-band ugrizy,            │
│                                     8 colors gr/ri/iz/zy,          │
│                                     quality, survey flag)          │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────┐               │
│  │           Gold (object-epoch snapshots)            │               │
│  │  ┌────────────────────────────────────────────┐   │               │
│  │  │  For each (object_id, n_det=1..20):        │   │               │
│  │  │    • 51 LC features (no-leakage truncated) │   │               │
│  │  │    • survey flag (ZTF=0, LSST=1)           │   │               │
│  │  │    • 13 expert projections (ternary probs)  │   │               │
│  │  │    • truth labels (if available)            │   │               │
│  │  └────────────────────────────────────────────┘   │               │
│  │  data/gold/object_epoch_snapshots.parquet          │               │
│  └──────────────────────┬───────────────────────────┘               │
└─────────────────────────┼───────────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────────┐
│                    MODEL TRAINING                                   │
│                         │                                           │
│        ┌────────────────┴────────────────┐                          │
│        │    Expert Helpfulness Table      │                          │
│        │    (is each expert correct at    │                          │
│        │     each epoch?)                 │                          │
│        └────────────────┬────────────────┘                          │
│                         │                                           │
│        ┌────────────────▼────────────────┐                          │
│        │  Per-Expert Trust Heads (×13)    │                          │
│        │  LightGBM binary classifier     │                          │
│        │                                  │                          │
│        │  Input:  51 LC features          │                          │
│        │        + expert's own projection │                          │
│        │        + availability metadata   │                          │
│        │  Target: is_topclass_correct     │                          │
│        │  Output: trust q ∈ [0,1]         │                          │
│        │                                  │                          │
│        │  → models/trust/{expert}/        │                          │
│        └────────────────┬────────────────┘                          │
│                         │                                           │
│        ┌────────────────▼────────────────┐                          │
│        │    Follow-Up Model (shared)      │                          │
│        │    LightGBM binary classifier   │                          │
│        │                                  │                          │
│        │    Input:  all trust scores q_i  │                          │
│        │          + all expert projections │                          │
│        │          + 51 LC features        │                          │
│        │    Target: target_follow_proxy   │                          │
│        │    Output: p_follow ∈ [0,1]      │                          │
│        │                                  │                          │
│        │    → models/followup/            │                          │
│        └────────────────┬────────────────┘                          │
└─────────────────────────┼───────────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────────────────┐
│                    SCORING OUTPUT                                   │
│                         │                                           │
│        ┌────────────────▼────────────────┐                          │
│        │     score_nightly.py             │                          │
│        │                                  │                          │
│        │  Per expert:                     │                          │
│        │    trust, projected probs,       │                          │
│        │    mapped_pred_class             │                          │
│        │                                  │                          │
│        │  Ensemble:                       │                          │
│        │    p_snia_weighted (transparent   │                          │
│        │      trust-weighted average)     │                          │
│        │    p_follow_proxy (learned)      │                          │
│        │    recommended (yes/no)          │                          │
│        │                                  │                          │
│        │  → reports/scores/*.jsonl        │                          │
│        └─────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Registered Experts

| Expert key | Survey | Type | Trust head | Notes |
|------------|--------|------|------------|-------|
| `fink/snn` | ZTF | alert-level posterior | yes | SuperNNova via Fink alerts |
| `fink/rf_ia` | ZTF | alert-level scalar | yes | Random Forest Ia score |
| `parsnip` | any | local rerun posterior | yes | when weights available |
| `supernnova` | any | local rerun posterior | yes | Fink LSTM; batch CPU |
| `alerce_lc` | any | local LC expert | yes | LightGBM trained on TNS |
| `alerce/lc_classifier_transient` | ZTF | object snapshot | yes | legacy LC classifier |
| `alerce/lc_classifier_BHRF_forced_phot_transient` | ZTF | object snapshot | yes | BHRF forced-phot (newer) |
| `alerce/LC_classifier_ATAT_forced_phot(beta)` | ZTF | object snapshot | yes | transformer classifier |
| `alerce/stamp_classifier` | ZTF | object snapshot | yes | legacy stamp |
| `alerce/stamp_classifier_2025_beta` | ZTF | object snapshot | yes | updated stamp |
| `alerce/stamp_classifier_rubin_beta` | LSST | object snapshot | yes | Rubin-specific stamp |
| `lasair/sherlock` | any | context expert | context-only | static-safe crossmatch |

## ML Algorithm

All classifiers use **LightGBM** gradient-boosted trees (Ke et al. 2017):

- **Native NaN handling** — missing broker scores passed through directly;
  LightGBM learns optimal split direction. No imputation.
- **Regularisation** — `learning_rate=0.05`, `num_leaves=31`, `reg_alpha=0.1`,
  `reg_lambda=0.1`, `feature_fraction=0.8`, `bagging_fraction=0.8`.
- **Class balance** — binary models use `is_unbalance=True`.
- **Three-way split** — train/cal/test (60/20/20) with no data leakage.
- **Grouped object splits** — all epochs of same object stay in same fold.
- **Mixed-survey ready** — `survey_is_lsst` flag lets models learn per-survey trust.

## No-Leakage Epoch Design

For each object with N total detections, the pipeline builds N training rows.
Row at n_det=k uses **only** detections 1..k. Features at n_det=4 only use
information available after the 4th detection.

Expert evidence follows the same rule:
- exact events must satisfy `event_time_jd <= alert_jd`
- unsafe historical snapshots are marked and excluded from trust training by default

## Quick Start (Local)

```bash
# Requires Python >=3.10 (for ALeRCE LSST support)
pip install -r env/requirements.txt

# 1. Download seed objects + lightcurves
python3 scripts/download_alerce_training.py --limit 20

# 2. Fetch all broker scores (parallel)
python3 scripts/backfill.py --broker all --from-labels data/labels.csv --parallel 8

# 3. Normalize and build truth
python3 scripts/normalize.py
python3 scripts/crossmatch_tns.py --labels data/labels.csv \
    --bulk-csv data/tns_public_objects.csv --positional-fallback

# 4. Build gold tables and train
python3 scripts/build_object_epoch_snapshots.py
python3 scripts/build_expert_helpfulness.py
python3 scripts/train_expert_trust.py
python3 scripts/train_followup.py

# 5. Score
python3 scripts/score_nightly.py --from-labels data/labels.csv --n-det 4
```

## Full Training (BU SCC)

See [README_scc.md](README_scc.md) for environment setup.

### v2 Retrain (recommended)

```bash
export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon
cd $DEBASS_ROOT && git pull origin main

# Update venv (gets alerce v2.1+ for LSST)
source ${DEBASS_VENV:-$HOME/debass_meta_env}/bin/activate
pip install -r env/requirements.txt

# Dry-run first (see what would be submitted)
bash jobs/submit_retrain_v2.sh --update-alerce --dry-run

# Submit (re-fetches ALeRCE for BHRF/ATAT/Rubin classifiers, then retrains)
bash jobs/submit_retrain_v2.sh --update-alerce

# Full: also refresh TNS truth
bash jobs/submit_retrain_v2.sh --update-alerce --update-tns
```

The `--update-alerce` flag re-queries ALeRCE using 8 parallel threads (~5 min
for 2000 objects) to pick up BHRF/ATAT/Rubin classifiers. Without it, existing
cached data is reused.

### Nightly scoring pipeline

```bash
bash jobs/submit_nightly.sh                      # today's date
bash jobs/submit_nightly.sh --date 2026-04-02    # specific date
bash jobs/submit_nightly.sh --local              # run locally (no SGE)
```

### Environment variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `DEBASS_ROOT` | Project root on SCC | `/projectnb/pi-brout/rubin_hackathon` |
| `LASAIR_TOKEN` | Lasair API token ([get here](https://lasair.lsst.ac.uk)) | `abc123...` |
| `LASAIR_ENDPOINT` | Lasair LSST endpoint | `https://lasair.lsst.ac.uk` |
| `TNS_API_KEY` | TNS API key (for `--update-tns`) | |
| `TNS_TNS_ID` | TNS bot ID | |
| `TNS_MARKER_NAME` | TNS marker name | |

## Output

### Primary Science Product: `expert_confidence` Payload

The scoring pipeline (`score_nightly.py`) emits one JSONL record per
(object, n_det) pair:

```json
{
  "object_id": "ZTF21abcdef",
  "n_det": 4,
  "alert_jd": 2460170.91,
  "expert_confidence": {
    "fink/snn": {
      "available": true, "trust": 0.92,
      "projected": {"p_snia": 0.87, "p_nonIa_snlike": 0.08, "p_other": 0.05}
    },
    "alerce/lc_classifier_BHRF_forced_phot_transient": {
      "available": true, "trust": 0.78,
      "projected": {"p_snia": 0.65, "p_nonIa_snlike": 0.30, "p_other": 0.05}
    },
    "alerce/stamp_classifier_rubin_beta": {
      "available": true, "trust": 0.71,
      "projected": {"p_snia": 0.0, "p_nonIa_snlike": 0.82, "p_other": 0.18}
    },
    "lasair/sherlock": {
      "available": true, "trust": null,
      "prediction_type": "context_only", "context_tag": "SN"
    }
  },
  "ensemble": {
    "p_snia_weighted": 0.74,
    "p_nonIa_snlike_weighted": 0.19,
    "p_other_weighted": 0.07,
    "n_trusted_experts": 3,
    "p_follow_proxy": 0.88,
    "recommended": true
  }
}
```

**Per-expert fields:**
- `trust` — calibrated P(expert is correct at this epoch)
- `projected` — ternary probability distribution
- `exactness` — temporal correctness category

**Ensemble fields:**
- `p_snia_weighted` — transparent trust-weighted average across available experts
- `p_follow_proxy` — learned follow-up probability (LightGBM)
- `recommended` — binary decision at threshold (default 0.5)

### File Inventory

| Stage | Output |
|-------|--------|
| Bronze (raw) | `data/bronze/*.parquet` |
| Silver (events) | `data/silver/broker_events.parquet` |
| Local experts | `data/silver/local_expert_outputs/<expert>/` |
| TNS truth | `data/truth/object_truth.parquet` |
| Gold snapshots | `data/gold/object_epoch_snapshots.parquet` |
| Trust snapshots | `data/gold/object_epoch_snapshots_trust.parquet` |
| Helpfulness | `data/gold/expert_helpfulness.parquet` |
| Trust models | `models/trust/<expert>/model.pkl` + `metadata.json` |
| Follow-up model | `models/followup/model.pkl` + `metadata.json` |
| Score payloads | `reports/scores/scores_ndet{3,5,10}.jsonl` |
| Metrics | `reports/metrics/expert_trust_metrics.json` |
| Summary | `reports/results_summary.txt` |

## Ternary Labels

| Label | Meaning |
|-------|---------|
| `snia` | Type Ia supernova |
| `nonIa_snlike` | Core-collapse SN (II, Ibc/SESN, SLSN, IIn, IIb) |
| `other` | AGN, variable star, asteroid, bogus, TDE |

## Repository Structure

```
src/debass_meta/
  access/           Broker adapters (ALeRCE ZTF+LSST, Fink, Lasair ZTF+LSST)
  features/         Detection normalization + no-leakage LC feature extraction
  ingest/           Bronze → silver → gold snapshot builders
  models/           Trust heads, follow-up head, baseline model
  projectors/       Expert-specific score → ternary projections
  experts/local/    Local rerunnable experts (SNN, ParSNIP, ALeRCE LC)
scripts/            Training and scoring entry points
jobs/               SCC submission scripts
tests/              pytest suite (139+ tests)
schemas/            JSON schemas
docs/               Science contract, feasibility notes
```

## Citation

If you use this code, please cite the relevant broker papers:
- ALeRCE: Forster et al. 2021, AJ 161 242
- Fink: Moller & de Boissiere 2020, MNRAS 491 4277
- SuperNNova: Moller & de Boissiere 2020, MNRAS 491 4277
- ParSNIP: Boone 2021, AJ 162 275
