# CLAUDE.md — DEBASS Project Context

This file tells Claude Code how to work with this repository.

## Project Summary

DEBASS (Detection-Based Astronomical Source Spectroscopic) meta-classifier.
Fuses heterogeneous broker outputs (ALeRCE, Fink, Lasair) + per-epoch lightcurve
features to classify transients as **SN Ia / non-Ia SN-like / other** after only
3-5 detections, for spectroscopic follow-up prioritisation with Rubin/LSST.

Supports **mixed ZTF + LSST training**. Both surveys flow through the same
pipeline with a union feature set (51 LC features across ugrizy bands).

## Key Design Decisions

- **No-leakage epoch table**: features at n_det=N are computed from lightcurve
  truncated to exactly N detections. Never use future detections.
- **Mixed-survey support**: ZTF (g,r) and LSST (ugrizy) detections are normalised
  to a common schema via `features/detection.py`. Missing bands are NaN —
  LightGBM handles this natively. `survey_is_lsst` flag lets models learn
  survey-specific trust patterns.
- **18 registered experts** across both surveys. Each gets a per-expert trust
  head (LightGBM binary) and projection to ternary (p_snia, p_nonIa, p_other).
- **Fink LSST gives per-alert scores** (`api.lsst.fink-portal.org/api/v1/sources`
  with `diaObjectId`). SNN/CATS/EarlySNIa scores genuinely evolve per detection —
  the richest signal for LSST trust training.
- **Dual-survey broker queries**: For LSST objects with ZTF counterparts,
  backfill queries BOTH endpoints (LSST native + ZTF supplement via association).
  Associations filtered at 2 arcsec max separation for high confidence.
- **LightGBM (not RandomForest)** for all classifiers. Native NaN handling
  means missing broker scores are passed through directly — no imputation.
- **Three-way train/cal/test split** with grouped object-level splits.
- **Multi-source truth**: TNS spectroscopic + Fink crossmatch (free) +
  broker consensus + host galaxy context (SIMBAD + photo-z + Gaia).

## Expert Registry (18 experts)

| Expert Key | Survey | Temporal | Source |
|---|---|---|---|
| `fink/snn` | ZTF | exact_alert | Fink ZTF SuperNNova |
| `fink/rf_ia` | ZTF | exact_alert | Fink ZTF Random Forest |
| `fink_lsst/snn` | LSST | exact_alert | Fink LSST SuperNNova |
| `fink_lsst/cats` | LSST | exact_alert | Fink LSST CATS |
| `fink_lsst/early_snia` | LSST | exact_alert | Fink LSST EarlySNIa |
| `alerce/lc_classifier_transient` | ZTF | latest_unsafe | ALeRCE legacy LC |
| `alerce/lc_classifier_BHRF_forced_phot_transient` | ZTF | latest_unsafe | ALeRCE BHRF |
| `alerce/lc_classifier_BHRF_forced_phot_top` | ZTF | latest_unsafe | ALeRCE BHRF top-level |
| `alerce/LC_classifier_ATAT_forced_phot(beta)` | ZTF | latest_unsafe | ALeRCE transformer |
| `alerce/stamp_classifier` | ZTF | static_safe | ALeRCE stamp (ZTF) |
| `alerce/stamp_classifier_2025_beta` | ZTF | static_safe | ALeRCE stamp 2025 |
| `alerce/stamp_classifier_rubin_beta` | LSST | static_safe | ALeRCE Rubin stamp |
| `lasair/sherlock` | any | static_safe | Lasair Sherlock context |
| `pittgoogle/supernnova_lsst` | LSST | exact_alert | Pitt-Google LSST SuperNNova (BigQuery) |
| `pittgoogle/supernnova_ztf` | ZTF | latest_unsafe | Pitt-Google ZTF SuperNNova (BigQuery) |
| `parsnip` | any | rerun_exact | Local ParSNIP |
| `supernnova` | any | rerun_exact | Local SuperNNova |
| `alerce_lc` | any | rerun_exact | Local ALeRCE LC |

Defined in `src/debass_meta/projectors/base.py:EXPERT_REGISTRY`.

## Python Environment

```bash
# Requires Python >= 3.10 (for ALeRCE LSST support)
# Local dev (macOS — use venv with Python 3.13)
source ~/.venvs/debass_py313/bin/activate
pip install -r env/requirements.txt

# SCC
python3 -m venv ~/debass_meta_env
source ~/debass_meta_env/bin/activate
pip install -r env/requirements.txt
```

## Full Pipeline (recommended)

```bash
bash jobs/submit_retrain_v2.sh --discover-lsst 400 --update-tns --local
```

Steps:
1. Discover 400 LSST candidates from ALeRCE (representative: SN/bogus/AGN/VS/asteroid)
2. Merge into labels.csv (deduplicates)
3. Fetch lightcurves + backfill all brokers (parallel, dual-survey)
4. TNS crossmatch (name + positional fallback)
5. Normalize bronze → silver (with dedup)
6. Build multi-source truth (TNS + Fink xm + host context)
7. Collect per-epoch local expert history
8. Build gold tables (51 LC features + 18 expert projections + survey flag)
9. Train expert trust + followup
10. Score + analyze

## Truth Sources (5 tiers)

| Tier | Source | Quality | How |
|---|---|---|---|
| 1 | TNS spectroscopic type | `spectroscopic` | crossmatch_tns.py + Fink `xm_tns_type` |
| 2 | TNS name (AT prefix) | `tns_untyped` | Fink `xm_tns_fullname` |
| 3 | Broker consensus | `consensus` | >=2 experts agree on class |
| 4 | Host galaxy context | `context` | SIMBAD=G + pstar<0.1 (SN come from galaxies) |
| 5 | ALeRCE stamp class | `weak` | Discovery label |

Fink LSST provides `xm_tns_type`, `xm_simbad_otype`, `xm_legacydr8_zphot`,
`xm_gaiadr3_Plx` for free in every query. `build_truth_multisource.py` merges all.

## Lightcurve Features (51)

```
Detection counts (7): n_det, n_det_{u,g,r,i,z,y}
Temporal (2):         t_since_first, t_baseline
All-band (8):         mag_first/last/min/mean/std/range, dmag_dt, dmag_first_last
Per-band (24):        mag_{first,last,mean,std}_{u,g,r,i,z,y}
Colours (8):          color_{gr,ri,iz,zy} + slopes
Quality (1):          mean_quality (rb/drb for ZTF, reliability for LSST)
Survey (1):           survey_is_lsst (0.0 = ZTF, 1.0 = LSST)
```

Defined in `src/debass_meta/features/lightcurve.py:FEATURE_NAMES`.

## Association Table (LSST ↔ ZTF)

For LSST objects with ZTF counterparts, the association table maps
diaObjectId → ZTF object ID. Used by:
- `fetch_lightcurves.py`: source ZTF lightcurve for LSST objects
- `backfill.py`: query ZTF brokers as supplement (dual-fetch)

**Confidence**: Default max separation 2 arcsec (conservative).
Each association records `sep_arcsec` so downstream can weight by confidence.

## Critical Column Name Contract

If you add a new expert:
1. Add to `EXPERT_REGISTRY` in `src/debass_meta/projectors/base.py`
2. Write a projector in `src/debass_meta/projectors/` (maps raw → ternary)
3. `ALL_EXPERT_KEYS` auto-updates. Gold/helpfulness/scoring auto-discover.

If you add a new LC feature:
1. Add to `FEATURE_NAMES` in `src/debass_meta/features/lightcurve.py`
2. Add to `DEFAULT_FEATURES` in `src/debass_meta/models/early_meta.py`

## File Map

```
src/debass_meta/
  access/         Broker adapters (ALeRCE ZTF+LSST, Fink ZTF+LSST, Lasair dual-mode)
  features/       Detection normalization + no-leakage LC feature extraction (51 features)
  ingest/         Bronze → silver (with dedup) → gold snapshot builders
  models/         Trust heads (per-expert LightGBM) + follow-up head + baseline
  projectors/     Expert-specific score → ternary projections (16 experts)
  experts/local/  Local rerunnable experts (SNN, ParSNIP, ALeRCE LC)

scripts/
  discover_lsst_training.py   Discover LSST candidates from ALeRCE (no token needed)
  merge_labels.py             Safely merge LSST candidates into labels.csv (dedup)
  backfill.py                 Fetch broker scores → bronze (--parallel 8, dual-survey)
  normalize.py                Bronze → silver (with dedup)
  crossmatch_tns.py           TNS crossmatch (name + positional fallback)
  build_truth_multisource.py  Multi-source truth (TNS + Fink xm + context + consensus)
  collect_epoch_history.py    Per-epoch local expert scores for LSST trust training
  build_object_epoch_snapshots.py  Build gold table (51 features + 16 experts)
  build_expert_helpfulness.py Build helpfulness rows for ALL 16 experts
  train_expert_trust.py       Train trust heads (auto-discovers 18 experts from helpfulness)
  train_followup.py           Train follow-up head
  score_nightly.py            Emit expert_confidence + trust-weighted ensemble
  fetch_lightcurves.py        Fetch + cache lightcurves (ZTF + LSST)

jobs/
  submit_retrain_v2.sh        Full pipeline (discover → train → score)
  submit_nightly.sh           Nightly scoring pipeline
  setup_local_experts.sh      Download SNN/RAPID/ORACLE weights for SCC

data/            (gitignored)
  labels.csv                          Object list (ZTF + LSST)
  lsst_candidates.csv                 LSST discovery output
  lightcurves/*.json                  Cached lightcurves
  bronze/*.parquet                    Raw broker payloads
  silver/broker_events.parquet        Event-level (with dedup)
  silver/local_expert_outputs/        Per-epoch local expert scores
  truth/object_truth.parquet          Multi-source truth table
  gold/object_epoch_snapshots.parquet 51 features + 18 expert projections
  gold/expert_helpfulness.parquet     Per-expert training rows (ALL 18)

models/          (gitignored)
  trust/<expert>/model.pkl + metadata.json
  followup/model.pkl + metadata.json
```

## Environment Variables

```bash
# In $DEBASS_ROOT/.env
LASAIR_TOKEN=your_token_here          # Works for both ZTF + LSST Lasair
# Optional per-endpoint:
# LASAIR_ZTF_TOKEN=...
# LASAIR_LSST_TOKEN=...
# TNS credentials (for --update-tns):
# TNS_API_KEY=...
# TNS_TNS_ID=...
# TNS_MARKER_NAME=...
# Pitt-Google Broker (BigQuery — public datasets, billing project required):
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-gcp-project
```

## Known Limitations

- Fink LSST EarlySNIa shows -1.0 for most objects (not triggered — needs min 7 epochs)
- ALeRCE LSST only has stamp classifier; no LC classifiers yet (ATAT coming)
- SuperNNova/ParSNIP local experts need LSST-trained weights (ELAsTiCC)
- Association table quality depends on crossmatch source and sky density
- Pitt-Google ZTF BQ table lacks timestamp columns (only candid) — ZTF PGB events
  are marked `latest_object_unsafe` since temporal matching requires `--allow-unsafe-latest-snapshot`
- Pitt-Google LSST has per-alert `kafkaPublishTimestamp` → `exact_alert` temporal matching
