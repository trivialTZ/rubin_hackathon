# DEBASS — Early-Epoch Transient Meta-Classifier

Fuses heterogeneous astronomical broker outputs with per-epoch lightcurve features
to classify transients as **SN Ia / non-Ia SN-like / other** after only 3–5 detections.
Built for Rubin/LSST follow-up prioritisation.

## Science Motivation

Rubin/LSST will generate ~10 million alerts per night. Spectroscopic follow-up
resources are scarce — you need to decide *which transients are worth observing*
within hours, often before a full lightcurve is available.

Existing brokers (ALeRCE, Fink, Lasair, SuperNNova, ParSNIP) each provide partial,
heterogeneous signals. No single broker is universally trusted. DEBASS fuses them
into a single calibrated posterior with a follow-up priority score.

## Architecture

```
Raw alerts
    │
    ├── ALeRCE  ─── lc_classifier_transient (SNIa/SNII/SNIbc/SLSN)
    │               stamp_classifier (SN/AGN/VS/asteroid/bogus)
    │               lc_classifier_top (Transient/Periodic/Stochastic)
    │
    ├── Fink    ─── rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all
    │
    ├── Lasair  ─── sherlock_classification (context crossmatch)
    │
    ├── SuperNNova ─ prob_ia per epoch [SCC GPU]
    ├── ParSNIP  ── prob_ia per epoch [SCC GPU]
    │
    └── Raw lightcurve ── 24 features truncated at each n_det
                          (rise rate, color, per-band mags, etc.)

    All inputs → LightGBM EarlyMetaClassifier
                      │
                      ├── p(SN Ia)
                      ├── p(non-Ia SN-like)
                      ├── p(other)
                      └── follow_up_priority ∈ [0,1]
```

## No-Leakage Epoch Design

For each object with N total detections, the pipeline builds N training rows.
Row at n_det=k uses **only** detections 1..k. This means the model learns how
classification confidence evolves over time — and can be applied at any stage
of a new transient's lightcurve.

## Quick Start (Local)

```bash
pip install -r env/requirements.txt

# Download 20 labelled ZTF objects
python3 scripts/download_alerce_training.py --limit 20

# Fetch broker scores, build epoch table, train, score
python3 scripts/backfill.py --broker all --from-labels data/labels.csv
python3 scripts/normalize.py --skip-gold
python3 scripts/build_epoch_table_from_lc.py
python3 scripts/train_early.py --n-estimators 100
python3 scripts/score_early.py --from-labels data/labels.csv --n-det 4
```

## Full Training Run (BU SCC)

See [README_scc.md](README_scc.md) for environment setup.

```bash
export DEBASS_ROOT=/projectnb/<yourproject>/rubin_hackathon
bash jobs/submit_all.sh --limit 2000 --gpu
```

This submits a 5-job SGE chain:
1. `download_training` — 2000 labelled objects + lightcurves (~15 min)
2. `build_epochs` — broker scores + no-leakage epoch table (~30 min)
3. `local_infer_gpu` — SuperNNova + ParSNIP per (object, n_det) [A40 GPU]
4. `train_early` — LightGBM on full epoch table (~30 min)
5. `score_all` — scores at n_det=3,5,10 + follow-up reports

## Output

After training:
- `models/early_meta/early_meta.pkl` — trained classifier
- `reports/metrics/early_meta_metrics.json` — macro-F1, Brier, NLL at all epochs and n_det≤5
- `reports/scores/scores_ndet{3,5,10}.txt` — per-object classification + follow-up priority

## Brokers

| Broker | Type | Phase | Notes |
|--------|------|-------|-------|
| ALeRCE | Multiclass posterior | 1 | Public API, no credentials |
| Fink | SN Ia probability | 1 | Public REST API |
| Lasair | Sherlock crossmatch | 1 | Requires `LASAIR_TOKEN` |
| SuperNNova | SN Ia probability | 1 | Local weights on SCC GPU |
| ParSNIP | Multiclass posterior | 1 | Local weights on SCC GPU |
| ANTARES | — | 2 | Stub — future work |
| Pitt-Google | — | 2 | Stub — future work |

## Repository Structure

```
src/debass/         Python package
  access/           Broker adapters
  features/         Lightcurve feature extractor
  ingest/           Bronze/silver/gold Parquet pipeline
  models/           EarlyMetaClassifier + baselines + calibration
  experts/local/    SuperNNova, ParSNIP wrappers
scripts/            Pipeline scripts (run in order)
jobs/               SGE job scripts for BU SCC
schemas/            JSON schemas for bronze/silver/gold layers
inventory/          Broker + classifier metadata YAML
fixtures/           Offline test fixtures
env/                requirements.txt
```

## Ternary Labels

| Label | Meaning |
|-------|---------|
| `snia` | Type Ia supernova |
| `nonIa_snlike` | Core-collapse SN (II, Ibc, SLSN, IIn, IIb) |
| `other` | AGN, variable star, asteroid, bogus, TDE |

## Follow-Up Priority

```
follow_up_priority = 1.0 × p(SN Ia) + 0.3 × p(non-Ia SN-like)
```

Range [0, 1]. Objects with priority > 0.6 are recommended for immediate
spectroscopic follow-up.

## Citation

If you use this code, please cite the relevant broker papers:
- ALeRCE: Förster et al. 2021, AJ 161 242
- Fink: Möller & de Boissière 2020, MNRAS 491 4277
- SuperNNova: Möller & de Boissière 2020, MNRAS 491 4277
- ParSNIP: Boone 2021, AJ 162 275
