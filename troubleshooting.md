# Troubleshooting

This file tracks recurring DEBASS data-engineering failures and the fastest known fix.

## Standard Check Loop

Run these after any pipeline change that touches broker ingest, normalization, gold snapshots, or training inputs:

```bash
python3.11 -m pytest -q
python3.11 scripts/audit_data_readiness.py --root <run-root> --labels data/labels.csv --lightcurves-dir data/lightcurves
python3.11 scripts/check_data_benchmarks.py --root <run-root> --labels data/labels.csv --lightcurves-dir data/lightcurves
```

If the benchmark check needs a specific score file:

```bash
python3.11 scripts/check_data_benchmarks.py \
  --root <run-root> \
  --labels data/labels.csv \
  --lightcurves-dir data/lightcurves \
  --scores <run-root>/reports/scores_ndet4.jsonl
```

To rebuild everything before ML in the correct order:

```bash
python3.11 scripts/run_preml_pipeline.py --root <run-root> --labels data/labels.csv --lightcurves-dir data/lightcurves
python3.11 scripts/check_preml_readiness.py --root <run-root> --labels data/labels.csv --lightcurves-dir data/lightcurves
```

## Weak Truth Contamination

Symptom:
- readiness report warns about weak ALeRCE self-label truth
- trust metrics look good on smoke data but are not science-grade

Cause:
- `scripts/build_truth_table.py --labels ...` builds an explicitly weak truth table

Fix:
- prefer `scripts/build_truth_table.py --input <curated_truth.csv>`
- or run `scripts/train_expert_trust.py --strict-labels` to block weak-label trust training

Prevention:
- keep `allow_weak_truth = false` in the benchmark file once curated truth exists

## Unsafe ALeRCE History Leaks Into Trust Training

Symptom:
- ALeRCE experts appear available in snapshots for historical rows
- trust training quietly starts learning from object-level ALeRCE snapshots

Cause:
- historical ALeRCE classifier snapshots are `latest_object_unsafe`

Fix:
- keep the default path in `src/debass_meta/ingest/gold.py` and `scripts/train_expert_trust.py`
- only use `--allow-unsafe-alerce` for explicit experiments, never for baseline science claims

Prevention:
- benchmark against `require_snapshot_experts` that are truly epoch-safe, not historical ALeRCE

## Lasair Falls Back To Fixtures

Symptom:
- readiness warning says Lasair is not returning live bronze payloads
- `lasair/sherlock` rows exist but are fixture-backed or unavailable

Cause:
- `LASAIR_TOKEN` is missing or the `lasair` client is unavailable

Fix:
- add `LASAIR_TOKEN` to your shell or `.env`
- confirm the adapter sees it in `src/debass_meta/access/lasair.py`

Prevention:
- keep Lasair as an explicit benchmark requirement only when credentials are available on that machine

## Fink Live Fetch Breaks

Symptom:
- Fink bronze rows disappear or downgrade to fixture-only

Cause:
- API endpoint drift or SSL issues on the public Fink endpoint

Fix:
- re-check `src/debass_meta/access/fink.py`
- confirm the adapter still reaches `https://api.fink-portal.org`

Prevention:
- benchmark for live `fink` bronze rows so this fails fast

## Local Experts Stay Stub-Only

Symptom:
- readiness warns that local expert outputs are present but none are available
- `parsnip` and `supernnova` never show up as available in snapshots

Cause:
- optional local-expert packages or model weights are not installed

Fix:
- install optional dependencies from `pyproject.toml`
- provide real ParSNIP and SuperNNova model assets
- rerun `scripts/local_infer.py`

Prevention:
- do not treat stub outputs as trainable evidence
- benchmark local experts only after real weights are installed

## Missing Lightcurves For Labelled Objects

Symptom:
- benchmark fails on complete lightcurve coverage
- snapshots build only for a subset of labelled objects

Cause:
- `data/lightcurves/` does not contain one file per labelled object

Fix:
- rerun `scripts/download_alerce_training.py` or the relevant lightcurve fetch step

Prevention:
- keep `require_complete_lightcurves = true` in the benchmark spec

## Exact-Alert Rows Missing `alert_id` Or `n_det`

Symptom:
- benchmark fails on exact-alert contract coverage
- silver contains alert-safe rows without `alert_id`, `n_det`, or `event_time_jd`

Cause:
- broker adapters did not emit full alert metadata, or empty broker responses were being promoted into fake event rows

Fix:
- check `src/debass_meta/access/fink.py` for alert-id extraction
- check `src/debass_meta/ingest/silver.py` to make sure empty payloads are skipped instead of becoming synthetic events

Prevention:
- keep the exact-alert benchmark checks enabled in `benchmarks/data_engineering.toml`

## Score Payload Contains `NaN`

Symptom:
- JSONL payload has `NaN` instead of `null`
- downstream readers reject the payload or silently coerce it

Cause:
- pandas scalars were serialized without cleaning

Fix:
- check `scripts/score_nightly.py`
- rerun the scoring step after fixing serialization

Prevention:
- keep the payload contract tests and benchmark check enabled

## Metrics Written To The Wrong Run Root

Symptom:
- smoke metrics appear under repo-level `reports/` instead of the chosen run directory

Cause:
- trainer scripts used hardcoded report paths

Fix:
- rely on the model-relative defaults in `scripts/train_expert_trust.py`, `scripts/train_followup.py`, and `scripts/train_early.py`

Prevention:
- always benchmark against the exact run root you intend to ship or compare
