# Rubin Alignment ExecPlan

This ExecPlan is for the next project question only:

- Are we actually building a system that gives reliable confidence for broker classifiers on new Rubin transients?
- If current training is mostly ZTF-backed, how do we cross-check and then move toward Rubin-valid operation?
- How should we handle broker/API differences, especially Lasair Rubin vs Lasair ZTF?

The answer from the current codebase is:

1. The architecture is pointed at the right science product: per-expert `expert_confidence`.
2. The implemented data and training paths are still mostly ZTF-backed.
3. Rubin operations are discussed in notes, but the production scoring path is not yet Rubin-complete.

## Current Code Reality

### What is already correct

- The main science contract is per-expert confidence, not a single fused posterior.
- Event-level timing and no-future-leakage lightcurve slicing are already built into the silver/gold path.
- Lasair Sherlock is already treated as context, not as a fake posterior.
- Unsafe historical ALeRCE object snapshots are already marked `latest_object_unsafe`.

### What is still blocking the Rubin goal

- `scripts/download_alerce_training.py` still builds weak seed data from ALeRCE ZTF objects.
- `scripts/download_tns_training.py` still filters TNS on ZTF internal names and then fetches ALeRCE lightcurves.
- `src/debass_meta/access/alerce.py`, `src/debass_meta/access/fink.py`, and `src/debass_meta/access/lasair.py` still hard-code `survey="ZTF"`.
- `inventory/brokers.yaml` still points Lasair at the ZTF endpoint and ALeRCE at the ZTF API surface.
- `scripts/score_new_object.py` is not the real broker-fusion scoring path yet. It only runs local SuperNNova and leaves broker scoring as TODO.
- `src/debass_meta/features/lightcurve.py` mentions LSST `fid=3`, but the actual feature set is still effectively g/r only.
- Published trust artifacts are trained only for `fink/rf_ia`, `fink/snn`, and `supernnova`, and the published metadata still reports weak labels.

## Delivery Principle

Do not pretend we can jump directly from ZTF bootstrap to Rubin-trained science claims.

The rollout must be:

1. ZTF bootstrap remains allowed.
2. Rubin shadow-mode ingestion starts immediately.
3. Rubin cross-check and calibration happen before Rubin-confidence claims.
4. Rubin production scoring only starts after expert-level validation exists on Rubin-era data.

## ExecPlan R0 - Freeze the operating claim

**Goal**
Make the repo explicit about what is true now versus what is still planned.

**Why first**
Right now the codebase can easily be misunderstood as "Rubin-ready" when it is really "Rubin-oriented, ZTF-bootstrapped".

**Tasks**
1. Add a short document that defines three operating modes:
   - `ztf_bootstrap`
   - `rubin_shadow`
   - `rubin_validated`
2. State which models, metrics, and payloads are allowed to be reported in each mode.
3. State that ZTF-trained trust heads are bootstrap models, not final Rubin calibration.
4. State that Lasair/ALeRCE/Fink API surfaces are survey-dependent and must not be silently mixed.

**Outputs**
- `docs/rubin_operating_modes.md`
- `README.md` update

**Done when**
- Nobody can confuse "current code runs" with "current code is Rubin-validated".

---

## ExecPlan R1 - Make survey a first-class data contract

**Goal**
Remove implicit ZTF assumptions from adapters, normalized events, snapshots, and model inputs.

**Why now**
Today the system assumes object identity, endpoint semantics, and feature meaning in a ZTF-shaped way.

**Tasks**
1. Add explicit survey fields and enums to the data contracts:
   - `survey`
   - `survey_object_id`
   - `survey_program`
   - `broker_object_id`
2. Update `schemas/broker_events.json`, `schemas/object_truth.json`, and `schemas/object_epoch_snapshot_v2.json` to require survey provenance.
3. Update silver normalization so survey provenance survives bronze to silver to gold unchanged.
4. Update readiness checks to report coverage by survey and expert, not only by expert.
5. Ensure truth rows record both truth source and survey source separately.

**Files likely touched**
- `schemas/broker_events.json`
- `schemas/object_truth.json`
- `schemas/object_epoch_snapshot_v2.json`
- `src/debass_meta/ingest/silver.py`
- `src/debass_meta/ingest/gold.py`
- `scripts/check_preml_readiness.py`
- `scripts/audit_data_readiness.py`

**Outputs**
- Survey-aware schemas
- Survey-aware readiness report

**Done when**
- A single run can contain ZTF and Rubin objects without silent field collisions or hidden assumptions.

---

## ExecPlan R2 - Split broker adapters by survey/API surface

**Goal**
Support Rubin-facing broker ingestion without reusing ZTF-specific assumptions.

**Why now**
Lasair Rubin is not the same operational API surface as Lasair ZTF, and the same issue can appear for ALeRCE and Fink.

**Tasks**
1. Introduce a survey-aware adapter contract:
   - adapter name
   - broker
   - survey
   - endpoint/version
   - semantic type
2. Refactor current adapters so `survey="ZTF"` is no longer hard-coded.
3. Add separate endpoint configuration for:
   - ALeRCE ZTF
   - ALeRCE Rubin
   - Lasair ZTF
   - Lasair Rubin
   - Fink ZTF
   - Fink Rubin, if the broker surface differs
4. Preserve raw payload and endpoint metadata in bronze.
5. Add fixture coverage for at least one Rubin example per broker path before turning on live runs.
6. Update `inventory/brokers.yaml` to list broker-survey pairs rather than a single blended broker entry.

**Files likely touched**
- `src/debass_meta/access/base.py`
- `src/debass_meta/access/alerce.py`
- `src/debass_meta/access/fink.py`
- `src/debass_meta/access/lasair.py`
- `src/debass_meta/access/__init__.py`
- `inventory/brokers.yaml`
- `tests/test_fink_adapter.py`
- new adapter tests for Rubin fixtures

**Outputs**
- Survey-aware adapters
- Broker inventory by survey/API
- Rubin fixtures for adapter tests

**Done when**
- The code can ingest Rubin broker payloads without pretending they are ZTF payloads.

---

## ExecPlan R3 - Replace broker object ID as the only join key

**Goal**
Introduce a canonical candidate identity strategy that works across Rubin brokers.

**Why now**
ZTF bootstrap used broker object IDs heavily. Rubin cross-broker matching will be less reliable if we assume one object ID namespace.

**Tasks**
1. Define a canonical candidate manifest with:
   - `candidate_id`
   - sky position
   - first-seen time
   - survey
   - list of broker aliases
2. Make nightly discovery deduplicate by sky position and recency, not only object ID.
3. Store alias mappings between canonical candidate IDs and broker object IDs.
4. Update bronze and silver records to carry both canonical candidate ID and broker object ID when known.
5. Make truth crossmatch operate on candidate aliases, not only ZTF internal names.

**Files likely touched**
- `scripts/discover_nightly.py`
- new `scripts/build_candidate_manifest.py`
- `src/debass_meta/ingest/bronze.py`
- `src/debass_meta/ingest/silver.py`
- `src/debass_meta/access/tns.py`

**Outputs**
- `data/nightly/<date>/candidate_manifest.parquet`
- alias table between canonical IDs and broker IDs

**Done when**
- The pipeline can follow the same Rubin transient across multiple brokers even when broker identifiers differ.

---

## ExecPlan R4 - Build Rubin shadow-mode ingestion and archival

**Goal**
Start collecting Rubin-era broker evidence now, even before the training regime is fully Rubin-native.

**Why now**
Without archived Rubin broker outputs, there is nothing to validate, calibrate, or train against later.

**Tasks**
1. Add a nightly Rubin discovery mode that pulls candidate lists from live brokers and/or FastDB prefilters.
2. For every nightly candidate, archive:
   - candidate manifest row
   - raw broker payloads
   - normalized event rows
   - exact query timestamp and endpoint version
3. Record which experts were actually available that night.
4. Generate a nightly cross-check report:
   - broker overlap
   - score availability
   - score drift summary
   - object counts by detection count
5. Keep shadow-mode outputs separate from ZTF bootstrap training data.

**Files likely touched**
- `scripts/discover_nightly.py`
- `scripts/backfill.py`
- `scripts/normalize.py`
- new `scripts/report_rubin_shadow.py`
- `jobs/submit_nightly.sh`

**Outputs**
- `data/nightly/<date>/candidates.csv`
- `data/nightly/<date>/candidate_manifest.parquet`
- `data/nightly/<date>/bronze/*.parquet`
- `data/nightly/<date>/silver/broker_events.parquet`
- `reports/nightly/<date>/rubin_shadow_report.md`

**Done when**
- The team can point to archived Rubin broker evidence for concrete nights, not only notes about future feasibility.

---

## ExecPlan R5 - Define the training regime explicitly

**Goal**
Move from "one mixed training set" to a staged training strategy that respects domain shift.

**Why now**
A ZTF-heavy trust model may still be useful, but only as a bootstrap prior. It should not be the final Rubin trust calibration.

**Tasks**
1. Define three training pools:
   - `ztf_bootstrap_train`
   - `rubin_shadow_unlabeled`
   - `rubin_strong_truth`
2. Add dataset provenance columns to truth, snapshots, helpfulness, and metrics:
   - `train_survey`
   - `eval_survey`
   - `truth_quality`
   - `truth_source`
3. Update training scripts so metrics are always sliced by survey and truth quality.
4. Add a training option to:
   - train on ZTF only
   - train on ZTF plus survey flag
   - calibrate on Rubin-only holdout
5. Add a hard guard against silently reporting mixed-survey metrics as if they were Rubin-only metrics.

**Files likely touched**
- `scripts/build_truth_table.py`
- `scripts/build_object_epoch_snapshots.py`
- `scripts/build_expert_helpfulness.py`
- `scripts/train_expert_trust.py`
- `src/debass_meta/models/expert_trust.py`
- `scripts/train_followup.py`
- metrics/reporting scripts

**Outputs**
- Survey-sliced trust metrics
- Survey-aware training metadata

**Done when**
- Every reported model artifact states whether it is ZTF bootstrap, mixed-survey, or Rubin-calibrated.

---

## ExecPlan R6 - Add Rubin cross-check and calibration reports

**Goal**
Measure how each expert behaves on Rubin data before retraining claims are made.

**Why now**
The scientific question is per-expert reliability. That reliability may shift under Rubin cadence, bands, image properties, and broker implementations.

**Tasks**
1. Build a per-expert Rubin cross-check report with:
   - availability rate
   - prediction distribution
   - exactness distribution
   - disagreement rate between experts
   - drift against ZTF bootstrap score distributions
2. Add a small strong-truth Rubin evaluation table as soon as confirmed events exist.
3. Add per-expert reliability plots and calibration curves on Rubin holdout data.
4. Treat Lasair Sherlock separately as context:
   - measure usefulness as context
   - do not force it into posterior calibration
5. Gate Rubin production mode on minimum expert coverage and calibration thresholds.

**Files likely touched**
- new `scripts/report_expert_shift.py`
- new `scripts/plot_expert_reliability.py`
- `scripts/analyze_results.py`
- reporting docs

**Outputs**
- `reports/metrics/expert_shift_rubin.json`
- `reports/metrics/expert_reliability_rubin.json`
- reliability plots by expert and survey

**Done when**
- We can answer, with evidence, whether each broker expert is transferring cleanly from ZTF bootstrap to Rubin operation.

---

## ExecPlan R7 - Make the real new-object scoring path complete

**Goal**
Replace the current partial `score_new_object.py` path with actual broker-fusion scoring for a new transient.

**Why now**
The current user-facing script is not yet the real science path for a new Rubin transient.

**Tasks**
1. Replace the current "local SuperNNova only" behavior with:
   - fetch broker outputs for one object or candidate manifest row
   - normalize to event-level rows
   - build a single-object snapshot
   - project expert evidence
   - run trust heads
   - emit `expert_confidence`
2. Support both:
   - local lightcurve file input
   - nightly candidate manifest input
3. Emit provenance in the payload:
   - survey
   - broker endpoint version
   - exactness
   - expert availability reason
4. Add smoke tests using fixtures for:
   - ZTF path
   - Rubin path
   - missing-expert path

**Files likely touched**
- `scripts/score_new_object.py`
- `scripts/score_nightly.py`
- new single-object snapshot helper
- payload tests

**Outputs**
- real broker-fusion scoring for a new object
- tested payload parity with nightly scoring

**Done when**
- A new Rubin candidate can be scored end to end and the payload shows all available broker experts, not only local inference.

---

## ExecPlan R8 - Modernize photometry features for Rubin

**Goal**
Stop relying on a mostly g/r feature set when Rubin operation can require broader band handling.

**Why now**
The current feature extractor is still mostly ZTF-shaped even though it mentions LSST `fid=3`.

**Tasks**
1. Introduce a survey-aware band map registry.
2. Extend feature extraction to support Rubin filters without silently dropping information.
3. Separate:
   - survey-agnostic temporal features
   - survey-specific color features
   - missing-band indicators
4. Update local-expert wrappers to use the same survey-aware band map.
5. Add unit tests for ZTF and Rubin filter mappings.

**Files likely touched**
- `src/debass_meta/features/lightcurve.py`
- `src/debass_meta/experts/local/parsnip.py`
- `src/debass_meta/experts/local/supernnova.py`
- lightcurve and expert tests

**Outputs**
- survey-aware lightcurve features
- band-mapping tests

**Done when**
- Rubin photometry is handled intentionally rather than as an accidental near-match to ZTF g/r.

---

## ExecPlan R9 - Rollout and decision gates

**Goal**
Move safely from bootstrap to Rubin production.

**Why last**
We need operational gates that prevent premature science claims.

**Tasks**
1. Add an explicit rollout ladder:
   - Stage A: ZTF bootstrap only
   - Stage B: Rubin shadow mode
   - Stage C: Rubin human-reviewed candidate lists
   - Stage D: Rubin production candidate ranking
2. Define minimum gates for Stage C and Stage D:
   - archived nightly broker evidence exists
   - survey-aware schemas live
   - complete new-object scoring path live
   - Rubin cross-check report generated for consecutive nights
   - at least one Rubin strong-truth evaluation slice exists
3. Add a simple status dashboard or markdown summary command to report current stage.

**Outputs**
- `docs/rubin_rollout.md`
- `reports/summary/rubin_readiness.md`

**Done when**
- The repo can say exactly which Rubin stage it is in and why.

---

## Suggested Execution Order

1. `R0` Freeze operating claim.
2. `R1` Make survey first-class.
3. `R2` Split adapters by survey/API surface.
4. `R3` Introduce canonical candidate identity.
5. `R4` Start Rubin shadow archival.
6. `R5` Make training provenance explicit.
7. `R6` Add Rubin cross-check and calibration reporting.
8. `R7` Complete new-object broker-fusion scoring.
9. `R8` Modernize photometry features.
10. `R9` Define rollout gates.

## Non-Goals For This Plan

- Do not replace the trust architecture with a single fused classifier.
- Do not silently mix Rubin and ZTF metrics into one headline number.
- Do not force Lasair Sherlock into a probability interpretation.
- Do not claim Rubin-ready calibration until Rubin holdout evaluation exists.

## Decision Rule

Until `R4`, `R5`, `R6`, and `R7` are complete, the honest project description is:

`DEBASS is a ZTF-bootstrapped, Rubin-oriented trust-aware broker fusion system.`

After those are complete, and only after Rubin holdout evaluation exists, the claim can move to:

`DEBASS provides Rubin cross-checked broker confidence for new transient candidates.`
