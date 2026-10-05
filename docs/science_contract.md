# Science Contract

metaDEBASS is a trust-aware early-epoch transient system.

## Primary Output

Revised 2026-10-04 (`docs/fusion_v13g_plan.md`, "Product decision"). The primary science product is a calibrated
P(SN) at a given object epoch, for ranking follow-up, reported with its input regime (`serving_regime`,
`n_broker_inputs`, `n_local_inputs`). P(SN Ia) is reported only where an Ia model beats chance on held-out data
(ZTF, not Rubin). On Rubin, P(SN) is calibrated to the LSST training object mix (about 4% SNe, mostly
catalogue-confirmed stars, variables and AGN, many of which a broker called SN-like). Its level is therefore
conservative: a typical spectroscopic SN scores about 0.5 (fusion v13j, `docs/fusion_v13g_plan.md`). Rank by it,
and read the level as evidence against that mix.

For an object at `(object_id, n_det, alert_jd)`, the system should answer:

- How likely is this to be a supernova, given everything available right now?
- Which experts were available, and what did each contribute?

## Secondary Output

Per-expert `call_trust`: the calibrated probability that this expert's SN-vs-not call is correct at this epoch, one
definition for every expert (`src/debass_meta/models/call_trust.py`). It is a diagnostic: it does not feed the heads,
and it is shown for an expert only where its reliability has been checked on held-out Rubin data. The `q__` trust
readouts that feed the heads are model internals, not a product (their target differs between experts).

Before 2026-10-04 the contract named per-expert trust (`expert_confidence`) as primary and a trust-weighted
`p_follow_proxy` as secondary. On Rubin that trust had two meanings (P(SN) for SN-filter experts, P(top class
correct) for the rest) and was miscalibrated on the explorer population, and the follow-up target (SN Ia) is at
chance.

## Phase-1 Expert Keys

- `fink/snn`
- `fink/rf_ia`
- `parsnip`
- `supernnova`
- `alerce/lc_classifier_transient`
- `alerce/stamp_classifier`
- `lasair/sherlock`

## Terms

- `expert`: one classifier or context source, not a whole broker.
- `trust` (`q__<expert>`): a Stage-A readout that feeds the heads; P(object is SN) for SN-filter experts, P(top class
  correct) for the rest.
- `call_trust`: calibrated probability that the expert's SN-vs-not call is correct at this epoch (all experts).
- `temporal_exactness`: whether the expert evidence is truly safe for historical epoch training.
- `follow_proxy`: temporary follow-up target, currently `1{final_class_ternary == 'snia'}` unless replaced with operational follow-up truth.

## Hard Guards

- Preserve the no-future-leakage light-curve slicing.
- Preserve event-level expert timing through silver and gold.
- Do not treat broker-derived weak labels as final science truth.
- Do not train trust heads on `latest_object_unsafe` historical ALeRCE rows by default.
- Do not train the follow-up head on in-fold trust predictions.
- Do not claim calibration unless calibration and final evaluation use separate object splits.

## Data Policy

- `data/truth/object_truth.parquet` is the canonical truth layer.
- `data/silver/broker_events.parquet` is the canonical event-level broker layer.
- `data/gold/object_epoch_snapshots.parquet` is the canonical object-epoch feature table.
- `data/gold/expert_helpfulness.parquet` is the canonical trust-target table.
- `scripts/train_early.py` and `models/early_meta.py` remain baseline-only.

## Output Contract

Each scored row should include:

- `object_id`
- `n_det`
- `alert_jd`
- `expert_confidence`
- `ensemble`

Each `expert_confidence[expert_key]` block should state:

- `available`
- `trust` or `null`
- `prediction_type`
- mapped class or context tag where applicable
- projected evidence where applicable
- exactness/provenance note where needed
