# Science Contract

DEBASS is a trust-aware early-epoch transient system.

## Primary Output

The primary science product is `expert_confidence`: calibrated per-expert trust at a given object epoch.

For an object at `(object_id, n_det, alert_jd)`, the system should answer:

- Which expert/classifier is available right now?
- Which expert is trustworthy right now?
- What evidence did that expert contribute?

## Secondary Output

The secondary product is a trust-weighted `p_follow_proxy` score for follow-up prioritization.

This score is useful, but it must remain downstream of the expert-confidence layer.

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
- `trust`: calibrated probability that the expert's mapped recommendation is correct or useful at this epoch.
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
