# v9c Status for Hack-Day Teammates

## Current Meaning of v9c

`fusion_v9c` is not the frozen-embedding v9 arm.  It promotes the sequence
model into a registered local expert:

- expert key: `seq_v9`
- model artifact: `models/seq_classifier_v9/`
- local expert wrapper: `src/debass_meta/experts/local/seq_v9.py`
- projector: `src/debass_meta/projectors/local_seq_v9.py`
- silver output: `data/silver/local_expert_outputs/seq_v9/part-latest.parquet`
- gold snapshots: `data/gold/object_epoch_snapshots_fusion_v9c*.parquet`
- reports: `reports/fusion_v9c/` and `reports/metrics/fusion_v9c_train.json`

The classifier emits per-prefix ternary probabilities from raw lightcurves. It
is then treated like any other expert: projected into gold, assigned a trust
head, and gated into or out of Stage B.

## Local Facts Already Present

- Local v9c gold snapshot: 8,668 objects, 137,281 epoch rows.
- Local survey coverage: 100% ZTF.
- Local object labels: 3,853 spectroscopic, 4,744 weak, 71 demoted
  BTS-unclassified.
- `seq_v9` silver coverage: 137,281 epoch rows.
- `seq_v9` trust head: calibrated test AUC 0.868 on 5,529 test rows.
- Local gate: `seq_v9_expert` dropped on cal, delta macro-AUC -0.008
  with CI crossing zero.

Interpretation: locally, `seq_v9` is a useful standalone expert and the
best-single comparator in several tables, but the gated fusion stack did not
keep it at this data scale.

## LSST / SCC Status

Local checkout has `data/lsst_candidates.csv` with 3,998 LSST weak-label
candidates:

- SN stamp -> `nonIa_snlike`
- AGN / VS / asteroid / bogus -> `other`
- source remains weak/self-labeled, not final science truth

Locally, those LSST objects do not enter the v9c gold snapshot because their
lightcurves are not present in this local cache. The SCC run is the real LSST
test path.

Latest user-provided SCC log summary on 2026-06-12:

- pretrain + classifier fine-tune completed;
- frozen-embedding arm completed and was dropped again;
- v9c job was running during gold rebuild at 10,000 / 12,772 objects, which
  indicates the 3,998 LSST weak-label objects were entering gold on SCC.

Live status command:

```bash
ssh scc 'cd /project/pi-brout/rubin_hackathon && qstat -u tztang; tail -30 logs/fusion_v9c.qsub.out'
```

This machine could not independently verify SCC because SSH requested
interactive authentication.

## New Label Workflow

1. Put teammate labels in `incoming_new_labels.csv`.
2. Run:

```bash
~/.venvs/debass_py313/bin/python hackday/v9c_label_prep/prepare_v9c_label_workspace.py
```

3. Review:

- `generated/new_label_merge_plan.csv`
- `generated/labeling_queue.csv`
- `generated/object_truth_with_new_labels.csv`
- `generated/labels_with_new_objects.csv`

4. Only after review, use the scratch rebuild commands in `README.md`.

Important: labels for locked test objects should be treated as audit updates
unless the team deliberately defines and registers a new split.

