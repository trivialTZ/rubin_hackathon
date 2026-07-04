# v9c Hack-Day Label Prep

This folder is intentionally separate from the live `data/`, `models/`, and
`src/` v9/v9c paths.  It is for preparing new labels with teammates without
mutating the current pipeline artifacts.

## What v9c Already Is

`fusion_v9c` registers the sequence model as a real local expert named
`seq_v9`.

- `models/seq_classifier_v9/`: fine-tuned ternary sequence classifier.
- `data/silver/local_expert_outputs/seq_v9/`: per-epoch `seq_v9` probabilities.
- `data/gold/object_epoch_snapshots_fusion_v9c*.parquet`: gold snapshots with
  `seq_v9` expert projections/trust columns.
- `reports/metrics/fusion_v9c_train.json`: local gate/training ledger.
- `reports/fusion_v9c/`: local evaluation tables.

Local v9c is still effectively ZTF-only.  The SCC chain is the run that should
bring the 3,998 LSST weak-label candidates into gold, provided their lightcurves
exist on SCC.

## Input File for New Labels

Fill `incoming_new_labels.csv`.

Required columns:

- `object_id`: ZTF name or Rubin/LSST diaObjectId.
- `final_class_ternary`: one of `snia`, `nonIa_snlike`, `other`.
- `label_quality`: usually `spectroscopic`; accepted values are
  `spectroscopic`, `strong`, `consensus`, `weak`, `context`.
- `label_source`: short provenance string, for example `hackday_spec`,
  `tns_spectroscopic`, or `team_manual_review`.

Useful optional columns:

- `final_class_raw`: original class string, such as `SN Ia` or `SN II`.
- `tns_name`, `redshift`, `ra`, `dec`, `source_ref`, `notes`.

Do not use a generic `SN` label as final truth.  If a source only says `SN`
without subtype, leave the row out or mark it for follow-up in `notes`.

## Build the Isolated Prep Artifacts

From the repo root:

```bash
~/.venvs/debass_py313/bin/python hackday/v9c_label_prep/prepare_v9c_label_workspace.py
```

With labels filled in:

```bash
~/.venvs/debass_py313/bin/python hackday/v9c_label_prep/prepare_v9c_label_workspace.py \
  --new-labels hackday/v9c_label_prep/incoming_new_labels.csv
```

Outputs are written only under `hackday/v9c_label_prep/generated/`.

Key outputs:

- `v9c_status_summary.json`: current local artifact summary.
- `current_v9c_inventory.csv`: one row per current v9c object.
- `labeling_queue.csv`: suggested objects to discuss or upgrade.
- `lsst_candidates_manifest.csv`: LSST weak-label candidate status locally.
- `new_label_merge_plan.csv`: validation/merge audit for incoming labels.
- `object_truth_with_new_labels.csv/parquet`: scratch truth table with incoming
  labels applied.
- `labels_with_new_objects.csv`: scratch labels list that includes incoming
  objects, useful for local inference commands.

## Scratch Rebuild Command

After `object_truth_with_new_labels.parquet` exists, a no-touch rebuild can be
run into this folder:

```bash
~/.venvs/debass_py313/bin/python scripts/build_snapshots_fusion.py \
  --truth hackday/v9c_label_prep/generated/object_truth_with_new_labels.parquet \
  --labels hackday/v9c_label_prep/generated/labels_with_new_objects.csv \
  --output hackday/v9c_label_prep/generated/object_epoch_snapshots_v9c_newlabels.parquet \
  --split-manifest hackday/v9c_label_prep/generated/split_v9c_newlabels.json \
  --n-jobs 8
```

For a quick structural smoke test:

```bash
~/.venvs/debass_py313/bin/python scripts/build_snapshots_fusion.py \
  --truth hackday/v9c_label_prep/generated/object_truth_with_new_labels.parquet \
  --labels hackday/v9c_label_prep/generated/labels_with_new_objects.csv \
  --output hackday/v9c_label_prep/generated/object_epoch_snapshots_v9c_newlabels.smoke.parquet \
  --split-manifest hackday/v9c_label_prep/generated/split_v9c_newlabels.smoke.json \
  --limit 200 --skip-traj --skip-experts --n-jobs 4
```

These commands do not overwrite live `data/gold/*v9c*` artifacts.

