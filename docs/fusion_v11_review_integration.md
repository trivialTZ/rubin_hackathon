# fusion_v11 — Integration-Risk Review (design §7 ownership + pinned contracts)

Reviewer: integration-risk agent, 2026-07-04. Target: `docs/fusion_v11_design.md`.
Every claim below was verified against the working tree at review time (branch `main`,
venv sklearn 1.6.1 / torch 2.6.0-cpu, 320 tests collected).

Verdict: the P1–P6 file lists are **pairwise disjoint as written** (verified file-by-file;
no file appears in two packages). They are **not complete**: five gaps would force a
package to touch a file it does not own, and two guards (G5b, G6-at-build) are not
implementable as literally specified. Amendments A1–A12 below; A1, A3, A4/A6, A5, A7
are blocking (must be folded into the spec before parallel implementation starts).

---

## 1. Amendments (A1–A12)

### A1 [BLOCKING — P5 gap] `src/debass_meta/experts/local/__init__.py` is unowned but required
`ALL_LOCAL_EXPERTS` lives in `src/debass_meta/experts/local/__init__.py:10-19`. Silver
events for a new expert key exist only if a `LocalExpert` subclass with `name = "seq_v11"`
is exported there: `scripts/local_infer.py:196-202` builds its pool from
`ALL_LOCAL_EXPERTS` and filters by `cls.name == args.expert` (this is exactly how the v10
chain injects seq silver: `jobs/run_fusion_v10_expert.sh` runs
`local_infer.py --expert seq_v9`). Without the export, `proj__seq_v11__*` columns never
fire and the registry entry is dead.
**Re-assignment**: add `src/debass_meta/experts/local/__init__.py` to P5.
`scripts/local_infer.py` and `scripts/collect_epoch_history.py` need NO edits
(name-based selection; `_EXPERT_RUNNERS` in collect_epoch_history has no seq entry and
v10 never needed one).
Also pin (P5 already owns `projectors/base.py`): `_dispatch_projector`
(`projectors/base.py:163-216`) is **per-key if-chains, not auto-discovered** — the
`seq_v11` key needs an explicit branch (`if expert_key == "seq_v11": from .local_seq_v9
import project_events`). Auto-discovery covers gold/helpfulness/trust
(`build_helpfulness_fusion.py:128,170` iterates `ALL_EXPERT_KEYS`), not dispatch.

### A2 [P1] `scripts/fetch_tns_bulk.py` duplicates an existing script
`scripts/download_tns_bulk.py` already downloads the same
`tns_public_objects.csv.zip` (API-key/user-agent via `access/tns.py:load_tns_credentials`),
and `scripts/crossmatch_tns.py` already parses it (`_load_tns_bulk_csv`, line 131:
optional timestamp header row, `"name"`→`objname` normalization).
**Re-assignment (minimal)**: P1 owns and **extends `scripts/download_tns_bulk.py`**
(caching, resume, `data/truth/tns_public.parquet` emit) instead of creating
`fetch_tns_bulk.py`; parsing reuses `crossmatch_tns._load_tns_bulk_csv` **by import**
(`crossmatch_tns.py` itself stays read-only for everyone). If a new filename is preferred
anyway, it must wrap — not re-implement — the existing downloader.

### A3 [BLOCKING — P1↔P2 interface] G6-at-build has no owner
G6 requires "benchmark manifest ∩ (train ∪ cal) = ∅ asserted **at build**". The split
builder is `build_split_manifest` inside `scripts/build_snapshots_fusion.py` (P2), which
today only knows the locked v6e2 split via `--trust-metadata`. Nobody's file list wires
the LSST-live manifest into it.
**Pin**: P2 adds `--lsst-live-locked` to `build_snapshots_fusion.py` (default
`data/gold/lsst_live_locked_test.json`; missing file → no-op, mirroring
`--association-csv` behaviour). Ids listed there are forced to TEST (or quarantined out
of train/cal) and asserted disjoint from train∪cal. P1 pins the manifest schema:
`{"test_ids": [...], "frozen_utc": "...", "policy": "...", "source": "..."}`.
**Constraint**: implement inside `build_snapshots_fusion.py`, NOT in
`src/debass_meta/models/splitters.py` — splitters.py (`_association_clusters`,
`group_train_cal_test_split`) is shared with the legacy pipeline and owned by no package.

### A4 [BLOCKING — P2] G5b "byte-identical" is unachievable as file bytes — redefine as value-identity
v11 gold adds columns (`n_pos_det` + neg features), so the parquet file cannot be
byte-identical. Also the reference locked gold
(`data/gold/object_epoch_snapshots_fusion_v10.parquet`) exists **only on SCC** — local
`data/gold/` has v8/v9c golds only.
**Pin**: G5b = on ZTF-survey rows of the reference gold, (a) the row multiset keyed by
`object_id × n_det` is identical, and (b) every v10-era column (base-51 + EXT + traj +
`proj__`/`avail__`/`exact__`) is exactly equal. `scripts/assert_locked_gold_identity.py`
takes `--reference` and `--rebuilt`, compares column-wise, and accepts a
`--reference-hash-manifest` fallback (per-column/per-row-block hashes) so the guard can
run locally where the SCC parquet is absent (skipif convention, §4). Never compare file
bytes.

### A5 [BLOCKING — P2, coordinate P5] The negative-det mechanism is pinned — it is the all-negative FALLBACK, in three lockstep copies
Verified: the positive filter exists everywhere; negatives enter `n_det` only through the
explicit fallback `if not pos_dets: pos_dets = ndets` in **three** places that the code
itself documents as lockstep:
- `src/debass_meta/features/lightcurve.py:258` (`extract_features_at_each_epoch`)
- `scripts/build_snapshots_fusion.py:208-210` (`_truncated_detection_lists` — docstring:
  "Replicates the preprocessing inside … extract_features_at_each_epoch", parity is
  asserted per-row in `_extract_object_rows`, lines 244-256)
- `src/debass_meta/features/sequence_dataset.py:72-74`
  (`truncated_positive_detections` — docstring: "Must stay in lockstep with
  `_truncated_detection_lists`") — **P5-owned**.
So the design's "investigate the builder's loader bypassing is_positive" is answered: no
loader bypass; fix the fallback. Two consequences:
1. P2 and P5 must change their copies in the same sense or explicitly diverge and update
   the parity assert in `_extract_object_rows`. **Recommendation**: P2 exports ONE
   canonical truncation helper from `features/lightcurve.py`; P5 imports it (dependency
   direction P5→P2 keeps ownership clean).
2. Removing the fallback deletes **all rows** for all-negative objects. If any locked
   v6e2/v10 ZTF object is all-negative, `build_split_manifest`'s hard assert
   ("original locked-split objects missing from the fusion snapshot", lines 519-525)
   fails the build. **Measure on SCC before flipping the contract**; if hits exist, keep
   the fallback per-survey (ZTF) or quarantine-list those ids in the guard script.

### A6 [BLOCKING — P2] Do not feed negative detections into the base-51 extractor
`extract_features` computes `n_det = float(len(ndets))` (lightcurve.py:191 — counts ALL
dicts passed in, including NaN-mag) and `mean_quality` over every det with finite
quality (lines 239-243). If the v11 truncation window (positives + negatives with
t ≤ t_Nth-positive) is handed to the base extractor, `n_det` and `mean_quality` change on
any ZTF object with `isdiffpos='f'` epochs → value-identity G5b (A4) breaks on ZTF too.
**Pin the implementation**: base-51 + EXT keep consuming the positives-only prefix
(unchanged); the new neg features are computed in `_extract_object_rows` from the
negatives-included window, following the existing EXT pattern — a separate
`NEG_FEATURE_NAMES` list (may live in `features/lightcurve.py`, P2-owned) consumed by
the builder, NOT a literal extension of what `extract_features` must emit (it cannot
compute `n_det_neg` from a positives-only list). The CLAUDE.md "add to FEATURE_NAMES +
DEFAULT_FEATURES" rule is satisfied at the column-contract level: gold emits the new
columns, `DEFAULT_FEATURES` (`models/early_meta.py:29`) is extended in sync; note the
fusion Stage-B does NOT read `DEFAULT_FEATURES` (it auto-discovers via
`multiclass_followup._numeric_feature_cols`), so the sync only protects the legacy
early_meta path — still do it.

### A7 [BLOCKING — P5] Sequence schema v11 must be versioned, not edited in place
`sequence_arrays` is a single shared function emitting
`SEQ_CONTINUOUS_DIM = len(SEQ_CONTINUOUS_FIELDS) = 9` columns
(`sequence_dataset.py:35-48`). The deployed v9/v10 artifacts persist `cont_dim=9` in
`config.json` (`SeqEncoderConfig.from_json`, restored at
`seq_classifier.py:278`) and are fed by `seq_v9.py:147` calling `sequence_arrays(...)`.
Changing SEQ_CONTINUOUS_FIELDS in place gives every existing artifact a (L,10+)-vs-9
shape mismatch at inference — the seq_v9 expert (and the v10 re-score) dies.
**Pin**: `sequence_arrays(dets, *, schema="v9")` with `"v11"` opt-in; the schema tag
rides in the artifact's `config.json` meta (written by `train_seq_classifier.py` — P5
owns it; `SeqClassifierArtifact.save/load` pass `meta` through untouched, so
`models/seq_classifier.py` and `models/seq_encoder.py` need **no edits** — keep them
read-only). `seq_v9.py` routes schema per artifact from `artifact.meta.get("seq_schema",
"v9")`. New env var `DEBASS_SEQ_V11_MODEL` for the seq_v11 expert;
`DEBASS_SEQ_V9_MODEL` semantics stay untouched (v10 job scripts export it —
`jobs/run_fusion_v10_expert.sh:51`).

### A8 [P4] score_fusion_v11 import mechanics + one missing interface method
- Import convention (verified working pattern, `scripts/export_lsst_candidates.py:14-15,30`):
  insert BOTH `Path(__file__).resolve().parents[1] / "src"` AND `parents[1]` (repo root)
  into `sys.path`, then `from scripts.score_fusion_v8 import ...`. Safe at module level:
  `score_fusion_v8.py` has no import-time side effects (argparse is entirely inside
  `main()`); tests also resolve it because `tests/conftest.py` inserts repo-root + src.
- The design pins `HierarchicalFollowup.fit/predict_proba/save/load` — **insufficient**:
  the v8 scorer calls BOTH `artifact.predict_proba_raw(df)` and
  `artifact.predict_proba(df)` (score_fusion_v8.py:340-345). Pin `predict_proba_raw`
  (uncalibrated composed ternary) into the P3 interface, plus the `survey` column routing
  convention (§3 below).

### A9 [P1] Harvest output must match two existing consumer schemas exactly
- Association CSV — consumed by `access/associations.py:load_lsst_ztf_associations`
  (lines 39-88): required columns `lsst_object_id`, `ztf_object_id`,
  `match_status == "matched"`, `sep_arcsec` (2″ default filter applied by the loader);
  optional `association_kind`, `association_source`, `match_count`. The existing
  `scripts/crossmatch_lsst_to_ztf.py` already emits this schema (plus
  `candidate_matches_json`) — reuse its helpers by import.
- Truth parquet — the exact 20-column schema of `data/truth/object_truth.parquet`
  (pinned in §3 with dtypes). RUNBOOK rule inherited: an LSST object missing from the
  truth parquet silently never enters gold.

### A10 [P6] Reuse, never edit, the shared train/score/eval scripts
`train_fusion_v11.py` imports from `scripts/train_fusion_v8.py` (read-only):
`load_split_manifest(path) -> (train_ids, cal_ids, test_ids, raw)`, `run_component_gates`,
`_macro_ovr_auc`, `_paired_delta_ci`, `ndet_bucket`, `class_index`; Stage-A via
`debass_meta.models.pooled_trust.train_pooled_trust`; helpfulness via CLI call to
`build_helpfulness_fusion.py` (auto-discovers seq_v11 from `ALL_EXPERT_KEYS`). If any
shared script needs an internal change, escalate — that is a spec change, not a P6 edit.
Chain scripts mirror v10 conventions: `qsub -terse -P pi-brout`, `-hold_jid` chaining,
`FUSION_V11_FORCE=1` idempotency + skip-markers, GPU stage `-l gpu_c=8.0` (never bare
`gpus=1`), artifact-pinning env vars exported in the job script.

### A11 [P3] Per-survey calibration constants
`MulticlassFollowupArtifact` routes rows by literal `survey` string to
`survey_calibrators` trained only when a survey has **≥150 cal objects**
(multiclass_followup.py:885-905); the design says "fall back when survey-cal n < 40".
Reconcile: make the threshold a `HierarchicalFollowup` constructor parameter (default
whatever the design gate decides), defined in `hierarchical_followup.py` —
`multiclass_followup.py` and `models/calibrate.py` stay read-only (import the calibrator
ladder, don't fork it).

### A12 [P1/P2/P6] RUNBOOK inheritance is a contract — see §5
Every scoring-path invocation in `jobs/refresh_lsst_live.sh` (P1),
`build_truth_lsst_live.py`'s cohort re-derivation (P1), the v11 gold build (P2/P6), and
the benchmark re-score (P6) must carry the §5 list. In particular
`refresh_lsst_live.sh` is a SCORING flow: empty locked-split JSON as `--trust-metadata`,
cohort-only `--labels` CSV (never default `data/labels.csv`), `--bts ""`,
`--no-lsst-weak`, never `--smoke`.

---

## 2. Ownership matrix verification (design §7)

Pairwise-disjointness: **PASS** — the 6 lists share no file. Existing-vs-new status:

| Package | Existing files edited | New files | Gaps found |
|---|---|---|---|
| P1 | `access/tns.py` | build_truth_lsst_live.py, harvest_ztf_lsst_associations.py, refresh_lsst_live.sh, test_truth_lsst_live.py | A2 (fetch_tns_bulk duplicates download_tns_bulk.py), A3 (manifest→builder interface), A9 (schemas) |
| P2 | features/lightcurve.py, ingest/gold.py, build_snapshots_fusion.py, models/early_meta.py | assert_locked_gold_identity.py, 2 tests | A3 (new flag), A4 (G5b redefinition), A5/A6 (fallback + window feeding), splitters.py must stay untouched |
| P3 | — | models/hierarchical_followup.py, test | A8 (predict_proba_raw), A11 (threshold param) |
| P4 | — | models/anchor_blend.py, score_fusion_v11.py, 2 tests | A8 (import mechanics) |
| P5 | sequence_dataset.py, train_seq_encoder.py, train_seq_classifier.py, experts/local/seq_v9.py, projectors/base.py, projectors/local_seq_v9.py | fetch_elasticc2.py, test_seq_v11.py | **A1 (experts/local/__init__.py missing from list)**, A7 (schema versioning; seq_classifier.py/seq_encoder.py stay read-only) |
| P6 | — | train_fusion_v11.py, 3+1 job scripts, smoke test | A10 (read-only reuse of train/score/eval_fusion_v8, pooled_trust, conformal, selection) |

Shared read-only modules no package may edit (any needed change = spec escalation):
`models/splitters.py`, `models/multiclass_followup.py`, `models/pooled_trust.py`,
`models/conformal.py`, `models/selection.py`, `models/calibrate.py`,
`models/seq_classifier.py`, `models/seq_encoder.py`, `scripts/train_fusion_v8.py`,
`scripts/score_fusion_v8.py`, `scripts/eval_fusion_v8.py`,
`scripts/build_helpfulness_fusion.py`, `scripts/crossmatch_tns.py`,
`scripts/crossmatch_lsst_to_ztf.py`, `scripts/local_infer.py`,
`features/lightcurve_ext.py`, `features/detection.py`, `access/associations.py`,
`tests/conftest.py`.

---

## 3. Pinned existing signatures (new code must match these exactly)

### 3.1 `scripts/build_snapshots_fusion.py`
- `main()` flags (all; v11 mode extends, never renames):
  `--lc-dir` (data/lightcurves) `--silver-dir` (data/silver) `--truth`
  (data/truth/object_truth.parquet) `--bts` (data/truth/ztf_bts.parquet; **falsy ""
  disables**) `--labels` (data/labels.csv; **ZTF-ids-only filter** via
  `_load_labels_ztf_ids`/`infer_identifier_kind`) `--trust-metadata`
  (models/trust/metadata.json) `--lsst-candidates` (data/lsst_candidates.csv)
  `--no-lsst-weak` `--seq-train-ids` `--association-csv`
  (data/crossmatch/lsst_to_ztf.csv; missing → per-object split) `--output`
  `--split-manifest` `--max-n-det` (20) `--n-jobs` (8) `--seed` (42) `--smoke` (caps
  200 objects + .smoke suffix) `--limit` `--skip-traj` `--skip-experts` `--dp1`
  `--dp1-only` `--dp1-snapshots` `--dp1-lc-dir` `--dp1-output`.
- `build_fusion_snapshots(lc_dir, silver_dir, truth_path, bts_path, labels_path,
  trust_metadata_path, output_path, split_manifest_path, lsst_candidates_path,
  seq_train_ids_path, association_csv, max_n_det, n_jobs, seed, smoke, limit,
  skip_traj, skip_experts)` — all keyword.
- `build_split_manifest(snapshot_object_ids, truth_for, trust_metadata_path, *, seed,
  smoke, snapshot_path, seq_train_ids, association_map)`: locked ids verbatim
  (hard-assert all present unless smoke); new-would-be-test → train; association
  clusters + seq-train diversion (v10). The v11 `--lsst-live-locked` handling (A3) goes
  here.
- Truncation lockstep trio pinned in A5. EXT/traj imports are guarded try/except with
  loud stubs (lines 96-123) — the v11 neg-feature block should follow the same
  degrade-loudly pattern.

### 3.2 `scripts/score_fusion_v8.py` (P4 imports; v11 CLI mirrors)
Importable, side-effect-free at import: `ndet_bucket(n_det: np.ndarray) -> np.ndarray`;
`_latest_per_object(df) -> df`; `attach_trust_columns(df, trust_dir: Path) -> df`
(adds `q__<expert>` / `q_prior__<expert>`, NaN = expert absent);
`compute_fdr_thresholds(...)`; `trust_weighted_p_snia(df) -> np.ndarray`
(the anchor's Ia-axis building block, line 250). Module constants:
`CLASSES` fallback tuple, `GOALS`, `FDR_LABEL_QUALITIES = ("spectroscopic",
"tns_untyped")`, `LOCAL_PSNIA_COLS`, `DP1_CATALOG_COLS`, `GOAL_SCORE_COLS`.
CLI to mirror verbatim (+ new `--blend-dir`): `--snapshots --dp1 --dp1-catalog
--followup-dir --trust-dir --conformal --out --scores-dir --budgets --fdr-gamma
--fdr-n-det-max --split --no-priority --smoke --tag`. Scorer calls BOTH
`predict_proba_raw` and `predict_proba` on the followup artifact (A8); conformal via
`MondrianAPS.load(...).predict_sets(p_cal, strata)` with strata columns
`survey` (lowercased str), `n_det` (int), `n_det_bucket`.

### 3.3 `models/seq_classifier.py::SeqClassifierArtifact` (read-only)
Dataclass fields `classifier, norm_stats, temperature, meta`; `classes` property →
tuple from the classifier head (3-class v9c and 4-way v10 both valid);
`load(art_dir, *, device="cpu")`: `config.json` → `meta`,
`SeqEncoderConfig.from_json(meta["config"])` (cont_dim persisted per-artifact),
`torch.load(classifier.pt, weights_only=True)`, `load_norm_stats(norm_stats.json)`;
`save(art_dir)` writes the same three files + `model_version`;
`predict_proba_prefixes(cont_raw, bands, *, device) -> (L, n_classes)` applying
NormStats + temperature. Extra meta keys pass through save/load untouched — the v11
`seq_schema` tag rides here (A7).

### 3.4 `experts/local/seq_v9.py` artifact routing (P5 edits, keep semantics)
`_MODEL_DIR_CANDIDATES = ("models/seq_classifier_v10", "models/seq_classifier_v9")`;
env override `DEBASS_SEQ_V9_MODEL`; OOF routing via `fold_map.json`
`{"assignments": {object_id: k}}` → `fold_{k}/` subartifact (the model that excluded
fold k), root artifact for unmapped ids; **corrupt fold_map.json raises RuntimeError by
design** (never silently skip OOF); missing fold dir → root fallback with
`fold_route="missing_fold_{k}_fallback_full"`. Emits canonical ternary
`class_probabilities` + `p4_<class>` extras for 4-way heads; `available=False` when the
artifact is missing (repo convention). seq_v11 must replicate ALL of this (new class,
new env var, schema routing per A7).

### 3.5 Expert registry / projector contract (CLAUDE.md critical contract, verified)
`EXPERT_REGISTRY: dict[key, (scope∈{"ztf","lsst","any"}, projector_module)]`
(`projectors/base.py:11-58`); `ALL_EXPERT_KEYS = list(EXPERT_REGISTRY)` auto-updates
gold (`build_snapshots_fusion` imports `ALL_EXPERT_KEYS, sanitize_expert_key`),
helpfulness (`build_helpfulness_fusion.py:128,170`) and trust auto-discovery.
NOT auto: `_dispatch_projector`'s per-key branch (A1). Projector fn signature:
`project_events(expert_key, events: list[dict]) -> dict` returning
`summarize_ternary(p_snia, p_nonIa_snlike, p_other)` output (renormalized + top1/margin/
entropy) or `{"prediction_type": "unknown", "reason": ...}`. Column prefix:
`proj__{key.replace('/','__')}__*`. seq_v11 is NOT an SN-filter — do not add to
`models/expert_trust.py:SN_FILTER_EXPERTS`.

### 3.6 Feature-name contracts
`features/lightcurve.py:FEATURE_NAMES` — exactly 51 names (counts 7, temporal 2,
all-band 8, per-band 24, colours 8, quality 1, survey 1);
`extract_features` returns every name (NaN-filled); `n_det = len(passed dets)` and
`mean_quality` over all passed dets — the A6 hazard. `models/early_meta.py:29
DEFAULT_FEATURES` must be extended in sync (legacy path only; fusion Stage-B
auto-discovers numeric cols and asserts no label leakage via
`multiclass_followup._assert_no_label_features`).

### 3.7 Truth parquet schema (P1 must emit byte-compatible)
20 columns of `data/truth/object_truth.parquet`, dtypes pinned:
`object_id(object) final_class_ternary(object) follow_proxy(int64) label_source(object)
label_quality(object) bts_type(object) tns_name(object) redshift(object)
final_class_raw(object) truth_timestamp(float64) tns_prefix(object) tns_type(object)
tns_has_spectra(object) tns_redshift(float64) tns_ra(float64) tns_dec(float64)
tns_discovery_date(object) consensus_experts(float64) consensus_n_agree(float64)
consensus_n_total(float64)`. Reference generator with correct dtypes:
`data/live_eval_20260704/tools/make_truth_tns.py`.

### 3.8 Association CSV schema (P1 harvest output)
Per `load_lsst_ztf_associations`: `lsst_object_id`, `ztf_object_id`,
`match_status` (only `"matched"` rows load), `sep_arcsec` (loader filters > 2″),
optional `association_kind` (default `position_crossmatch`), `association_source`
(default `alerce_conesearch`), `match_count`.

### 3.9 `MulticlassFollowupArtifact` conventions P3 must mirror
`CLASSES = ("snia", "nonIa_snlike", "other")`; `_SPEC_QUALITIES = ("spectroscopic",
"tns_untyped")`; `_prepare_frame` NaN-fills missing feature columns (the live-eval
relied on this — keep it in the hierarchical head); per-survey calibrators keyed by
literal survey string, ≥150-cal-object gate (A11); `save/load(dir)` with
`metadata.json` carrying `feature_cols` per head.

---

## 4. Locked-artifact hazards

| Artifact | Location | Must remain | Protecting assert today | v11 addition |
|---|---|---|---|---|
| Locked v6e2 split | `models/trust/metadata.json` (LOCAL: 1152/385/383; **SCC copy differs** — larger) | byte-identical, read-only | `build_split_manifest` hard-asserts disjointness + all-ids-present; docstring guarantees test ⊆ locked test | G1/G5b must key off the environment's file, never hardcoded counts |
| Locked ZTF gold | `data/gold/object_epoch_snapshots_fusion_v10.parquet` (**SCC only**; local has v8/v9c) | value-identical on v10 columns/rows (A4) | none beyond the split assert | `assert_locked_gold_identity.py` (P2) + hash-manifest fallback |
| v10 followup/trust/conformal | `models/{followup,trust,conformal}_fusion_v10/` (local + SCC) | loadable | — | needed for the §6 apples-to-apples re-score; pickled with sklearn 1.7.2 → `InconsistentVersionWarning` under venv 1.6.1 is benign; **never upgrade sklearn** |
| v10 seq artifacts | `models/seq_classifier_v10{,_lsst_adapt,_ztf}/`, `models/seq_encoder_v10/` | loadable + OOF-routable | `fold_map.json` corrupt → RuntimeError (by design); `run_fusion_v10_expert.sh` hard-checks `classifier.pt` + `fold_map.json` | A7 schema versioning keeps `sequence_arrays` 9-dim for them |
| v10 split manifest | `data/gold/split_fusion_v10.json` (local + SCC) | read-only | consumed verbatim by score/eval | |
| Live-eval evidence | `data/live_eval_20260704/**` | frozen | — | P1's cleaned-truth re-derivation writes NEW `*_cleaned.parquet`, never overwrites |
| New LSST-live benchmark | `data/gold/lsst_live_locked_test.json` | frozen after P1 emits | — | G6 assert at build (A3) AND at train (P6) |
| Weak-label source | `data/lsst_candidates.csv` (exists locally) | read-only input | — | v11 re-maps it at the is_sn level only — no file edit |

v11 writes ONLY `*_v11`-suffixed model/gold/report paths. Job scripts must not export
`DEBASS_SEQ_V9_MODEL` pointing at v11 artifacts (schema mismatch, A7) — use the new var.

---

## 5. Test-suite reality

- `pytest --collect-only -q` → **"320 tests collected in 1.64s"** (local venv).
- Per campaign memory: 320 green locally; **318/320 on SCC** — 2 pre-existing
  data-dependent failures, NOT bugs to fix in v11:
  1. `tests/test_pooled_trust.py::test_trust_target_parity_real_base_rates` — hardcodes
     base rates `{fink/snn: 0.290, fink/rf_ia: 0.652, supernnova: 0.478}` against the
     machine's real `data/gold/expert_helpfulness.parquet` (skipped where the file is
     absent; fails where it differs, i.e. SCC).
  2. A DP1-cache test asserting a `quality` field in the cached DP1 lightcurves
     (SCC cache predates the field).
- Conventions new tests MUST follow (verified in `tests/conftest.py` and
  `test_snapshots_fusion_g5.py`): synthetic/self-contained; anything touching real data
  guarded with `@pytest.mark.skipif(not PATH.exists(), ...)`; no network; import
  scripts either as `from scripts.X import ...` (conftest inserts repo-root + src and
  even evicts a shadowing `scripts` module) or via the `_load_script` importlib pattern;
  leakage tests follow the G5 style ("extending a lightcurve with future detections must
  NOT change features at fixed n_det") — `tests/test_gold_positive_only.py` should add
  the v11 variant: appending future NEGATIVE detections must not change any feature at
  fixed n_pos_det.
- New test files named in §7 (6 files) collide with nothing existing.

---

## 6. RUNBOOK gotchas P1/P2/P6 inherit (from `data/live_eval_20260704/RUNBOOK.md`, all smoke-verified)

1. **Environment**: venv only (`~/.venvs/debass_py313`); hand-written python must call
   `load_dotenv("/Users/tz/Documents/GitHub/rubin_hackathon/.env")` with the explicit
   path; `normalize_lightcurve/normalize_detection` survey arg is `"LSST"` (uppercase)
   or `"auto"` — lowercase silently routes to the ZTF normalizer and yields all-NaN.
2. **`--labels` is ZTF-only** (`_load_labels_ztf_ids` discards LSST diaObjectIds):
   LSST ids enter the object list ONLY as (lc-dir `*.json` stems) ∩ (truth parquet
   object_ids). ⇒ P1 must write a truth row for EVERY LSST object wanted in gold, even
   `label_quality='tns_untyped'` with ternary None; after any build, assert snapshot
   object count == cohort count (objects missing from truth drop silently).
3. **Locked-split JSON**: eval/scoring cohorts pass an EMPTY locked split as
   `--trust-metadata` (`{"train_ids": [], "cal_ids": [], "test_ids": []}`) — the default
   `models/trust/metadata.json` hard-fails on eval-only cohorts. TRAINING builds pass
   the real metadata.json. `refresh_lsst_live.sh` (P1) and the benchmark re-score (P6)
   are scoring flows → empty split + cohort-only `--labels` CSV (never default
   `data/labels.csv`).
4. **`--bts ""`** disables ZTF BTS fallback truth (falsy check in `main()`);
   **`--no-lsst-weak`** mandatory on eval cohorts (training-era weak labels must not
   enter); **never `--smoke`** at scale (caps ~200 objects in BOTH
   `build_snapshots_fusion.py` and `score_fusion_v8.py`); `--skip-traj` is safe
   (scorers NaN-fill).
5. **Fetch/backfill discipline**: `fetch_lightcurves.py --max-epochs 20` default matches
   the builder's `--max-n-det 20` — raise both or neither; chunk ~50 ids/fetch and
   ~40 ids/backfill at `--parallel 8`; bronze files are timestamped so chunks never
   collide; run `normalize.py` ONCE after ALL backfill chunks (whole-dir rewrite).
6. **All-negative-flux objects** (~50% of live LSST): all base+EXT LC features NaN at
   every epoch; they score on experts alone. Tag them (`lc_all_nan = mag_mean.isna()` at
   max n_det) and count per cohort — directly feeds G2's denominators (spec-Ia cal rows
   can be all-NaN) and P2's positive-only fix (A5).
7. **Calibrator routing**: v9c/v10 followup artifacts carry per-survey calibrators for
   ZTF only → all LSST rows use the global calibrator (expected, not a bug). sklearn
   `InconsistentVersionWarning` on v10 unpickle is benign — never "fix" by upgrading.
8. **Expert availability on live LSST**: `fink_lsst/early_snia` needs ≥7 epochs (mostly
   absent at n_det 3-5 — the anchor's Ia-capable pool must expect this and fall back to
   the cal base rate as designed); local rerun experts have no events unless
   `local_infer.py`/`collect_epoch_history.py` ran; Fink ZTF outage → `fink/*` NaN;
   `pitt_google/antares/ampel` report "0 fields … unavailable" — normal.
9. **Truth-side vs feature-side**: `fink_lsst/crossmatch` events are truth-side only and
   must never become gold feature columns (verified in smoke; keep the post-build sanity
   check: no `xm_`/`crossmatch` feature columns). Lasair `crossmatch_tns`
   (watchlist wl_id=46) is POSITIONAL — the stale-label trap P1's epoch-aware crossmatch
   exists to fix; apply the same discovery-epoch window to the Lasair-sourced names.

---

## 7. Bottom line

Disjointness holds; completeness does not. Fold A1 (P5 owns
`experts/local/__init__.py`), A3 (builder `--lsst-live-locked` flag + manifest schema),
A4+A6 (G5b as value-identity + positives-only feeding of the base extractor), A5
(fallback is the mechanism; three lockstep copies; SCC all-negative census before
flipping), and A7 (versioned sequence schema) into the spec before parallel work starts.
Everything else (A2, A8–A12) is a wording/pinning fix that prevents drift but does not
change the architecture.
