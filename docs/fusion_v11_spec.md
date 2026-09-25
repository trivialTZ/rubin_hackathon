# fusion_v11 — FINAL SPEC (2026-07-05)

Status: IMPLEMENTABLE. Merges `docs/fusion_v11_design.md` with the three reviews
(`fusion_v11_review_{data,ml,integration}.md`). Design decisions stand except where a reviewer
demonstrated infeasibility/error — every change is in §8. Architecture (hierarchical heads,
anchored blend, positive-only epochs, locked benchmarks) unchanged in outline. Line refs
verified by reviewers against the working tree.

## 0. Blocking-item resolutions (normative)

| # | Item | Resolution |
|---|---|---|
| B0 | Head-2 NO-GO on the CURRENT `object_truth.parquet`: 3,149/5,172 (61%) of `label_quality='spectroscopic'` nonIa rows are BTS-untyped (`bts_type='-'`) filler force-mapped to nonIa — TNS dump resolves 101 today (39 actually SN Ia, 25 not SNe). Reproduces the exact v11 bug class inside the ZTF spec corpus | P1 ships `scripts/rederive_spec_truth.py` (TNS-dump name join via `internal_names`; typed→corrected subtype; unresolved→NEW weak tier `label_quality='bts_untyped'`, ternary None, is_sn=1, Head-1 only) → `data/truth/object_truth_v11.parquet` + `data/truth/label_delta_v11.csv` (never overwrite). All v11 builds take `--truth object_truth_v11.parquet`. NEW guard G7 (§4): Head-2 rows with untyped provenance == 0. Locked ZTF-test labels re-derived under the fix, delta recorded, v10 re-scored on cleaned labels (G1 apples-to-apples). Inventory restated: ~7.5k typed + ~3.1k SN-candidate weak. LANDS BEFORE any Head-2 fit. `ztf_bts.parquet` is stale (content ≤2026-03-28) — refresh via TNS dump; BTS re-pull added to `refresh_lsst_live.sh` if BTS-only fields stay in use. |
| B1 | LSST spec TRAIN pool near-empty: only 47 epoch-valid typed rows → 42 unique TNS names in the whole stream (41 SN-like, 23 Ia, 34 ZTF-associated, median nDiaSources 3); the frozen benchmark (~150 spec) drew on the same pool; inflow ~3-10 epoch-valid spec objects/month (~66 typed TNS/30d at dec≤+13 × ~5% LSST capture) | (a) Head-2 and LSST Stage-A trust START ZTF-SPEC-ONLY — stated as fact, not risk. (b) Per-survey LSST calibration WILL hit its tiny-n fallback immediately (Platt, §3.3-P3). (c) §6 test-first policy REPLACED: post-freeze spec arrivals route by sha1(object_id) last hex digit — even→TEST (appended to benchmark manifest), odd→train/cal pool. Frozen ids never move. |
| B2 | Negative-det bug site | LSST `is_positive` keyed on `psfFlux > 0 and not isNegative` at `features/detection.py:163-165` (measured: 630/1060 negative-flux dets mis-flagged positive, 78% >3σ). ZTF isdiffpos rule (`detection.py:127-128`) untouched — never a uniform flux>0 rule (ZTF flux derived from magpsf, always >0). `detection.py` re-assigned to P2 for this fix only. |
| B3 | Truncation contract breaks G5b | Base-51+EXT keep POSITIVES-ONLY input on both surveys. Negatives-in-window feed ONLY the 5 new neg features. All-negative fallback survey-gated: KEPT for ZTF (11% = 336/3000 LCs fallback-dependent), REMOVED for LSST (0 positives ⇒ 0 epoch rows). SCC census of locked-object all-negativity ships in the build report before any further contract change. |
| B4 | G3 overclaimed / degenerate cells / circular α fit | G3 restated (§4). Anchor ε-clipped (1e-6 + renorm) before any log-loss. n_experts=0 bucket: α pinned 1, excluded from G3. Fixed a-priori bucket edges. α fit honesty-filtered (drop `broker_consensus` rows; drop rows whose label source is an anchor member) per `pooled_trust.py:369-375` precedent. |
| B5 | 150 catalog others double-booked | They stay in the frozen benchmark ONLY. `build_truth_lsst_live.py` harvests NEW catalog negatives (VSX/Gaia/SIMBAD basis, outside the frozen cohort) for train/cal, target ≥300 (deliverable, count reported). |
| B6 | G2 unevaluable | G2 evaluates on (OOF-train LSST spec-Ia rows) ∪ (LSST spec-tier cal rows) ∪ (assoc-spec rows); require n≥10; n<10 ⇒ loud FAIL status `G2_UNEVALUABLE` (nonzero exit; `--acknowledge-g2-unevaluable` lets the chain continue but stamps the report headline). Never a silent pass. |
| B7 | A1 seq_v11 dead without export | `src/debass_meta/experts/local/__init__.py` added to P5; explicit `seq_v11` branch in `_dispatch_projector` (`projectors/base.py:163-216` — per-key if-chain, not auto-discovered). |
| B8 | A7 in-place seq schema breaks v9/v10 artifacts | `sequence_arrays(dets, *, schema="v9")`, `"v11"` opt-in; `seq_schema` tag in artifact `config.json` meta; `seq_v9.py` routes per artifact via `artifact.meta.get("seq_schema","v9")`; new env `DEBASS_SEQ_V11_MODEL`; `DEBASS_SEQ_V9_MODEL` semantics untouched. |
| B9 | A4+A6 G5b byte-identity unimplementable | G5b = value-identity on v10 columns (§4). Base-51 extractor never sees negatives (`extract_features` counts `n_det=len(input)`, `mean_quality` over all input — lightcurve.py:191,239-243). |
| B10 | A5 fallback in 3 lockstep copies | P2 exports ONE canonical truncation helper from `features/lightcurve.py`; builder and P5's `sequence_dataset.py` import it (dependency P5→P2). Copies at `lightcurve.py:257-260`, `build_snapshots_fusion.py:207-212`, `sequence_dataset.py:72-74` all route through it. |
| B11 | A3 G6-at-build unowned | P2 adds `--lsst-live-locked` to `build_snapshots_fusion.py` (default `data/gold/lsst_live_locked_test.json`; missing file → no-op, mirroring `--association-csv`). Implemented in the builder, NOT `models/splitters.py`. P1 pins the manifest schema. |

## 1. Data layer (measured, 2026-07-05; all endpoints live-verified with .env creds)

**TNS bulk** — `POST https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip`,
form `api_key=<TNS_API_KEY>`, header `User-Agent: tns_marker{"tns_id":...,"type":"user","name":...}`
(existing `TNSCredentials.user_agent`). 13.6 MB, ~8 s, 200,145 rows × 23 cols
(`objid,name_prefix,name,ra,declination,...,type,discoverydate,internal_names,lastmodified`;
ra/declination decimal deg; 25,914 typed). CSV line 1 = snapshot timestamp → `skiprows=1`.
No Content-Length → NO byte-range resume; refresh = full master re-download + daily-diff upsert
`tns_public_objects_YYYYMMDD.csv.zip` (same auth; only that day's modified rows; keyed `objid`,
newest `lastmodified` wins; current day 404s until it closes). Hourly `_HH` files 404 — do not use.
`internal_names` contains ZTF ids → free TNS↔ZTF mapping.

**Lasair LSST** — `POST https://lasair.lsst.ac.uk/api/cone/` form `{ra,dec[deg],radius[arcsec],requestType}`,
header `Authorization: Token $LASAIR_LSST_TOKEN`. `/api/query/` with `tables="objects,crossmatch_tns"`
(implicit positional join) is the harvest mechanism — `watchlist_hits` wl_id=46 DOES NOT EXIST (struck).
Live fields: `objects.{firstDiaSourceMjdTai,lastDiaSourceMjdTai,nDiaSources,ra,decl}`;
`crossmatch_tns.{ra,decl,tns_name,tns_prefix,type,z,disc_int_name,disc_date,lastmodified_date,...}`.
The join is wider than 2″ (median 0.28″, 87%≤2″, max 2.97″) → compute sep client-side, cut ≤2″,
keep `sep_arcsec`. `crossmatch_tns` = match SEED only (footprint-filtered, 2,824 rows); TNS bulk is
the authoritative type/discoverydate source.

**Association yield (measured)** — ZTF→LSST conesearch direction FAILS (0/25 Lasair, 0/25 Fink,
2/233 name-match; extrapolated ~8 objects, none usable). Inverted harvest works: one paged
`objects,crossmatch_tns` query, parse ZTF ids from `disc_int_name` → 2,824 TNS-matched; 247 typed
(217 SN, 134 Ia); 177 typed ZTF-named (81 nDia≥3); 92 typed disc≥2025-10 (54 nDia≥3); 168 untyped
ZTF-named nDia≥3. Applying the design's own epoch cut keeps 47/247 typed rows → **42 names** — 81%
of Lasair's typed crossmatches are epoch-stale (live proof the §3.6 epoch filter is necessary).

**Fink LSST xm backup** — `POST https://api.lsst.fink-portal.org/api/v1/sources` json `{diaObjectId}`,
no auth; per-alert `f:xm_tns_{fullname,type,redshift}`, `f:xm_simbad_otype`, `f:xm_legacydr8_*`,
`f:xm_gaiadr3_*`, `f:xm_mangrove_*`, `f:xm_vsx_Type` + `f:xm_{gcvs_type,spicy_class,x3hsp_type,
x4lac_type}`. Sentinels `"Fail"` AND literal `"nan"` → missing. xm values are per-alert and
intermittently missing even after classification (SN 2026ctw: type absent on 178/470
post-classification alerts INCL. the latest) → truth extraction MUST aggregate across ALL alerts
(max-MJD non-nan); aggregated `xm_tns_fullname` is a name pointer only — backup, never primary.

**ELAsTiCC2** — `https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/`,
no auth, `Accept-Ranges: bytes`. Full tar 7.91 GB; per-class dirs `ELASTICC2_TRAIN_02_<MODEL>/`
(~40 HEAD + ~40 PHOT SNANA `.FITS.gz`, ~15 MB each; SNIa-SALT3 172,180 LCs; ~1.5M total, 36
classes). Plain `astropy.io.fits` reads them. Selective per-class fetch (~3-4 GB) is the default.

## 2. Architecture (deltas from design; unchanged parts by reference)

**2.1 Hierarchical head [P3]** — as designed (§3.1) with: head-1 per-survey is_sn LightGBM
(gate vs pooled decided on an is_sn-level LSST cal frame — weak+catalog, honesty-filtered;
preregistered default = POOLED when that frame is small); LSST head-1 negatives = newly harvested
catalog negatives (B5) + cleaned weak. Head-2 spec-only from `object_truth_v11.parquet` (B0), row
selection requires concrete subtype provenance (`tns_type` non-empty or `bts_type` ∉ {'-',''}) —
hard assert G7 in `fit`; `bts_untyped` rows are Head-1 only. Head-2 STARTS ZTF-ONLY (B1);
**drops `survey_is_lsst` + survey-degenerate columns by DEFAULT** (not a gate — the gate is
undecidable and the v10 harm was survey-flag-mediated); head-2-on-LSST is a routing decision with
default = per-survey constant base rate P(Ia|SN) (clipped [0.05,0.95]); shared head-2 applies to
LSST rows only if the LSST spec cal/OOF frame has n≥30 AND Spearman(p_snia,is_Ia) ≥ 0 AND 1-SE
improvement. Composition p=(P1·P2, P1·(1−P2), 1−P1); each head clipped to [1e-6,1−1e-6] BEFORE
compose (`multiclass_followup.py:624` precedent). Per-survey calibrators: isotonic when survey-cal
n ≥ threshold (constructor param `survey_cal_min`, default 40), else `PlattCalibrator`
(`pooled_trust.py:192-224`) — NOT temperature (cannot shift intercept; intercept bias was the v10
failure). Weights: `compute_base_weights` (w_qual × 1/n_rows × w_bts) + per-head class weights;
head-1 `weak_weight` gated ∈ {0.1, 0.3}. Provenance masking reused; on LSST head-1, EQUALIZE:
mask the alerce family on ALL LSST weak+context rows (not only alerce-labeled), or
`expert_dropout_augment` at matched rate; emit an availability-audit metric (dependence of
avail-pattern on y within LSST train rows) in the fit report. Head-2 covariate shift (isotonic fit
on true-SN rows, applied everywhere) accepted — document in module docstring. No per-class
recalibration of composed marginals; only simplex-preserving Dirichlet rung may follow composition.

**2.2 Anchored blend [P4]** — anchor = trust-weighted linear ternary pool of fired experts
(extends `trust_weighted_p_snia`, `score_fusion_v8.py:250`), ε-clipped 1e-6 + renormalized.
Only ternary-emitting experts enter; `lasair/sherlock` and `babamul` are excluded entirely.
Trust semantics documented: SN-filter q targets is_sn, ternary q targets is_topclass_correct —
both pool on the SN axis (v8 precedent); Ia|SN axis weights by Ia-capable experts' q only.

`IA_CAPABLE` (strings in `models/anchor_blend.py`; membership derived from projectors):
`fink/snn` (design omitted it — strongest ZTF Ia broker, `fink.py:17-28`), `fink/rf_ia`,
`fink_lsst/early_snia` (sparse: needs ≥7 epochs), `alerce/lc_classifier_transient`,
`alerce/lc_classifier_BHRF_forced_phot_transient`, `alerce/LC_classifier_ATAT_forced_phot(beta)`,
`pittgoogle/supernnova_lsst`, `pittgoogle/supernnova_ztf`, `parsnip`, `supernnova`, `alerce_lc`,
`salt3_chi2`, `lc_features_bv`, `seq_v9`, `seq_v11`, `antares/oracle`, `antares/superphot_plus`,
`ampel/parsnip_followme`, `oracle_lsst`.
NOT Ia-capable (constant Ia|SN ratio): all `alerce/stamp_*`, `alerce/lc_classifier_BHRF_forced_phot_top`,
`fink/slsn`, `fink_lsst/snn`, `ampel/snguess`, `pittgoogle/upsilon_lsst`, and **`fink_lsst/cats` OUT**
(row-conditionally capable — classes 21/31 carry a negative Ia claim, `fink_lsst.py:77-90`;
forfeited for v11, documented; row-level capability is future work).
Drift guards: train-time assert `IA_CAPABLE ⊆ ALL_EXPERT_KEYS`; unit test feeds synthetic events
through every NON-member projector and asserts constant Ia|SN ratio.
When no Ia-capable expert fired: Ia axis = per-survey cal base rate P(Ia|SN), clipped [0.05,0.95].

Blend: p_final = α·p_model + (1−α)·p_anchor; α ∈ {0,.25,.5,.75,1} per (survey × coverage-bucket),
cal log-loss + 1-SE preferring the anchor. Buckets FIXED a priori: n_experts_fired ∈ {0,1,2+} ×
lc_cov = frac-non-NaN-base-51 ∈ {<0.25, ≥0.25}. n_experts=0 ⇒ α=1 (no anchor; excluded from G3).
Per-cell n_min=50; fallback ladder cell → survey → global → (α=0 if anchor exists else 1).
α fit honesty-filtered (B4). Scorer emits p_* (blended, deployed), p_*_model, p_*_anchor, alpha,
alpha_fallback_level.

**2.3 LC features [P2]** — B2/B3/B9/B10 above. New `NEG_FEATURE_NAMES = [n_det_neg, frac_neg,
n_pos_det, t_since_last_pos, neg_run_frac]` in `features/lightcurve.py` (separate list; base
FEATURE_NAMES stays 51). Computed by the BUILDER in `_extract_object_rows` from the
negatives-included window, following the EXT try/except degrade-loudly pattern
(`build_snapshots_fusion.py:96-123`). Gold v11 = v10 columns + the 5 `NEG_FEATURE_NAMES` columns
(`n_pos_det` is one of the 5 — 6 new columns total) + `lc_fallback_all_negative` flag (1 only on
ZTF all-negative fallback rows, where B3 keeps the epoch rows alive). Builder tripwires: `n_det == n_pos_det` on every row with
`lc_fallback_all_negative == 0`; fallback rows assert `n_pos_det == 0 and survey == 'ztf'`;
metrics/G2 denominators gate on `n_pos_det ≥ 1`; assert every det dict carries `is_positive`
(no silent `get(...,True)` default hits). `DEFAULT_FEATURES`
(`models/early_meta.py:29`) extended in sync (legacy path only; fusion Stage-B auto-discovers).
Canonical truncation helper (P2 exports, single implementation):
```python
def truncated_detection_windows(dets, *, survey, max_n_det) -> list[tuple[list[dict], list[dict]]]
# per epoch N: (pos_prefix, full_window). pos_prefix = first N positive dets (feeds base-51+EXT;
# ZTF-identical incl. all-negative fallback for survey != lsst). full_window = all dets with
# t <= t(Nth positive) (feeds neg features + seq v11 tokens). LSST 0 positives -> [].
```

**2.4 Sequence arm [P5, all gated]** — as designed (§3.4) + B7/B8. v11 schema: negatives as tokens,
`is_negative` channel + signed flux. ELAsTiCC2 pretraining selective-fetch default (SN classes).
Gate preregistration: ZTF gate as usual (seq lost twice: v9c −0.0009, v10 −0.0002); additionally an
LSST-assoc-spec cal slice iff n≥30, else default OUT of the deployed blend with standalone
diagnostics on the benchmark at eval time only. seq_v11 is NOT in `SN_FILTER_EXPERTS`.

**2.5 Stage-A trust** — retrained, not redesigned. LSST trust starts effectively ZTF-spec-only (B1);
n_det-sliced trust table re-emitted.

**2.6 Calibration/conformal/FDR** — per-head calibration inside P3 (§2.1). Single cal set accepted
for calibrator + α + conformal + FDR (v8/v10 precedent) WITH the empirical-coverage tripwire kept
as a hard eval check on both locked tests (v10: 0.894 @ 0.90). Conformal refit on POST-BLEND
deployed probabilities; Mondrian strata stay (survey, n_det, n_det_bucket) — per-coverage-bucket
conditional coverage not guaranteed. FDR per survey when support allows.

## 3. Work packages (disjoint file ownership — final)

Shared READ-ONLY for all packages (any needed change = spec escalation):
`models/{splitters,multiclass_followup,pooled_trust,conformal,selection,calibrate,seq_classifier,seq_encoder}.py`,
`scripts/{train,score,eval}_fusion_v8.py`, `scripts/build_helpfulness_fusion.py`,
`scripts/crossmatch_tns.py`, `scripts/crossmatch_lsst_to_ztf.py`, `scripts/local_infer.py`,
`features/lightcurve_ext.py`, `access/associations.py`, `tests/conftest.py`.
(`features/detection.py` moved OUT of this list to P2 — see B2.)
v11 writes ONLY `*_v11`-suffixed model/gold/report paths.

### P1 — truth/label engine
Files: `scripts/download_tns_bulk.py` (EXTEND — replaces the design's fetch_tns_bulk.py),
`scripts/rederive_spec_truth.py` (new, B0), `scripts/build_truth_lsst_live.py` (new),
`scripts/harvest_ztf_lsst_associations.py` (new), `src/debass_meta/access/tns.py` (add bulk fn
only), `jobs/refresh_lsst_live.sh` (new), `tests/test_truth_lsst_live.py` (new).
Emits `data/gold/lsst_live_locked_test.json`, `data/truth/object_truth_v11.parquet`.
TRUTH PLUMBING PIN: the builder takes ONE `--truth` file (`build_snapshots_fusion.py:1205`), and
LSST ids enter gold only via (lc-dir stems) ∩ (truth object_ids) — so P1's FINAL step merges
`lsst_live_truth.parquet` rows into `object_truth_v11.parquet` (keyed `object_id`; LSST-live rows
win for their ids; `stale_xmatch`/`tail_xmatch` rows carried with their demoted quality so the
builder can exclude them). Training builds pass the merged file; eval cohorts pass cohort-only
truth per RUNBOOK.
- `access/tns.py`: `fetch_tns_bulk(credentials, *, date: str | None = None) -> bytes`
  (master when None, else dated diff). Nothing else changes.
- `download_tns_bulk.py`: `--out data/truth/tns_public.parquet --mode {auto,full,diff}`;
  parse via `crossmatch_tns._load_tns_bulk_csv` BY IMPORT (skiprows already handled there);
  diff upsert keyed `objid`/`lastmodified` (§1). No byte-range resume.
- `rederive_spec_truth.py` (B0): inputs `object_truth.parquet` + `ztf_bts.parquet` +
  `tns_public.parquet`; name join via `internal_names`/BTS ZTF ids. Rules: `bts_type='-'` (or any
  unmapped BTS type) NEVER emits `label_quality='spectroscopic'` + a subtype ternary; TNS-typed →
  corrected subtype (101 resolve today: 39 Ia, 15 "SN", 25 non-SN incl. 19 CV); unresolved →
  `label_quality='bts_untyped'` mirroring `tns_untyped` rows (ternary None; Head-1 is_sn=1 only).
  Outputs: `data/truth/object_truth_v11.parquet` (same 20-col schema; original file untouched) +
  `data/truth/label_delta_v11.csv` incl. the locked-765-test label delta (recorded, never silent).
- `build_truth_lsst_live.py`: paged `objects,crossmatch_tns` seed → TNS-bulk-authoritative
  type/discoverydate → client-side sep ≤2″ → epoch window tns_discovery ∈
  [firstDiaSourceMjdTai−90d, lastDiaSourceMjdTai+30d] (MJD↔date client-side; the disc_date cut MAY
  additionally go server-side in `conditions`) → out-of-window matches demoted, NOT dropped:
  discovery before window → `label_quality='stale_xmatch'`; after window →
  `label_quality='tail_xmatch'` (same-transient late match, e.g. AT 2026aok LSST first det +130 d)
  — both excluded from train AND eval, kept in output with the rejection reason.
  Harvests NEW catalog negatives for train/cal (B5, target ≥300).
  Fink xm backup per §1 ("Fail"→missing). Outputs: `data/truth/lsst_live_truth.parquet` in the
  EXACT 20-col `object_truth.parquet` schema (dtypes pinned in review-integration §3.7; reference
  generator `data/live_eval_20260704/tools/make_truth_tns.py`); cleaned 2026-07-04 cohort truth as
  NEW `*_cleaned.parquet` (never overwrite `data/live_eval_20260704/**`); benchmark manifest
  `{"test_ids":[...],"frozen_utc":"...","policy":"...","source":"..."}`. RUNBOOK rule: a truth row
  for EVERY cohort LSST object (even tns_untyped/ternary-None) — objects missing from truth drop
  silently; assert snapshot count == cohort count after builds.
- `harvest_ztf_lsst_associations.py`: inverted direction (§1); parse ZTF ids from `disc_int_name`;
  association CSV per `load_lsst_ztf_associations` schema: `lsst_object_id, ztf_object_id,
  match_status="matched", sep_arcsec` (+ `association_kind`, `association_source=
  "lasair_crossmatch_tns"`, `match_count`); reuse `crossmatch_lsst_to_ztf.py` helpers by import;
  spec rows `label_source='ztf_assoc_spec'`; cone endpoint only as fallback for names lacking
  disc_int_name.
- `refresh_lsst_live.sh`: weekly TNS refresh → truth rebuild → route new spec by hash (B1) →
  rescore → append benchmark row. SCORING flow (RUNBOOK): empty locked-split JSON as
  `--trust-metadata`, cohort-only `--labels` CSV, `--bts ""`, `--no-lsst-weak`, never `--smoke`.
- Tests: synthetic; epoch-window demotion (stale AND tail); ≤2″ cut; manifest schema; diff upsert;
  "Fail"/"nan" mapping + max-MJD cross-alert aggregation; rederive rules (untyped never
  spectroscopic; bts_untyped tier; delta emitted); hash-routing determinism.

### P2 — features/gold
Files: `src/debass_meta/features/detection.py` (is_positive fix ONLY),
`src/debass_meta/features/lightcurve.py`, `src/debass_meta/ingest/gold.py`,
`scripts/build_snapshots_fusion.py` (v11 mode), `src/debass_meta/models/early_meta.py`,
`scripts/assert_locked_gold_identity.py` (new), `tests/test_lightcurve_neg.py` (new),
`tests/test_gold_positive_only.py` (new).
- B2 fix at `detection.py:163-165`; B3/B9/B10 contract in §2.3; `--lsst-live-locked` flag (B11):
  ids forced to TEST/quarantined out of train∪cal inside `build_split_manifest`
  (`build_snapshots_fusion.py`, NOT splitters.py), disjointness hard-asserted (G6-at-build).
  All existing `main()` flags keep names/semantics (pinned in review-integration §3.1);
  v11 extends, never renames.
- `assert_locked_gold_identity.py --reference <v10 parquet> --rebuilt <v11 parquet>
  [--reference-hash-manifest <json>]`: on ZTF-survey rows, (a) row multiset keyed
  `object_id × n_det` identical; (b) every v10-era column (base-51 + EXT + traj +
  `proj__/avail__/exact__`) exactly equal (`assert_frame_equal` after align). Reference
  `data/gold/object_epoch_snapshots_fusion_v10.parquet` is SCC-ONLY → hash-manifest fallback +
  skipif convention locally. NEVER compares file bytes.
- SCC census deliverable: count of locked v6e2/v10 objects that are all-negative
  (fallback-dependent) in the build report, BEFORE any further fallback change.
- Tests: `test_lightcurve_neg.py` — neg features correct; LSST 0-pos ⇒ 0 epochs; ZTF fallback
  preserved; helper windows. `test_gold_positive_only.py` — G5-style: appending future NEGATIVE
  detections must not change ANY feature at fixed n_pos_det; n_det==n_pos_det tripwire.

### P3 — hierarchical head
Files: `src/debass_meta/models/hierarchical_followup.py` (new),
`tests/test_hierarchical_followup.py` (new).
- `class HierarchicalFollowup`: `fit(df, ...)`, `predict_proba(df) -> (N,3)`,
  `predict_proba_raw(df) -> (N,3)` (uncalibrated-heads composition — REQUIRED: the v8 scorer calls
  both, `score_fusion_v8.py:340-345`), `save(dir)`/`load(dir)` with `metadata.json` carrying
  `feature_cols` per head + head-2-on-LSST routing decision + seq of gate verdicts.
  Mirrors `MulticlassFollowupArtifact` conventions: `CLASSES=("snia","nonIa_snlike","other")`,
  `_SPEC_QUALITIES`, `_prepare_frame` NaN-fills missing feature cols, survey routing by literal
  lowercase `survey` string. `survey_cal_min` constructor param (default 40; reconciles the ≥150
  constant in `multiclass_followup.py:885-905`). Imports calibrator ladder from `models/calibrate.py`
  and `PlattCalibrator` from `pooled_trust` — never forks them. Everything else per §2.1.
- Tests: simplex sum-to-1 + clipping; P2-constant ⇒ p_snia rank == P1 rank (graceful degradation);
  save/load roundtrip incl. predict_proba_raw; tiny-n survey → Platt fallback; head-2 default
  excludes survey_is_lsst; LSST routing default = base rate; G7 fires on an untyped-provenance row.

### P4 — anchor/blend/scorer
Files: `src/debass_meta/models/anchor_blend.py` (new), `scripts/score_fusion_v11.py` (new),
`tests/test_anchor_blend.py` (new), `tests/test_score_v11.py` (new).
- `anchor_blend`: `compute_anchor(df) -> df` (adds p_{snia,nonia,other}_anchor, n_experts_fired;
  ε-clip+renorm; only `proj__*` ternary experts — regex discovery per `score_fusion_v8.py:256-259`);
  `fit_alpha(cal_df) -> BlendSpec` (grid/1-SE/honesty filters/fallback ladder per §2.2; BlendSpec
  JSON-serialized to `models/anchor_blend_v11/blend.json` with alpha table, bucket edges, per-survey
  base rates, per-cell n after filtering); `apply(df, spec) -> df`. IA_CAPABLE + asserts per §2.2.
- `score_fusion_v11.py`: sys.path inserts BOTH repo-root and `src` (pattern
  `scripts/export_lsst_candidates.py:14-15,30`), then `from scripts.score_fusion_v8 import
  ndet_bucket, _latest_per_object, attach_trust_columns, compute_fdr_thresholds,
  trust_weighted_p_snia` + constants (`CLASSES, GOALS, FDR_LABEL_QUALITIES, LOCAL_PSNIA_COLS,
  DP1_CATALOG_COLS, GOAL_SCORE_COLS`) — extend by import, never reimplement. CLI mirrors v8
  verbatim + `--blend-dir` (default `models/anchor_blend_v11`). Calls `predict_proba_raw` AND
  `predict_proba`; conformal `MondrianAPS.load(...).predict_sets` on POST-BLEND probs with strata
  `survey` (lowercase str), `n_det` (int), `n_det_bucket`.
- Tests: mask self-verification (synthetic events through every non-IA_CAPABLE projector ⇒ constant
  Ia|SN ratio); `IA_CAPABLE ⊆ ALL_EXPERT_KEYS`; ε-clip ⇒ finite log-loss on stamp-only cells;
  0-expert row ⇒ α=1 + p_anchor NaN handled; scorer smoke on synthetic snapshot emits pinned columns.

### P5 — sequence arm (all gated)
Files: `src/debass_meta/features/sequence_dataset.py`, `scripts/train_seq_encoder.py`,
`scripts/train_seq_classifier.py`, `scripts/fetch_elasticc2.py` (new),
`src/debass_meta/experts/local/seq_v9.py`, `src/debass_meta/experts/local/__init__.py` (B7),
`src/debass_meta/projectors/base.py`, `src/debass_meta/projectors/local_seq_v9.py`,
`tests/test_seq_v11.py` (new).
- `sequence_arrays(dets, *, schema="v9")` (B8); v11 schema adds is_negative channel + signed flux;
  `truncated_positive_detections` delegates to P2's canonical helper (import; lockstep copies die).
  `SEQ_CONTINUOUS_FIELDS`/9-dim behaviour bit-identical under schema="v9" (v9/v10 artifacts safe).
- `train_seq_classifier.py` writes `seq_schema` into artifact meta (`SeqClassifierArtifact.save/load`
  pass meta through — `models/seq_{classifier,encoder}.py` stay READ-ONLY). `seq_v9.py`: schema
  routing per artifact; new `SeqV11Expert(name="seq_v11")` replicating ALL of the v9 expert's
  semantics (env `DEBASS_SEQ_V11_MODEL`; OOF `fold_map.json` routing incl. corrupt⇒RuntimeError and
  `fold_route="missing_fold_{k}_fallback_full"`; canonical ternary + `p4_*` extras;
  `available=False` when artifact missing).
- Registry: `EXPERT_REGISTRY["seq_v11"]` scope "any"; explicit `_dispatch_projector` branch;
  projector reuses `local_seq_v9.project_events`; NOT in `SN_FILTER_EXPERTS`. Column prefix
  `proj__seq_v11__*` then auto-discovers through gold/helpfulness/trust.
- `fetch_elasticc2.py --models <list>` (default SN classes, ~3-4 GB) — per-class HEAD/PHOT
  `.FITS.gz` fetch with resume (`Accept-Ranges`); `--full` for the 7.91 GB tarball; URL pinned §1.
- Training ladder: (optional) ELAsTiCC2 pretrain on SCC GPU (`-l gpu_c=8.0`) → ZTF fine-tune →
  LSST-assoc/live-spec-train (association-aware exclusions; locked live test NEVER seen) → 5-fold
  OOF. If fetch infeasible in-session: ship the script + SCC job, proceed without.
- Tests: schema="v9" 9-dim regression; v11 dims/channels; meta routing; registry+dispatch; no network.

### P6 — orchestrator
Files: `scripts/train_fusion_v11.py` (new), `jobs/run_fusion_v11_build.sh`,
`jobs/run_fusion_v11_pretrain.sh`, `jobs/run_fusion_v11_expert.sh`,
`jobs/submit_fusion_v11_chain.sh` (all new), `tests/test_train_v11_smoke.py` (new).
- `train_fusion_v11.py` imports (read-only) from `scripts.train_fusion_v8`: `load_split_manifest`,
  `run_component_gates`, `_macro_ovr_auc`, `_paired_delta_ci`, `ndet_bucket`, `class_index`;
  Stage-A via `pooled_trust.train_pooled_trust`; helpfulness via CLI `build_helpfulness_fusion.py`
  (auto-discovers seq_v11). All builds/train take `--truth data/truth/object_truth_v11.parquet`
  (B0); train refuses to fit Head-2 if the v11 truth file is absent (B0 ordering). Pipeline:
  Stage-A → P3 heads (G7 inside fit) → P3 calibration → P4 anchor+α → conformal (post-blend) →
  FDR → guards G2/G3/G6 → gates (§5) → reports (availability audit, coverage tripwire, α fallback
  ledger, SCC negativity census echo, per-cell n after honesty filtering). G1/G4 run in the EVAL
  stage only (see §4). Internal changes to shared scripts = escalation, not a P6 edit.
  Order of operations: P1 rederive (B0) + P2 detection fix (B2) land FIRST; P3/P4/P5 build in
  parallel against the pinned interfaces; P6 integrates last.
- Job scripts: v10 conventions — `qsub -terse -P pi-brout`, `-hold_jid` chaining,
  `FUSION_V11_FORCE=1` + skip-markers, GPU stage `-l gpu_c=8.0` (never bare gpus=1), artifact env
  vars exported; NEVER export `DEBASS_SEQ_V9_MODEL` pointing at v11 artifacts. Build job runs
  `assert_locked_gold_identity.py` (G5b) after the SCC gold build. Benchmark re-score = scoring
  flow (RUNBOOK, as P1's refresh script); includes v10 re-scored on cleaned truth.
- Smoke test: end-to-end on synthetic gold; asserts guard statuses present in report JSON.

## 4. Guards (definitions + assert locations)

| Guard | Definition | Location / semantics |
|---|---|---|
| G1 | ZTF locked-test macro OvR AUC within v10's paired CI (no regression) | EVAL-TIME ONLY (test-touching; never consulted in training) — eval stage of train_fusion_v11.py / eval report; `_macro_ovr_auc` + `_paired_delta_ci`; nonzero exit on breach |
| G2 | On (OOF-train LSST spec-Ia) ∪ (LSST spec-tier cal) ∪ (assoc-spec) rows: median p_snia ≥ 0.15 AND max > 0.2; REQUIRE n ≥ 10 | train_fusion_v11.py guard stage (OOF/cal only, no test). n<10 ⇒ status `G2_UNEVALUABLE`, nonzero exit; `--acknowledge-g2-unevaluable` continues but stamps report headline. Count all-NaN-LC (`lc_all_nan`) rows in the denominator report |
| G3 | Per fitted (survey × bucket) cell: cal log-loss(blend) ≤ cal log-loss(ε-clipped anchor) [guaranteed: α=0 in grid]. POST-FIT verification: per survey, pooled cal log-loss(blend) ≤ anchor AND macro OvR AUC(blend) ≥ anchor; on failure collapse to single per-survey α, then α=0 (terminates: α=0 ⇒ blend==anchor). n_experts=0 bucket excluded (α≡1) | `anchor_blend.fit_alpha` implements + records; train_fusion_v11.py asserts final status |
| G4 | LSST-live benchmark SN-vs-other AUC(blend) ≥ AUC(anchor) − 0.02; aspiration ≥ 0.751 (raw stamp) | EVAL-TIME ONLY — benchmark scorer/eval report (P6) |
| G5 | Existing no-leakage asserts unchanged; v11 addition: future NEGATIVE dets don't change features at fixed n_pos_det | existing tests + `tests/test_gold_positive_only.py` (P2) |
| G5b | Value-identity: ZTF rows of rebuilt v11 gold vs locked v10 gold — row multiset (object_id × n_det) identical AND every v10-era column exactly equal. Never file bytes | `scripts/assert_locked_gold_identity.py` (P2); SCC build job (P6); hash-manifest fallback + skipif locally |
| G6 | `lsst_live_locked_test.json` test_ids ∩ (train ∪ cal) = ∅ | (a) AT BUILD: `build_snapshots_fusion.py --lsst-live-locked` (P2); (b) AT TRAIN: train_fusion_v11.py after load_split_manifest (P6). Hard assert both |
| G7 (new, B0) | Head-2 training rows with untyped provenance == 0 (row admitted only if `tns_type` non-empty or `bts_type` ∉ {'-',''}) | `HierarchicalFollowup.fit` (P3) hard assert; presupposes P1's `object_truth_v11.parquet` |

## 5. Gates (cal-decided, 1-SE; preregistered defaults)

per-survey head-1 vs pooled (is_sn-level LSST cal frame, honesty-filtered; default POOLED if frame
too small) · head-2 binary vs 3-way (ZTF decision — labeled as such in report) · head-2-on-LSST
routing (default per-survey constant base rate; enable needs n≥30 + non-inversion + 1-SE) ·
negative-flux features (gated on the is_sn-level frame, BOTH surveys — ZTF has negatives too) ·
tns_untyped/bts_untyped weight w ∈ {0, 0.5} (one shared gate; ZTF-decidable; applies to
NON-benchmark untyped only — §6's "SSL corpora only" governs just the frozen 78, enforced by G6) ·
head-1 weak_weight ∈ {0.1, 0.3} ·
seq_v11 in/out (§2.4 preregistration) · Dirichlet rung (ZTF-decidable; LSST rung skips per ladder).
NOT a gate: survey_is_lsst in head-2 — ablated by default (§2.1).

## 6. Locked benchmarks & split policy

`data/gold/lsst_live_locked_test.json` = cleaned 2026-07-04 spec cohort (~105-120) + the 150
catalog others, frozen by P1; the 78 untyped join SSL corpora only. POST-FREEZE spec arrivals:
sha1(object_id) last hex digit even→TEST (manifest append via refresh), odd→train/cal (B1).
New catalog negatives (B5) go to train/cal, never the benchmark. Headline v11 = this benchmark +
locked ZTF 765-spec test (labels re-derived per B0, delta recorded), side by side with v10
re-scored on the same cleaned truth (both surveys) for apples-to-apples.
Locked artifacts (must remain intact): `models/trust/metadata.json` (LOCAL differs from SCC — key
off the environment's file, never hardcoded counts), `data/gold/object_epoch_snapshots_fusion_v10.parquet`
(SCC only), `models/{followup,trust,conformal}_fusion_v10/` (sklearn 1.7.2-pickle warning benign —
NEVER upgrade sklearn), `models/seq_classifier_v10*/` + `models/seq_encoder_v10/`,
`data/gold/split_fusion_v10.json`, `data/live_eval_20260704/**`, `data/lsst_candidates.csv`.

## 7. RUNBOOK contract (P1/P2/P6 scoring/build flows inherit; smoke-verified)

(1) venv `~/.venvs/debass_py313`; `load_dotenv("/Users/tz/Documents/GitHub/rubin_hackathon/.env")`
explicit path; normalize_* survey arg `"LSST"` uppercase or `"auto"`. (2) `--labels` is ZTF-only —
LSST ids enter gold via (lc-dir stems) ∩ (truth object_ids): P1 writes a truth row for every wanted
object; assert snapshot count == cohort count. (3) eval/scoring cohorts: EMPTY locked-split JSON as
`--trust-metadata`; training builds: real metadata.json. (4) `--bts ""` on LSST cohorts;
`--no-lsst-weak` mandatory on eval cohorts; never `--smoke` at scale; `--skip-traj` safe.
(5) fetch `--max-epochs 20` == builder `--max-n-det 20` (raise both or neither); chunk ~50/fetch,
~40/backfill `--parallel 8`; `normalize.py` ONCE after all chunks. (6) tag `lc_all_nan` objects,
count per cohort (feeds G2 denominators). (7) LSST rows use the global calibrator in v9c/v10
artifacts (expected). (8) `fink_lsst/early_snia` mostly absent at n_det 3-5 → Ia pool falls back to
base rate by design. (9) truth-side `fink_lsst/crossmatch` events never become gold feature columns
(keep the post-build check); the discovery-epoch window applies to Lasair-sourced names too.

## 8. Deviations from design (each: what changed → why)

1. `fetch_tns_bulk.py` → extend existing `scripts/download_tns_bulk.py`; parse via
   `crossmatch_tns._load_tns_bulk_csv` import — script already exists; duplication risk (A2).
2. TNS "cached, resumable" → full master re-download + daily-diff upsert (objid/lastmodified);
   no byte-range resume (no Content-Length); hourly files 404 — measured (data §1).
3. Association harvest INVERTED (Lasair `objects,crossmatch_tns` paged join + `disc_int_name`
   parsing; conesearch = fallback only) — measured 0/25, 0/25, 2/233; design's O(50-300)
   conesearch yield fails by ~an order of magnitude (data §3).
4. §6 test-first policy → hash alternation from the start for post-freeze spec arrivals —
   otherwise LSST spec train/cal stays single-digit for months (blocking B1).
5. Stated as fact: head-2 + LSST Stage-A trust start ZTF-spec-only; LSST per-survey calibration
   tiny-n fallback triggers immediately (blocking B1).
6. `watchlist_hits wl_id=46` struck — table does not exist on Lasair LSST; mechanism is the
   implicit join; sep computed client-side, cut ≤2″ (join max 2.97″) (data §4).
7. Negative-det bug site corrected: LSST `is_positive` keyed on `psfFlux>0` at
   `detection.py:163-165` (not a builder-loader bypass); `detection.py` re-assigned from
   shared-read-only to P2 for this fix only (ml §1, integration A5).
8. All-negative fallback survey-gated (ZTF kept — 11% of LCs depend on it; LSST removed);
   negatives feed ONLY the 5 new features; ONE canonical truncation helper replaces 3 lockstep
   copies; SCC locked-object negativity census before any further change (ml §2, A5/A6).
9. G5b redefined as value-identity on v10 columns over locked ZTF rows + hash-manifest fallback —
   new columns make byte identity impossible; reference parquet is SCC-only (ml §2.3, A4).
10. G3 restated: per-fitted-cell cal log-loss guarantee + post-fit per-survey pooled verification
    with terminating fallback ladder; anchor ε-clipped; n_experts=0 ⇒ α=1 excluded; α fit
    honesty-filtered — original G3 unprovable (AUC/pooled), infinite log-loss on stamp cells,
    circular on stamp-derived weak labels (ml §3).
11. IA_CAPABLE corrected: `fink/snn` ADDED (design omitted the strongest ZTF Ia broker);
    `fink_lsst/cats` OUT (row-conditional, documented forfeit); sherlock/babamul excluded from the
    anchor entirely; self-verifying unit test + subset assert (ml §4).
12. 150 catalog others stay benchmark-only; NEW catalog negatives harvested for train/cal
    (target ≥300) — design double-booked them (ml §5a, blocking B5).
13. G2 pinned to OOF-train ∪ spec-tier-cal ∪ assoc-spec LSST spec-Ia rows, n≥10, loud
    UNEVALUABLE fail — original had zero eligible rows (ml §7, blocking B6).
14. survey_is_lsst ablation in head-2 = DEFAULT, not a gate — gate undecidable (no LSST spec cal)
    and the v10 harm was survey-flag-mediated; inherited small-frame fallback would keep it (ml §7).
15. head-2-on-LSST default = per-survey constant base rate with cal-gated enable + non-inversion
    check — measured anti-correlated ZTF-flavored Ia scores on LSST make "worse than constant"
    possible; G2 checks level, not sign (ml §6.5).
16. G1/G4 moved to eval-time-only — they are test-touching; "hard asserts in train/score" would
    break anti-peeking (ml §7).
17. Sequence schema versioned (`schema="v9"` default; `seq_schema` artifact meta;
    `DEBASS_SEQ_V11_MODEL`) — in-place edit breaks deployed v9/v10 artifacts (cont_dim=9
    persisted) (A7).
18. P5 file list += `experts/local/__init__.py`; explicit `_dispatch_projector` seq_v11 branch —
    without them the expert emits nothing (A1).
19. G6-at-build via `--lsst-live-locked` on `build_snapshots_fusion.py` (P2) with P1-pinned
    manifest schema; NOT in shared `splitters.py` (A3).
20. `predict_proba_raw` added to the P3 interface (v8 scorer requires it); `survey_cal_min`
    constructor param reconciles the 40-vs-150 threshold conflict (A8, A11).
21. Platt (not temperature) as tiny-n binary calibration fallback — temperature cannot fix
    intercept bias, the v10 failure mode (ml §6.6).
22. `fetch_elasticc2.py --models` selective per-class fetch default (~3-4 GB) instead of the
    7.91 GB tarball — avoids 30-40 GB decompress scratch (data §2).
23. head-1 `weak_weight` gated {0.1, 0.3} instead of silently inheriting 0.1 — tuned for the
    wrong (subtype-forced) regime (ml §5d).
24. Neg features computed in the builder from the negatives-window via `NEG_FEATURE_NAMES`,
    not by extending `extract_features` (which cannot see negatives without breaking G5b);
    `DEFAULT_FEATURES` still extended in sync per CLAUDE.md (A6).
25. LSST head-1 provenance-mask equalization + availability audit — masking otherwise creates a
    class-correlated availability pattern, the live-eval harm-#2 mechanism (ml §5b).
26. Cal-reuse: single cal set accepted (v8/v10 precedent) + empirical-coverage tripwire kept as a
    hard eval check on both locked tests — the sub-split alternative was optional (ml §6.2).
27. NEW P1 file `scripts/rederive_spec_truth.py`, new weak tier `bts_untyped`, new guard G7,
    all v11 flows on `object_truth_v11.parquet`; design §1's "~10.7k spectroscopic rows" restated
    as ~7.5k typed + ~3.1k weak — 61% of the spec nonIa class was BTS-untyped filler; Head-2 on the
    current truth table reproduces bug-class #1 (data §6, blocking B0).
28. `tail_xmatch` demotion label added (late same-transient matches recorded, not silently
    dropped) — measured real case AT 2026aok at +130 d (data §4).
29. Fink LSST xm truth use requires cross-alert aggregation (max-MJD non-nan) + `"nan"`/`"Fail"`
    sentinel mapping — per-alert type missing on 178/470 post-classification alerts (data §5).
30. `n_det == n_pos_det` tripwire scoped to non-fallback rows via new `lc_fallback_all_negative`
    flag — the blanket assert would fire on the 11% ZTF all-negative rows that B3 deliberately
    keeps alive (ml §2 + ml §8 reconciled).
