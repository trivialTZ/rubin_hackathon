# fusion_v11 — Design (2026-07-04)

Author: Claude (Fable 5), for TZ / metaDEBASS. Status: DESIGN → review → spec → implement.
Grounding: the 2026-07-04 live-LSST evaluation (`data/live_eval_20260704/report/{analysis,skeptic}.md`,
skeptic verdict CONFIRMED-WITH-CAVEATS) and the v10 campaign results (`reports_from_scc/fusion_v10`).

## 0. Why v11 (what the live eval proved)

1. **[BUG] GBM v10's Ia channel is structurally dead on LSST.** max p_snia = 0.0215 over all
   3,947 live rows; 100% of 83 spectroscopic Ia got p_nonia > p_snia. Mechanism: LSST weak
   labels were mapped stamp-SN → nonIa **never** snia while `survey_is_lsst` is a feature →
   the model learned "LSST ⇒ not Ia". This is a label-taxonomy error, not a data-volume problem.
2. **[HARM] Trained fusion can rank below its own inputs.** SN-vs-other on live LSST: raw
   ALeRCE Rubin stamp 0.728/0.751; GBM v9c 0.421 and GRU v9 0.366 (significantly
   ANTI-correlated). Mechanism (hypothesis, design against it): on ZTF training data,
   coverage/NaN patterns correlate with junk, so Stage-B learned "no LC features + stamp-only
   ⇒ other"; on LSST that pattern describes *real SNe*. Coverage priors leaked across surveys.
3. **[NULL] Nothing separates Ia from non-Ia on live LSST** (all systems AUC 0.42–0.66, CIs
   cross 0.5) — brokers included (LSST SNN/CATS are SN-vs-not; EarlySNIa needs ≥7 epochs).
   The Ia axis on LSST must come from LOCAL photometric experts — which are currently
   ZTF-flavored (ZTF-only GRU rank-corr with LSST-adapted arm: −0.04 to −0.12).
4. **[LABELS] The bottleneck is labels, not photometry**, and our label plumbing has two bugs:
   44/150 spec labels were STALE (≤2024 TNS names positionally matched to 2025–26-only flux),
   and gold `n_det` counts negative-flux detections (31% of the gate-3 spec slice had
   all-NaN LC features).

## 1. Data inventory (what we have / what we can get online)

HAVE (local + SCC):
- ZTF: ~10.7k spectroscopic truth rows; v10 split 8,462 train / 2,555 cal / 1,755 test objects;
  locked 765-spec-object ZTF test (byte-identical since v6e2 — MUST stay so).
- LSST commissioning weak: 3,998 objects (`data/lsst_candidates.csv`) — the bug source; safe
  to reuse ONLY at the SN-vs-other level (see §3).
- LSST live labeled (2026-07-04 eval cohort, never trained on): 150 spec (→ ~105–120 after
  stale-label cleaning), 150 catalog-basis others, 78 TNS-untyped. **Freeze as the locked
  LSST-live benchmark** (§6).
- DP1 50k snapshot harness + enrichment metrics; silver broker events both surveys;
  ZTF↔LSST association machinery (`--association-csv`, association-aware split exclusions from v10).

GET ONLINE (verified live or to verify in review):
- **TNS bulk daily CSV** (`tns_public_objects.csv.zip`, API-key POST) — every TNS object with
  ra/dec/type/discoverydate. Fixes staleness (epoch-aware crossmatch) and removes per-object
  rate limits. [P1]
- **BTS/ZTF spec stream ∩ LSST footprint**: ZTF-classified SNe since 2025-10 at dec ≲ +12,
  conesearched against Lasair/Fink LSST (2″) → real LSST lightcurves with ZTF-grade spec
  labels. Expected yield O(50–300); measure, don't assume. [P1]
- **Live LSST alerts**: Fink LSST (SNN/CATS/EarlySNIa + free xm_*), ALeRCE LSST (Rubin stamp),
  Lasair LSST (sherlock, crossmatch_tns via watchlist_hits wl_id=46) — 12.9M alerts/124 nights
  and growing nightly now the survey is official. Spec labels expected to grow ~tens/week.
- **ELAsTiCC2** (DESC public, NERSC portal): millions of simulated LSST lightcurves with truth —
  GRU pretraining corpus. OPTIONAL arm; time-boxed feasibility check in review. [P5, gated]

## 2. Design principles

- **Every label supervises exactly the level it constrains.** Weak "SN" labels know SN-ness,
  not subtype. Spectroscopic labels know subtype. Never force a weak label into a subtype class
  (that is bug #1).
- **The meta-layer must never underperform its best input.** Fusion output is anchored to the
  broker pool with a learned, bounded deviation (fixes harm #2 by construction, not by hope).
- **Per-survey where labels allow, survey-agnostic where physics transfers.** SN-vs-junk is
  survey-idiosyncratic (artifacts, cadence, coverage) → per-survey. Ia-vs-nonIa is physics
  (color, rise, timescale) → shared, with per-survey calibration.
- **Locked tests are sacred**: ZTF 765-spec test stays byte-identical; the new LSST-live
  benchmark is frozen and never trained on.

## 3. Architecture

### 3.1 Hierarchical follow-up head (replaces the flat ternary/4-way Stage-B)  [P3]
- **Head-1 `P(SN|x)`** (binary LightGBM): snia+nonIa_snlike vs other. Trained on ALL tiers:
  spec, weak (stamp/consensus SN → is_sn=1 — now used at the level it actually constrains),
  catalog others (is_sn=0), tns_untyped optionally is_sn=1 at weight w ∈ {0, 0.5} (gated).
  Per-survey: separate LSST head trained on LSST rows (weak 3,998 + catalog + spec + assoc)
  and ZTF head on ZTF rows; fall back to a pooled head with `survey_is_lsst` if the per-survey
  gate does not pay on cal.
- **Head-2 `P(type|SN,x)`**: trained ONLY on label_quality='spectroscopic' SN rows (ZTF ~4k
  objects + LSST associations + live spec-train). Two variants, gated on cal: binary Ia-vs-nonIa,
  or 3-way {Ia, II, Ibc/other-SN} folded back to the ternary. Shared across surveys; ablation
  arm drops `survey_is_lsst` (+ any survey-degenerate columns) from head-2's features.
- **Composition**: p_snia = P1·P2(Ia), p_nonia = P1·(1−P2(Ia)), p_other = 1−P1. Sums to 1 by
  construction. Property: when P2 is uninformative (early LSST), p_snia ranking degrades
  gracefully to SN-ness ranking — never to an inverted signal.
- Provenance masking / grouped weak-SN loss from v10 reused for head-1. q-as-features stays out
  (hurt in v8/v10). Artifact: `models/followup_fusion_v11/{sn_head,type_head}/` + metadata with
  feature_cols per head.

### 3.2 Anchored fusion with domain-aware shrinkage  [P4]
- **Anchor** = trust-weighted linear pool of fired experts' ternaries (reuse `ensemble_p_snia`
  machinery, extended to the full ternary), with an **Ia-capability mask**: experts whose
  projector carries no Ia information (ALeRCE Rubin stamp p_snia ≡ 0 structurally; LSST
  SNN/CATS 50/50 splits) contribute to the SN axis only; the Ia|SN axis pools only Ia-capable
  experts (fink/rf_ia, fink_lsst/early_snia, salt3_chi2, supernnova, seq_v11, …), else falls
  back to the cal-set base rate P(Ia|SN). `IA_CAPABLE` set lives in `models/anchor_blend.py`
  (strings; no registry edit → no file conflict with P5).
- **Blend**: p_final = α·p_model + (1−α)·p_anchor, α ∈ [0,1] fit on cal per
  (survey × coverage-bucket) grid {0,.25,.5,.75,1} minimizing log-loss with a 1-SE rule
  preferring the anchor. Coverage bucket = f(frac non-NaN LC features, n_experts fired,
  n_pos_det). **Guarantee target: blended ≥ anchor on every cal slice** (guard G3). Scorer
  emits p_* (blended, deployed), p_*_model, p_*_anchor, alpha.

### 3.3 Lightcurve features: negative-flux signal + positive-only epochs  [P2]
- Fix the epoch contract: n_det = POSITIVE detections only, uniformly across surveys;
  truncation at epoch N = all detections (incl. negatives) with time ≤ time of Nth positive
  detection. Investigate why LSST negative dets counted (likely the builder's loader bypassing
  is_positive); **assert the locked ZTF gold stays byte-identical** after the fix (guard G5b,
  `scripts/assert_locked_gold_identity.py`).
- Add negative-flux features (the ~50% negative-only LSST population is signal, not noise —
  fading tails/variables are "other"/old): `n_det_neg`, `frac_neg`, `n_pos_det`,
  `t_since_last_pos`, `neg_run_frac` (≤6 features). Contract: extend
  `features/lightcurve.py:FEATURE_NAMES` + `models/early_meta.py:DEFAULT_FEATURES` (CLAUDE.md
  rule). Gated arm in training.
- Gold v11 adds `n_pos_det` column; metrics gate on positive detections.

### 3.4 Sequence arm (GRU v11)  [P5, all gated]
- Sequence schema v11: include negative detections as tokens with an is_negative channel +
  signed flux; per-survey NormStats (exists since v10); new encoder/classifier artifacts
  (`models/seq_{encoder,classifier}_v11*`).
- Training ladder: (optional) ELAsTiCC2 pretraining on SCC GPU → staged fine-tune
  ZTF → LSST-assoc/live-spec-train (association-aware exclusions; locked live test NEVER seen);
  5-fold OOF for train-row projections (v10 machinery). Registered as expert `seq_v11`
  (registry + projector + ALL_LOCAL_EXPERTS, contract steps in CLAUDE.md).
- Expectations honest: the seq expert has been gated out twice at ZTF scale; its job here is
  the LSST Ia axis. If ELAsTiCC2 fetch is infeasible in-session, ship
  `scripts/fetch_elasticc2.py` + SCC job and proceed without.

### 3.5 Stage-A pooled trust: retrained, not redesigned
Same pooled LightGBM (expert-ID categorical). New LSST rows (associations + live spec-train +
cleaned weak) give LSST trust real support; n_det-sliced trust table (v10) re-emitted.

### 3.6 Truth & label engine  [P1]
- `scripts/fetch_tns_bulk.py`: daily TNS public-objects dump → `data/truth/tns_public.parquet`
  (cached, resumable).
- `scripts/build_truth_lsst_live.py`: **epoch-aware** positional crossmatch (sep ≤ 2″ AND
  tns_discovery ∈ [first_det−90d, last_det+30d]) → `data/truth/lsst_live_truth.parquet`;
  stale matches demoted to label_quality='stale_xmatch' (excluded from train AND eval).
  Also re-derives the 2026-07-04 eval cohort's truth (`*_cleaned.parquet`) and emits the
  frozen benchmark manifest (§6).
- `scripts/harvest_ztf_lsst_associations.py`: BTS/TNS-classified ZTF objects (2025-10+,
  dec ≤ +12) → Lasair/Fink LSST conesearch → association CSV rows + spec-labeled LSST objects
  (label_source='ztf_assoc_spec'). Same-transient leakage handled by association-aware grouped
  splits (v10 machinery).
- `jobs/refresh_lsst_live.sh` (the ops tool): weekly TNS refresh → truth rebuild → new spec
  objects appended to the benchmark manifest (test-first policy) → rescore → append one
  benchmark row. metaDEBASS becomes a *continuously evaluated* system.

## 4. Calibration / conformal / selection
- P1 and P2 calibrated separately (isotonic per survey; fall back to shared temperature when
  survey-cal n < 40 — LSST head-2 cal will start tiny). Optional Dirichlet rung on the composed
  ternary via the v10 calibration ladder (gated).
- Mondrian conformal + utility/budget/FDR selection unchanged, refit on v11 probs; FDR fit
  per survey when cal support allows.

## 5. Guards (hard asserts in train/score) and gates (cal-decided)
Guards: **G1** ZTF locked-test macro AUC within v10's CI (no regression). **G2** anti-Ia
regression test: on LSST spec-Ia cal rows, median p_snia ≥ 0.15 and max > 0.2 (the v10 failure
mode cannot recur silently). **G3** blended ≥ anchor on every cal slice (log-loss + OvR AUC).
**G4** live-LSST SN-vs-other ≥ anchor − ε (aspiration: beat stamp 0.751). **G5** no-leakage
asserts unchanged; **G5b** locked ZTF gold byte-identity after the positive-only fix.
**G6** benchmark manifest ∩ (train ∪ cal) = ∅, asserted at build AND at train.
Gates (1-SE rule on cal): per-survey head-1 vs pooled; head-2 binary vs 3-way; survey-flag in
head-2; negative-flux features; tns_untyped weight; seq_v11 expert in/out; Dirichlet rung.

## 6. Locked LSST-live benchmark (new)
`data/gold/lsst_live_locked_test.json`: the cleaned 2026-07-04 spec cohort (~105–120 objects)
+ the 150 catalog others, frozen; policy for future spec arrivals: append to TEST until
n_test ≥ 300, then alternate test/cal by object-id hash. These ids never enter train/cal
(guard G6). The 78 untyped may join SSL corpora only. Headline v11 numbers = this benchmark +
the locked ZTF test, reported side by side with v10's (v10 re-scored on the cleaned truth for
apples-to-apples).

## 7. Work packages (disjoint file ownership — enforce during implementation)
- **P1 truth/label engine**: scripts/fetch_tns_bulk.py, scripts/build_truth_lsst_live.py,
  scripts/harvest_ztf_lsst_associations.py, src/debass_meta/access/tns.py (bulk fn only),
  jobs/refresh_lsst_live.sh, tests/test_truth_lsst_live.py.
- **P2 features/gold**: src/debass_meta/features/lightcurve.py, src/debass_meta/ingest/gold.py,
  scripts/build_snapshots_fusion.py (v11 mode), src/debass_meta/models/early_meta.py,
  scripts/assert_locked_gold_identity.py, tests/test_lightcurve_neg.py, tests/test_gold_positive_only.py.
- **P3 hierarchical head**: src/debass_meta/models/hierarchical_followup.py (+ per-survey
  calibration inside), tests/test_hierarchical_followup.py.
- **P4 anchor/blend/scorer**: src/debass_meta/models/anchor_blend.py, scripts/score_fusion_v11.py
  (imports from score_fusion_v8 — extend by import, never reimplement), tests/test_anchor_blend.py,
  tests/test_score_v11.py.
- **P5 sequence arm**: src/debass_meta/features/sequence_dataset.py, scripts/train_seq_encoder.py,
  scripts/train_seq_classifier.py (v11 flags), scripts/fetch_elasticc2.py,
  src/debass_meta/experts/local/seq_v9.py (v11 artifact routing), src/debass_meta/projectors/base.py
  + projectors/local_seq_v9.py (seq_v11 registration), tests/test_seq_v11.py.
- **P6 orchestrator**: scripts/train_fusion_v11.py (Stage-A reuse + heads + calibration +
  conformal + guards + gates + reports), jobs/run_fusion_v11_{build,pretrain,expert}.sh,
  jobs/submit_fusion_v11_chain.sh, tests/test_train_v11_smoke.py. Depends on P2–P5 interfaces.

Interfaces pinned for parallel work: HierarchicalFollowup.fit/predict_proba/save/load(dir);
anchor_blend.compute_anchor(df)/fit_alpha(cal)/apply(df); gold v11 = v10 schema + n_pos_det +
neg features; score_fusion_v11 CLI mirrors v8 flags + --blend-dir.

## 8. Non-goals / known risks
- babamul/static_safe post-hoc timing (up to +249d) is training-consistent; documented, not fixed.
- DP1 harness unchanged. sklearn stays pinned (1.6.1 venv vs 1.7.2-pickled v10 warning is benign).
- SCC GPU: use `-l gpu_c=8.0` (bare gpus=1 can land on sm_60 P100 and crash torch≥2.x).
- Risk: LSST head-2 cal starts tiny → per-survey calibration falls back to shared (rule in code).
- Risk: association yield may be small → measure and report; hierarchy stands without it.
- ELAsTiCC2 sim-to-real gap → pretraining only, never eval.

## 9. Success criteria
1. G1–G6 pass. 2. On the LSST-live benchmark: p_snia no longer suppressed (G2) and Ia-vs-rest
AUC ≥ anchor (expected modest — honest fallback to SN-ness early); SN-vs-other ≥ 0.75 target.
3. ZTF headline preserved. 4. The weekly refresh tool runs end-to-end. 5. Everything
adversarially reviewed + full test suite green locally; SCC chain prepared (submit on green).
