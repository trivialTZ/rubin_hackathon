# fusion_v11 design — ML-soundness review (2026-07-04)

Reviewer role: ML soundness, against the actual repo code. Design under review:
`docs/fusion_v11_design.md`. Code read: `src/debass_meta/models/{multiclass_followup,
pooled_trust,expert_trust,conformal}.py`, `scripts/{train,score}_fusion_v8.py`,
`scripts/build_snapshots_fusion.py`, `src/debass_meta/features/{lightcurve,detection}.py`,
`src/debass_meta/ingest/gold.py`, all of `src/debass_meta/projectors/`, plus
`reports_from_scc/fusion_v10/*` and `data/live_eval_20260704/report/skeptic.md`.
New measurements made for this review are marked **[measured]**.

Verdict: the architecture is sound in outline (hierarchy at the right label level,
anchored shrinkage, positive-only epochs), but **five design claims are wrong or
unimplementable as written** and several gates/guards are undecidable at current cal
power. Amendments below; blocking items are marked ⛔.

---

## 1. Negative-flux bug: the design's hypothesis is wrong (§3.3) ⛔

Design §3.3 says: *"Investigate why LSST negative dets counted (likely the builder's
loader bypassing is_positive)"*. The loader does **not** bypass `is_positive`. The bug
has two real parts, in different places:

**(a) `is_positive` itself is wrong for ~60% of negative-flux LSST detections.**
`src/debass_meta/features/detection.py:163-165` sets
`is_positive = not det.get("isNegative", False)`. The Lasair `isNegative` flag does not
track flux sign. **[measured]** on 100 live-eval LSST lightcurves
(`data/live_eval_20260704/lightcurves/17*.json`, 2000 dets): 1,060 detections have
`psfFlux < 0`; **630 of them carry `is_positive=True`** (`is_positive == not isNegative`
exactly, 0 contradictions), and **78% of those mis-flagged dets are >3σ negative**
(|flux|/err quartiles 3.45 / 12.4 / 27.1, max 4554). These pass the positive filter,
count toward `n_det`, and contribute NaN mags (`flux_to_mag` returns NaN for flux ≤ 0,
`detection.py:37-39`) — the "all-NaN LC features at n_det=16" pathology.

**(b) The all-negative fallback resurrects every negative.**
`src/debass_meta/features/lightcurve.py:257-260`:

```python
pos_dets = [d for d in ndets if d.get("is_positive", True)]
if not pos_dets:
    pos_dets = ndets
```

Duplicated verbatim in `scripts/build_snapshots_fusion.py:207-212`
(`_truncated_detection_lists`, with a per-row parity assert at
`build_snapshots_fusion.py:238-254` — the fix must touch **both** copies or the assert
fires). This fallback is how the skeptic's 2024xom (0 positive / 50 negative) reached
16 gold epochs. Note also the `default=True` in `d.get("is_positive", True)`: any
detection dict lacking the key silently counts as positive.

**Amendment A1**: define positivity per survey, keyed on physics not the flag:
- LSST: `psfFlux > 0` **and** `not isNegative` (flux sign is the load-bearing test;
  keep the flag as an extra veto).
- ZTF: unchanged `isdiffpos`-based rule (`detection.py:127-128`). Do **not** use a
  uniform `flux > 0` rule: ZTF `flux` is derived from `magpsf` via `mag_to_flux` and is
  always positive (`detection.py:122-124`) — a flux-sign rule would reclassify every
  ZTF `isdiffpos=-1` detection as positive and rewrite ZTF gold wholesale.

## 2. The §3.3 truncation contract, as worded, breaks G5b ⛔

Two independent ways the locked-ZTF-gold byte-identity guard fails under the design text:

**(a) 11% of ZTF lightcurves depend on the fallback today.** **[measured]** 336/3,000
sampled `data/lightcurves/ZTF*.json` files have **zero** positive detections
(`isdiffpos` ∈ {1,−1} ints, present in 7,350/7,350 sampled dets, 3,922 negative).
Those objects' gold rows exist *only because of* the `pos_dets = ndets` fallback.
Removing the fallback globally deletes their epoch rows → row set changes → G5b fails.

**(b) Feeding negatives into the legacy 51 features changes ZTF values.** Design:
*"truncation at epoch N = all detections (incl. negatives) with time ≤ time of Nth
positive detection"*. If that window is passed to `extract_features`, then per-band
counts include NaN-mag dets (`lightcurve.py:218-220` counts by band regardless of mag
validity) and `mean_quality` includes any det with finite quality
(`lightcurve.py:236-240`). ZTF lightcurves are full of `isdiffpos=-1` dets that are
currently filtered out *before* truncation — under the new contract ZTF per-band counts
and mean_quality change → G5b fails even with the survey untouched.

**Amendment A2**:
1. The 51 legacy features keep **positive-only** input on both surveys (exactly as ZTF
   behaves today). Negatives-in-window feed **only** the new `n_det_neg / frac_neg /
   n_pos_det / t_since_last_pos / neg_run_frac` features.
2. Fallback handling must be survey-gated: keep the fallback for ZTF (or prove no
   locked-test object hits it, then remove with evidence); for LSST replace it with
   "0 positive detections ⇒ 0 epoch rows" (all-negative LSST objects are exactly the
   fading-tail/variable population the neg features are for; they can still be scored
   at their first positive epoch if one ever arrives).
3. Pin what G5b means: gold v11 **adds columns** (`n_pos_det` + neg features), so
   file-level byte identity is impossible. Define it as: rebuild restricted to the
   locked ZTF test object set, project to the v10 column set, assert frame equality
   (same rows, same order, same dtypes, `assert_frame_equal` exact). Put this in
   `scripts/assert_locked_gold_identity.py`'s contract, not just its name.

## 3. Anchor blend: G3 is overclaimed; degenerate cells are unhandled; the α fit is circular on LSST ⛔

**(a) What the construction actually guarantees.** With α ∈ {0,.25,.5,.75,1} fit per
(survey × coverage-bucket) minimizing cal log-loss and 1-SE preferring the anchor,
α=0 is in the grid, so per **fitted cell** the blended cal log-loss ≤ anchor cal
log-loss. That is all. G3 as written — *"blended ≥ anchor on every cal slice (log-loss
+ OvR AUC)"* — is **not** guaranteed:
- **AUC**: log-loss-optimal α can trade ranking for calibration on the same cell;
  nothing in the fit touches AUC.
- **Slices ≠ cells**: any slice finer than or different from the fitting grid (n_det
  slices, class-conditional slices) has no guarantee.
- **Pooled metrics**: per-cell α makes the deployed score a *piecewise mixture*.
  Even if every cell improves in log-loss, cross-cell ranking shifts, and the pooled
  benchmark AUC (the actual headline) can fall below the anchor's pooled AUC. This is
  the metric G4 and §9 are stated in.

Amend G3 to what is provable ("per fitted cell, cal log-loss(blend) ≤ cal
log-loss(anchor)") and add a post-fit cal verification: pooled log-loss + OvR AUC vs
anchor per survey; on failure collapse to a single per-survey α, then to α=0. That
repair is cal-decided, so anti-peeking discipline is preserved.

**(b) Anchor log-loss is infinite in stamp-only cells.** Stamp projectors emit exact
zeros (`projectors/alerce.py:137-155`: `p_snia ≡ 0`; `summarize_ternary` passes 0
through, `projectors/base.py:105-135`). A cal Ia row in a stamp-only cell gives the
anchor infinite log-loss → the α fit degenerates to α=1 and the 1-SE rule (an SE over
infinite losses) is undefined. The anchor must be ε-clipped (follow the
`PerClassIsotonicCalibrator` precedent: clip 1e-6 + renormalize,
`multiclass_followup.py:619-625`) *before* any log-loss comparison. Note the
Ia-capability mask already fixes the Ia axis in these cells to the base rate, which
mostly removes the ∞ — but the p_other/p_nonia axes of a stamp ternary are still exact
0/1 and need the clip.

**(c) 0-experts-fired rows have no anchor.** The v8 pool returns NaN when no usable
expert exists (`score_fusion_v8.py:273-277`, `den == 0 → NaN`). The blend must pin:
bucket "n_experts = 0" ⇒ α = 1 (pure model), and that bucket is **excluded from G3**
(there is no anchor to be ≥ than). n_experts = 1 is fine given the capability mask +
per-survey base-rate fallback — p_snia_anchor = c·p_SN is rank-equivalent to SN-ness
within the cell, matching the design's graceful-degradation story.

**(d) Out-of-support cells at score time.** Coverage-bucket edges must be fixed a
priori (constants, not cal quantiles), else score-time bucketing drifts. Any
(survey × bucket) cell with no cal support (or n below a pinned minimum) falls back to
the parent: survey-level α, then global α, then α=0-if-anchor-exists / 1-otherwise.
Emit `alpha` and the fallback level per row (the scorer already plans to emit alpha).

**(e) Circularity: the α fit will self-grade on LSST.** LSST cal is weak/context-
dominated (v10 Table 1: every locked-test `lsst` slice is `n=0`
`reports_from_scc/fusion_v10/tables_1.md`; §6 sends new spec to the frozen benchmark).
Weak LSST labels derive from the ALeRCE stamp; the anchor **contains** the stamp. On
weak-labeled cal rows the anchor's log-loss is near-zero by construction → α slams to 0
on every LSST cell for a circular reason, and G3 "passes" vacuously. Reuse the Stage-A
honesty filters (`pooled_trust.py:369-375`: drop `broker_consensus` rows; drop
`alerce_self_label` rows when grading alerce experts): either fit α only on rows whose
label source is independent of every anchor member, or provenance-mask the label-source
expert out of the anchor on those rows (mirroring
`multiclass_followup.apply_provenance_masking`, lines 223-269). Report per-cell n after
filtering; expect LSST cells to fall back to survey-level α — that is honest.

## 4. Ia-capability mask: correct membership derived from the projectors

Design §3.2 lists `IA_CAPABLE ⊇ {fink/rf_ia, fink_lsst/early_snia, salt3_chi2,
supernnova, seq_v11, …}`. The elided "…" hides the most important member and one
subtlety. Derived from `src/debass_meta/projectors/*`:

**NOT Ia-capable** (Ia|SN ratio structurally constant — exclude from the Ia axis):

| expert | evidence |
|---|---|
| `alerce/stamp_classifier`, `_2025_beta`, `_rubin_beta` (+dated variants) | `p_snia ≡ 0` — `alerce.py:137-155` |
| `alerce/lc_classifier_BHRF_forced_phot_top` | Transient→nonIa, `p_snia ≡ 0` — `alerce.py:84-103` |
| `fink/slsn` | `p_snia ≡ 0` — `fink.py:49-63` |
| `fink_lsst/snn` | fixed 50/50 split — `fink_lsst.py:38-60` |
| `ampel/snguess` | fixed 50/50 split — `ampel.py:66-83` |
| `pittgoogle/upsilon_lsst` | fixed 50/50 residual — `pittgoogle.py:62-99` |
| `lasair/sherlock`, `babamul` | context-only, emit **no** ternary at all (`lasair.py:9-23`, `babamul.py:30-44`) — they cannot enter a ternary pool on any axis; keep them out of the anchor entirely |

**Ia-capable** (projector emits a varying Ia|SN ratio):
`fink/snn` (**missing from the design's list** — it multiplies
`snn_snia_vs_nonia × snn_sn_vs_all`, `fink.py:17-28`, and is the single strongest ZTF
Ia broker), `fink/rf_ia` (`fink.py:30-47`), `fink_lsst/early_snia`
(`fink_lsst.py:97-116`; sparse — 9 objects in the live eval),
`alerce/lc_classifier_transient`, `alerce/lc_classifier_BHRF_forced_phot_transient`,
`alerce/LC_classifier_ATAT_forced_phot(beta)` (`alerce.py:55-81` — explicit SNIa
class), `pittgoogle/supernnova_lsst`, `pittgoogle/supernnova_ztf`
(`pittgoogle.py:102-140`), local `parsnip`, `supernnova`, `alerce_lc` (`local.py`),
`salt3_chi2` (`local_salt3.py`), `lc_features_bv` (`local_lc_features.py`),
`seq_v9`/`seq_v11` (`local_seq_v9.py`); dormant: `antares/oracle`,
`antares/superphot_plus` (`antares.py`), `ampel/parsnip_followme` (`ampel.py:86-126`),
`oracle_lsst` (`local_oracle.py`).

**Edge case — `fink_lsst/cats` is row-conditionally capable.** Class 11 (SN-like) maps
50/50 (ratio-uninformative) but classes 21/31 map `p_snia=0, p_nonia=score`
(`fink_lsst.py:77-90`) — a genuine *negative* Ia claim. A static mask either loses that
signal (out) or admits the arbitrary 0.5 (in). Recommendation: keep CATS **out** of the
Ia pool for v11 and document the forfeited class-21/31 signal; if it matters, implement
row-level capability (capable iff the emitted ratio ≠ the projector's constant) later.

**Drift guard**: `IA_CAPABLE` as a string set in `models/anchor_blend.py` will silently
rot as the registry grows. Add a train-time assert `IA_CAPABLE ⊆ ALL_EXPERT_KEYS` plus
a unit test that feeds synthetic events through each *non*-member's projector and
asserts the Ia|SN ratio is constant — that makes the mask self-verifying against
`projectors/`.

**Base rate fallback**: P(Ia|SN) must be per-survey and clipped away from {0,1}. A
pooled cal base rate is ZTF-dominated (Ia ≈ half the spec SN); on LSST it is a level
error (rank-neutral within a cell, but it feeds log-loss comparisons in the α fit).

## 5. Head-1 per-survey: well-posed at the is_sn level, but three defects ⛔ (a)

**(a) The 150 catalog others are double-booked ⛔.** §6 freezes *"the cleaned
2026-07-04 spec cohort (~105–120) + the 150 catalog others"* as the locked benchmark,
never in train/cal (G6). §1 and §3.1 simultaneously count "catalog others (is_sn=0)"
as head-1 LSST training rows ("weak 3,998 + catalog + spec + assoc"). G6 will
(correctly) evict them — leaving LSST head-1 negatives = weak-tier alerce-derived rows
only. Either (i) `build_truth_lsst_live.py` must harvest **new** catalog negatives
outside the frozen cohort for train/cal (state this as a deliverable with an expected
count), or (ii) split the 150 (weakening the benchmark). The design text currently
contradicts itself; pick (i).

**(b) Provenance masking creates a class-correlated availability pattern on LSST.**
`apply_provenance_masking` (`multiclass_followup.py:223-269`) sets `avail__ → 0.0` for
the alerce family on alerce-labeled rows *specifically so masked rows look like the
expert never fired* (`_mask_expert_blocks` docstring, lines 195-206). On ZTF that is
benign (stamp coverage is partial anyway). On LSST the Rubin stamp fires on ~every
object, so after masking: weak-tier rows (both classes) have stamp-invisible;
context/catalog rows (all is_sn=0) have stamp-visible; spec rows (mostly is_sn=1) have
stamp-visible. "Stamp available" then correlates with label **tier**, and tier
correlates with class → head-1 can relearn a coverage prior ("stamp fired ⇒ …") — the
exact mechanism of live-eval harm #2 that §2 promises to design against. Required:
(i) an audit metric in the train report — dependence of the avail-pattern on y within
survey=LSST train rows; (ii) equalization: mask the alerce family on **all** LSST
weak+context head-1 rows (not only alerce-labeled ones), or repurpose
`expert_dropout_augment` (`multiclass_followup.py:351-450`) to drop the stamp block on
unmasked rows at the matched rate.

**(c) The per-survey-vs-pooled gate cannot be decided by the inherited machinery.**
v8 gate frames are spec-only (`_gate_eval_frames` filters `SPEC_QUALITIES`,
`train_fusion_v8.py:218-237`) and **keep everything by default** when frames are small
(`train_fusion_v8.py:331-337`). LSST spec cal ≈ 0 (v10 Table 1 lsst n=0; §6 routes new
spec to the benchmark). As inherited, the head-1 per-survey gate would be decided on
ZTF spec rows — i.e., it cannot measure the thing it gates — or auto-kept. v11 must
define an **is_sn-level LSST gate frame**: LSST cal rows with weak+catalog labels,
honesty-filtered per §3(e) above (label source independent of the features being
graded), and preregister the default when that frame is still too small.

**(d) Weights to carry over.** Keep `w_qual × 1/n_rows(object) × w_bts`
(`compute_base_weights`, `multiclass_followup.py:276-332`) and per-head recomputed
class weights (`_class_weight_vector`, lines 335-344, binary now). One deliberate
change: `weak_weight=0.1` was tuned when weak labels were forced into subtypes; at the
is_sn level they are exact-level supervision, so gate `weak_weight ∈ {0.1, 0.3}` for
head-1 instead of inheriting 0.1 silently. Head-2 (spec-only) needs no tier weights;
keep object normalization + `bts_weight`. Provenance masking on head-2 is a near-no-op
(TNS labels are broker-independent) but keep it for uniformity.

## 6. Hierarchical calibration / conformal / FDR math (§3.1, §4)

The composition p = (P1·P2, P1·(1−P2), 1−P1) is a simplex point by construction; the
Mondrian APS (`conformal.py:81-111`) and utility/FDR selection
(`score_fusion_v8.py:369-377, 160-245`) consume only an (N,3) simplex, so "calibrate
per head, then compose" is coherent. Required amendments:

1. **Clip before compose.** Per-head isotonic can emit exact 0/1
   (`IsotonicRegression(y_min=0, y_max=1)`) → composed p_snia can be exactly 0 → ∞
   log-loss in the α fit and a vacuous G2. Clip each head to [1e-6, 1−1e-6] (precedent:
   `multiclass_followup.py:624`).
2. **Cal reuse ledger.** Cal now fits: head-1 isotonic, head-2 isotonic, optional
   Dirichlet rung, the α grid, conformal quantiles, FDR τ. Split-conformal validity
   assumes the score function was not tuned on the quantile rows; v8/v10 already
   accepted calibrator+quantiles on one cal, and v11 **adds the α fit**. Either
   sub-split cal (calibration half / conformal+FDR half) or explicitly accept and keep
   the empirical-coverage tripwire (v10 Table 6: 0.894 @ nominal 0.90,
   `reports_from_scc/fusion_v10/tables_6.md`) as a hard eval check on both locked tests.
3. **Do not recalibrate composed marginals per class.** Only a simplex-preserving rung
   (Dirichlet, which refits and renormalizes the full vector) may run after
   composition; a per-class isotonic on the composed p would break sum-to-1 and the
   hierarchy's semantics.
4. **Head-2 covariate shift** (isotonic fit on true-SN cal rows, applied everywhere) is
   acceptable — p_snia error on non-SN rows is damped by P1 — but say so in the module
   docstring; it is the standard hierarchical-calibration caveat.
5. **The graceful-degradation claim is conditional.** "When P2 is uninformative,
   p_snia degrades to SN-ness ranking" holds only for P2 ≈ constant. The live eval
   measured **anti-correlated** ZTF-flavored Ia scores on LSST (rank-corr −0.04 to
   −0.12; design §0.3) — a shared head-2 can be *worse than constant* on LSST, and G2
   checks the **level** of p_snia, not the sign of its correlation. Add an explicit
   cal-gated decision: "apply head-2 on LSST" vs "constant per-survey base rate on
   LSST", with default = constant (matches the eval's null result), plus a
   non-inversion check (Spearman(p_snia, is_Ia) ≥ 0) on whatever LSST spec rows exist
   in cal/OOF-train.
6. **Binary fallback calibrator.** For head-1/head-2 at survey-cal n<40, prefer the
   existing 2-parameter `PlattCalibrator` (`pooled_trust.py:192-224`) over a
   temperature adapter: single-parameter temperature cannot shift the intercept, and
   intercept bias (65× under-calibration) is precisely the v10 failure mode being
   guarded.

## 7. Gates/guards undecidable at current cal power (§5) ⛔ (G2)

LSST spec support in cal is ~zero by construction (§6 sends spec to the frozen
benchmark until n≥300; v10 locked test had 0 LSST spec rows at every gate,
`reports_from_scc/fusion_v10/tables_1.md`). Consequences, gate by gate:

| gate / guard | decidable? | required amendment |
|---|---|---|
| per-survey head-1 vs pooled | not on spec-only frames | define is_sn-level LSST cal frame (see §5c); preregister default = pooled |
| head-2 binary vs 3-way | ZTF-only | fine; label the verdict "a ZTF decision" in the report |
| survey-flag in head-2 | **undecidable** (needs LSST spec cal) | make the ablation the **default** (drop `survey_is_lsst` + survey-degenerate cols from head-2), not a gate — the whole v10 harm was survey-flag-mediated, and the inherited small-frame fallback *keeps* components by default (`train_fusion_v8.py:331-337`) |
| negative-flux features | decidable at head-1/weak level + on ZTF (ZTF has negatives too: 3,922/7,350 sampled dets) | gate on the is_sn-level frame, not the spec-only frame |
| tns_untyped weight | ZTF-decidable | resolve the §3.1 (w ∈ {0,0.5}) vs §6 ("78 untyped may join SSL corpora only") wording: §6 governs only the frozen LSST cohort (G6 enforces it); §3.1's gate applies to non-benchmark untyped. Say so. |
| seq_v11 in/out | gate will be decided on ZTF, where the seq expert lost twice (v9c −0.0009; v10 −0.0002 [−0.0037,+0.0036]) — it can never be judged on its declared target (the LSST Ia axis) | preregister: ZTF gate as usual; additionally report an LSST-assoc-spec cal slice iff n ≥ 30; else default OUT of the deployed blend, standalone diagnostics on the benchmark at eval time only |
| Dirichlet rung | ZTF-decidable; per-survey LSST rung will always skip (missing classes / tiny n) | fine — the ladder already degrades (`multiclass_followup.py:679-791`) |
| **G2** (LSST spec-Ia cal: median p_snia ≥ 0.15) | **unevaluable as designed** ⛔ — there are no LSST spec-Ia cal rows; the only source is `ztf_assoc_spec` with unmeasured yield | pin: G2 evaluates on (assoc-spec ∪ spec-tier LSST cal) rows, assert n ≥ 10, and if n < 10 the guard must fail loudly as "UNEVALUABLE" — never pass silently. Alternative with more support: evaluate on OOF train predictions for LSST spec-Ia train rows (associations land in train), which tests the same failure mode without touching test. |
| G1 / G4 | — | §5 calls guards "hard asserts in train/score", but G1 (locked-test AUC) and G4 (live benchmark) are **test-touching** — they must be eval-time-only checks (in `eval_fusion_v11`/benchmark scorer), never consulted during training, or the anti-peeking discipline of the v8 gate design is broken |
| G3 | see §3 | restate as per-fitted-cell cal log-loss + post-fit pooled verification w/ fallback |
| G5b | see §2 | define as v10-column-subset frame equality on locked-test objects |

## 8. Minor / consistency notes

- **Anchor expert set**: only ternary-emitting experts (prediction_type
  `class_correctness`) can join the anchor; `lasair/sherlock` and `babamul` emit no
  `proj__*__p_snia` (they are context experts) and the v8 pool's regex discovery
  (`score_fusion_v8.py:256-259`) already excludes them — keep that property in
  `anchor_blend.compute_anchor`.
- **Mixed trust semantics in the anchor**: q for SN-filter experts targets `is_sn`
  (`expert_trust.py:26-40`), for ternary experts `is_topclass_correct`. Pooling both as
  weights on the SN axis is v8 precedent and acceptable; the Ia|SN axis should weight by
  the Ia-capable experts' q only. Document the semantics in the module.
- **Pinned interface gap**: `HierarchicalFollowup.fit/predict_proba/save/load` omits
  `predict_proba_raw`, which `score_fusion_v8.py:338-342` uses to emit `p_*_raw`.
  Either add it (raw = uncalibrated-heads composition) or drop the raw columns from the
  v11 scorer contract deliberately.
- **"Grouped weak-SN loss reused for head-1"** (§3.1): the v10 grouped loss lives in
  the GRU head (`seq_classifier.py:22-23,109,177` — CE on the SN-group marginal). For a
  binary is_sn LightGBM head this reduces exactly to the is_sn target, so "reuse" is
  automatic; the sentence can be simplified to avoid implying extra machinery.
- **Conformal strata vs blend cells**: Mondrian strata stay (survey × n_det bucket)
  while α varies by (survey × coverage bucket). Marginal per-stratum coverage still
  holds if conformal is refit on the *blended* probabilities (design says "refit on v11
  probs" — make explicit that this means post-blend, deployed probabilities); per-
  coverage-bucket conditional coverage is not guaranteed — acceptable, note it.
- **`n_pos_det` column**: after the A1/A2 fix, `n_det == n_pos_det` on all v11 rows by
  construction. Keep both anyway (n_det is the epoch index contract; n_pos_det is the
  explicit metric gate), and assert their equality in the builder as a tripwire.

## 9. Summary of blocking items

1. §3.3 bug hypothesis wrong: LSST positivity must key on `psfFlux > 0` (measured:
   630/1,060 negative-flux dets flagged positive; 78% >3σ) — not on `isNegative` /
   loader plumbing. (§1 above)
2. §3.3 truncation contract + fallback removal break G5b as worded (measured: 11% of
   ZTF lightcurves are all-negative and fallback-dependent; per-band counts and
   mean_quality would change on ZTF). Negatives must feed only the new features;
   fallback survey-gated; G5b redefined as column-subset frame equality. (§2)
3. §3.2 G3 guarantee overclaimed (per-cell log-loss only; pooled AUC not guaranteed);
   anchor needs ε-clipping (stamps emit exact 0/1); 0-expert rows have no anchor
   (NaN — `score_fusion_v8.py:276-277`) and must default α=1 outside G3; the α fit is
   circular on stamp-derived LSST weak labels and needs honesty filtering. (§3)
4. §3.1/§6 double-book the 150 catalog others (frozen benchmark vs head-1 LSST
   negatives) — head-1 LSST negative support evaporates under G6 unless new catalog
   negatives are harvested for train. (§5a)
5. G2 is unevaluable at current cal composition — pin its data source, minimum n, and
   loud-fail semantics, or evaluate on OOF-train LSST spec-Ia rows. (§7)

Non-blocking but strongly recommended: Ia-capability list corrected (add `fink/snn`;
CATS documented out; self-verifying unit test) (§4); provenance-masking availability
audit + equalization for LSST head-1 (§5b); head-2-on-LSST as a gated decision with
constant-base-rate default + non-inversion check (§6.5); per-head clipping (§6.1);
survey-flag ablation as head-2 default (§7).
