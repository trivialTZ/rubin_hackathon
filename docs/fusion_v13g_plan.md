# fusion v13g — plan and pre-registered acceptance (2026-10-04)

Status: written before any model was scored on the new hard-negative test set. Builds on v13f
(`docs/fusion_v13_plan.md`), deployed on the explorer site since 2026-09-26.

## Why

A review of v13f (2026-10-04) found that the Rubin numbers answer an easier question than the one metaDEBASS is for,
and that the evaluation itself had worn out:

1. **The benchmark has no hard negatives.** All 150 of its catalogue "others" (50 AGN, 50 variable stars, 50 stars)
   were drawn from objects the ALeRCE Rubin stamp classifier had already called VS or AGN, then confirmed in Gaia,
   SIMBAD, VSX or Sherlock. One benchmark non-SN has a spectrum (a TDE). The SN-vs-other AUC of 0.95 measures "TNS
   spectroscopic SN vs a catalogued star or AGN a broker already rejected", not the follow-up decision among objects
   brokers call SN-like.
2. **Too few Rubin SNe.** Head 1 sees 38 Rubin SNe in train and 7 in cal; changing only the fold map (v13e vs v13f)
   moved the n = 3 AUC by 0.04, more than most version-to-version differences since v13b.
3. **The benchmark is no longer clean.** v13c and v13d were diagnosed on it and v13e's α rule was chosen after v13d
   failed on it.
4. **Trust means different things for different experts** (P(SN) for the SN-filter experts, P(top class correct) for
   the rest), is miscalibrated on the explorer population and hidden there, and the contract's follow-up target (SN
   Ia) is at chance on Rubin.
5. On the benchmark at the latest detection, the best single input alone (ALeRCE Rubin stamp, 0.87; local ALeRCE LC
   0.78; Fink SNN 0.73) is 0.04 to 0.08 below v13f at n = 3, 5 and latest and 0.015 below at n = 10. SALT3, Fink
   CATS, ParSNIP and SuperNNova carry no SN-vs-other information on Rubin (≈ 0.50).

## What the data allow (2026-10-04)

- **No fresh hold-out is possible yet.** Fink, Lasair and ALeRCE hold no Rubin alerts after the night of 2026-07-14.
  A full TNS refresh (+47 newly typed southern objects since 09-24) and the backlog sweep returned the same 181 LSST
  matches as on 09-24, all already used. The fresh hold-out tooling is ready
  (`data/label_refresh_20261004/tools/route_fresh.py`, sha1-even → test) for when the stream resumes; until then no
  Rubin positive is unseen.
- The 13-object "clean" hold-out is not clean: 10 of the 13 appear in benchmark predictions of v11 to v13f.
- **Negative-only Rubin SNe.** 23 of the 68 Rubin spectroscopic SNe with training lightcurves (16 Ia, 7 other SNe)
  have no positive detection and are dropped by the gold builder; on the benchmark truth, 15 of 107 spectroscopic
  SNe. All 23 training ones already have broker events and local-expert outputs in `silver_v13c`.
- **Hard negatives** (`data/hardneg_20261004/`): Rubin alert objects that the ALeRCE Rubin stamp classifier or Fink
  (SNN / CATS) called SN-like and that Gaia DR3 astrometry, Gaia variability or SIMBAD confirm are stars, variables or
  AGN; no TNS SN within 2"; none previously used. sha1 even → frozen test (`data/gold/lsst_hardneg_test_20261004.json`),
  odd → training.

## Product decision

The follow-up product on Rubin is a calibrated P(SN) ranking, reported with its input regime. Trust becomes a
diagnostic with one definition for every expert: `call_trust__<expert>` = P(this expert's SN-vs-not call is correct)
(`src/debass_meta/models/call_trust.py`). It does not feed the heads. P(Ia) stays hidden on Rubin.

## v13g recipe

v13f's settings, plus:
- **Labels:** the 23 negative-only Rubin SNe (`--lsst-all-negative-fallback`) and the hard-negative training half.
  The v13c split is passed as the locked split, so every v13c train/cal/test assignment stays verbatim; only new
  objects are routed. The benchmark and the hard-negative test ids are both quarantined (G6 on their union).
- **Head-1 LSST calibrator:** beta calibration (`--head1-calibrator lsst:beta`) instead of isotonic.
- **Stage A refitted** with the CPU-independent folds (v13f reused v13c's Stage A, fitted with the old folds), plus
  the call-trust heads (`--stage-a-call-trust`).
- Serving golds (benchmark, hard negatives, explorer cohort, DP2) built with the same fallback.

## Evaluation (`scripts/eval_rubin_sets.py`)

Each version is scored once on each test set; nothing below is used to choose settings.
- **Hard negatives:** P(SN) on the hard-negative test objects, AUC against the benchmark SNe at the same n_det slice,
  false-positive rate at P(SN) > 0.5.
- **Benchmark:** diagnostic only from now on.
- **Every table reports the best single input** and the paired-bootstrap difference metaDEBASS minus that input.
  The best input is picked on the test set itself, which favours the baseline.

## Pre-registered acceptance (v13g vs v13f, both scored on the v13g golds)

v13g replaces v13f on the explorer site only if all of these hold:
1. Guards G2, G3, G6, G7 pass; G8 ≤ 0.25.
2. Hard negatives at n = 5 and latest (full inputs): false-positive rate at P(SN) > 0.5 not higher than v13f's, and
   the paired AUC difference (hard negatives vs benchmark SNe) v13g − v13f has a 95% CI lower bound above −0.02.
3. Benchmark, full inputs: SN-vs-other AUC inside v13f's bootstrap CI at every slice; Brier P(SN) ≤ v13f + 0.01 at
   3 of 4 slices.
4. ZTF locked test macro AUC@5 inside v13f's CI.
5. Explorer cohort, typed ZTF at latest: AUC ≥ v13f − 0.01.

Reported, not gated: brokers-only and no-broker regimes, the negative-only SNe, Rubin Ia|SN, DP2 typed SNe (private
notes), best-single-input deltas, and per-expert call-trust reliability (ECE). A call-trust value is shown on the site
for an expert only if its ECE on the hard-negative test plus benchmark at latest is ≤ 0.10 over ≥ 30 objects.

## Result (2026-10-05)

Gold job 7869984 and training job 7869986 (SCC). Training gold: 13,430 objects, 214,858 rows; no v13c train/cal/test
assignment moved; the hard-negative training half went 485 train / 122 cal; no hard-negative test object is in any
split. The all-negative fallback kept 379 LSST objects in training (305 other, 58 non-Ia SN, 16 Ia; the SN rows include
weak stamp labels, which head 1 excludes). 290 of the 591 hard-negative test objects have no positive detection.
The LSST head-1 calibrator is beta (`head1_calibrator_kinds`); head 1 itself stayed pooled
(`lsst_is_sn_cal_objects` = 191 < 300).

### v13f on the hard negatives (deployed model, first scoring)

AUC of P(SN), hard negatives against the benchmark SNe, v13f's own golds (no fallback; 301 hard negatives with a
positive detection): 0.715 / 0.682 / 0.674 / 0.775 at n = 3 / 5 / 10 / latest (benchmark 0.945 / 0.907 / 0.868 /
0.958). P(SN) > 0.5 on 42% / 36% / 13% / 35% of the hard negatives (benchmark others 2–6%); at latest AGN 50%,
variable stars 32%, stars 29%; ALeRCE-stamp-called 46%, others 15%. The best single input is statistically tied
with v13f at every slice (local ALeRCE LC 0.772 / 0.705 at n = 3 / 5); on the stamp-called objects the stamp
classifier alone is at least as good (0.873 vs 0.762 at n = 5). Including the all-negative hard negatives (v13g golds)
v13f gives 0.881 / 0.893 / 0.737 / 0.869: it puts P(SN) ≈ 0 on lightcurves with only negative detections.

### Acceptance (both models on the v13g golds)

| # | criterion | v13f | v13g | |
|---|---|---|---|---|
| 1 | guards | — | G2, G3, G6, G7 pass; G8 max \|corr\| 0.214 | pass |
| 2 | hard negatives n = 5: FPR(P(SN) > 0.5); ΔAUC [95% CI] | 0.12 | 0.00; +0.029 [−0.003, +0.065] | pass |
|   | hard negatives latest | 0.18 | 0.01; +0.087 [+0.053, +0.126] | pass |
| 3 | benchmark AUC in v13f's CI, n = 3 / 5 / 10 / latest | 0.948 [0.907, 0.979] / 0.910 [0.837, 0.966] / 0.871 [0.762, 0.964] / 0.938 [0.896, 0.972] | 0.903 / 0.894 / 0.867 / 0.948 | **fail** (n = 3) |
|   | benchmark Brier ≤ v13f + 0.01 at 3 of 4 | 0.072 / 0.085 / 0.077 / 0.087 | 0.106 / 0.091 / 0.082 / 0.115 | **fail** (2 of 4) |
| 4 | ZTF locked test macro AUC@5 in v13f's CI | 0.921 [0.902, 0.938] | 0.922 | pass |
| 5 | explorer typed ZTF, latest AUC ≥ v13f − 0.01 | 0.947 | 0.955 | pass |

**v13g fails criterion 3 and does not replace v13f.** The explorer site keeps v13f.

Reported, not gated:
- Hard negatives with a positive detection (paired bootstrap, 1000 resamples): AUC v13f → v13g 0.715 → 0.890,
  0.682 → 0.845, 0.674 → 0.772, 0.755 → 0.938 at n = 3 / 5 / 10 / latest; every ΔAUC CI excludes 0. With
  lightcurve features only (no expert inputs) v13g still gets 0.802 / 0.778 / 0.888 (v13f 0.691 / 0.665 / 0.693), so
  the gain is not catalogue context. The hard-negative training and test halves come from one harvest, so this is an
  in-distribution test of that population.
- The FPR gain is partly a calibration shift: median P(SN) on benchmark SNe fell from 0.92 to 0.50 (n = 3) and 0.55
  (latest). That is what fails the benchmark Brier. The benchmark mixes SNe 1:2 with easy others; the new LSST cal set
  is dominated by broker-called non-SNe.
- On the stamp-called hard negatives v13g beats the stamp classifier alone at latest (+0.082 [0.000, 0.175]).
- Call-trust ECE at latest, hard negatives: ALeRCE stamp 0.038, local ALeRCE LC 0.064, Fink SNN 0.080, CATS 0.058,
  SNGuess 0.283; benchmark: 0.144 / 0.128 / 0.098 / 0.079 / 0.103. Not shown on the site (v13g not deployed).

## v13h (2026-10-05): written before any v13h output existed (SCC job 7876144)

Two diagnostics of the v13g result (`data/hardneg_20261004/eval_v13g/bench_level_v13g.json`; paired stratified
bootstrap, 1000 resamples) changed how it reads.

**The benchmark Brier criterion tested calibration on the wrong population.**
- On benchmark SNe plus the hard negatives with a positive detection (19 to 36% SNe by slice, the closest set to
  the follow-up decision), Brier P(SN) is 0.088 / 0.125 / 0.184 / 0.081 for v13g and 0.320 / 0.287 / 0.227 / 0.271 for
  v13f at n = 3 / 5 / 10 / latest. The logit shift that would best calibrate each model there is +0.2 to +0.6 for
  v13g and about −3 for v13f (−0.2 at n = 10).
- On the benchmark (15 to 34% SNe, others that a broker had already rejected), one logit shift brings the v13g head's
  Brier to 0.076 / 0.074 / 0.078 / 0.079 (v13f 0.070 / 0.085 / 0.080 / 0.085). The gap is confidence level, not shape.
- Among the 29,145 Rubin objects that ALeRCE or Fink called SN-like (n_det ≥ 3, `data/hardneg_20261004/README.md`), at
  least 33% are catalogue-confirmed stars, variables or AGN (5% of the ALeRCE-called, 50% of the Fink-tagged). A
  typical SN-like lightcurve in that stream does not warrant P(SN) ≈ 0.9.
- Correction to the Result section above: the level change is not a base-rate change. The LSST head-1 calibration
  frame was 4.2% SN (v13f) and 3.8% (v13g), object-weighted. The head now has SN-like non-SNe in training and
  calibration, so the same lightcurve is weaker evidence.

The PI's decision (2026-10-05) is that v13g's lower confidence on SNe is not a failure. Criterion 3's Brier clause is
replaced below.

**The n = 3 benchmark AUC dip came from the anchor blend, not the head.**
- The v13g head alone gives 0.945 at n = 3 against v13f's 0.951 (paired Δ −0.006 [−0.038, +0.020]).
- v13g's anchor gives 0.872, against 0.916 for v13f's.
- The served blend (LSST α 0.75 in the main cell, 0.5 / 0.25 elsewhere) gives 0.903.
- The α cells are fitted on out-of-fold train ∪ cal rows *including the head's availability-dropout copies*. On the
  real LSST rows of the same frame (training side only, `data/hardneg_20261004/tools/fitframe_cells.py`), the grid
  minimum is α = 1 in both populated cells:
  - 2+ experts, lc_cov < 0.25: log-loss 0.092 vs 0.128 at the served 0.5.
  - 2+ experts, lc_cov ≥ 0.25: 0.102 vs 0.133 at the served 0.75.
  - Over all real LSST rows, head alone 0.098 vs blend 0.123, AUC 0.913 vs 0.901.
- Serving rows are real rows, and a missing input already moves a row to another n_experts / lc_cov cell.

This was found by looking at the two test sets, and the training-side check came after. The test numbers for the head
alone (above) were seen before the rule was chosen, so v13h's test results are not an independent confirmation. Only
a fresh hold-out can confirm it: the reserve hard negatives, or new alerts once the stream resumes.

### v13h recipe

v13g's settings, Stage A reused from v13g, plus `--alpha-fit-rows lsst:original`: LSST α cells, survey fallback, base
rate and verification are fitted on real rows only. ZTF is unchanged. The heads are retrained with v13g's settings
and expected to match v13g's.

### Acceptance (v13h vs v13f, both on the v13g golds), amended

1. Guards G2, G3, G6, G7 pass; G8 ≤ 0.25.
2. Hard negatives at n = 5 and latest: as for v13g.
3. a. Benchmark SN-vs-other AUC inside v13f's bootstrap CI at every slice (unchanged).
   b. *Replaces the benchmark Brier.* Brier P(SN) on benchmark SNe plus hard negatives with a positive detection:
      not above v13f's at 3 of 4 slices, and not above v13g's + 0.01 at 3 of 4 slices. Benchmark Brier is reported,
      not gated.
4. ZTF locked test macro AUC@5 inside v13f's CI.
5. Explorer cohort, typed ZTF at latest: AUC ≥ v13f − 0.01.

Also reported:
- v13h − v13g paired on both test sets.
- The LSST α table.
- Whether the heads match v13g's.

Replacing v13f on the explorer site needs a rebuild and push, done only on the user's go-ahead.

### v13h result (SCC job 7876144): fails 3a, and the diagnosis above was wrong

With the copies left out, the in-training fit still chose LSST α 0.5 in both cells. The real rows (4,787 and 17,818)
were the same rows as in the diagnostic. The diagnostic had weighted each object equally. The α fit uses the head's
training weights, which multiply catalogue-context labels by 0.15.
`data/hardneg_20261004/tools/fitframe_weights.py` (training side only) on the real LSST rows:

| weights | SN share | 2+ experts, lc_cov < 0.25: best α (head alone, head + base-rate shift) | lc_cov ≥ 0.25 |
|---|---|---|---|
| object (the head-1 calibrator's) | 3.8% | 1 (0.092, 0.092) | 1 (0.102, 0.102) |
| head training (context × 0.15) | 20.7% | 0.25 (0.395, 0.272) | 0.5 (0.427, 0.284) |

The LSST calibrator maps P(SN) to the object mix (`--head1-cal-weights object`, v13b), but α is judged on a mix with
five times the SN share. There the calibrated head is underconfident, and the anchor raises P(SN) on SNe. So the blend
corrects the level, not the ranking. The head with only a base-rate shift beats every blend in both cells. The
dropout copies were not the cause, and the "grid minimum α = 1 on real rows" statement above holds only under object
weights.

v13h vs v13f (paired, `eval_v13h/paired_v13h_v13f.json`):
- Benchmark AUC: 0.904 / 0.888 / 0.861 / 0.947. **3a fails at n = 3** (v13f's CI lower bound 0.907).
- Brier, benchmark SNe plus detected hard negatives: 0.095 / 0.135 / 0.196 / 0.091 (v13f 0.320 / 0.287 / 0.227 /
  0.271).

v13h vs v13g:
- Benchmark: identical within ±0.006.
- Detected hard negatives: slightly worse, AUC −0.015 / −0.034 / −0.069 / −0.018, all CIs below 0. The main LSST cell
  moved from α 0.75 to 0.5.

v13h is a null result and is not deployed.

### v13i: written before any v13i output existed

v13g's settings, Stage A from v13g, plus `--alpha-weights lsst:object`: the LSST α rows are weighted as the LSST
head-1 calibrator's rows are, without the label-quality factor. The dropout copies stay in. ZTF is unchanged. Rationale
(training side only): α should be judged on the population P(SN) is calibrated to, otherwise it silently recalibrates
toward another one. Acceptance: the amended criteria above, unchanged.

The test-set numbers for the head alone were already seen, and v13i's training-side α is expected near 1, so a pass is
again not an independent confirmation.

### v13i result (SCC job 7876363): passes 2 to 5, fails guard G2 by 0.005

The fitted LSST α is 1 in three of the four cells (0.75 in the 1-expert, lc_cov ≥ 0.25 cell) and in the survey
fallback. The heads and the anchor are identical to v13g's (max |Δ| 7e-15 on the benchmark), and the ZTF predictions
are identical (explorer cohort max |Δ| = 0).

| # | criterion | v13f | v13i | |
|---|---|---|---|---|
| 1 | guards | — | G3, G6, G7 pass; G8 0.214; **G2 fails**: median P(Ia) on the 39 LSST spec-Ia OOF/cal rows 0.145 (min 0.15; v13g 0.150, v13h 0.174), max 0.476 | **fail** |
| 2 | hard negatives n = 5: FPR; ΔAUC v13i − v13f [95% CI] | 0.12 | 0.00; +0.050 [+0.017, +0.087] | pass |
|   | latest | 0.18 | 0.01; +0.097 [+0.063, +0.133] | pass |
| 3a | benchmark AUC in v13f's CI, n = 3 / 5 / 10 / latest | CIs as above | 0.945 / 0.936 / 0.905 / 0.955 | pass |
| 3b | Brier, benchmark SNe + detected hard negatives (≤ v13f; ≤ v13g + 0.01) | 0.320 / 0.287 / 0.227 / 0.271 | 0.087 / 0.126 / 0.189 / 0.079 (v13g 0.088 / 0.125 / 0.184 / 0.081) | pass |
| 4 | ZTF locked test macro AUC@5 in v13f's CI | 0.921 [0.902, 0.938] | 0.922 | pass |
| 5 | explorer typed ZTF, latest | 0.947 | 0.955 (identical to v13g) | pass |

Reported, not gated:
- v13i vs v13g. Benchmark AUC +0.041 / +0.042 / +0.038 / +0.007; the n = 3 and n = 5 CIs exclude 0. Detected hard
  negatives +0.014 / +0.040 / +0.064 / +0.012. Brier within ±0.006 on both sets.
- Benchmark Brier: 0.110 / 0.093 / 0.085 / 0.121 (v13f 0.072 / 0.085 / 0.077 / 0.087). This is a level difference, no
  longer gated.

G2 guards against an LSST Ia collapse with an absolute threshold on P(Ia) = P(SN) × P(Ia|SN). With the lower P(SN)
level, the median falls with it. P(Ia) is not shown on Rubin (science contract). Under the rule as written, v13i does
not replace v13f. Waiving or redefining G2 would be a second post-hoc change after seeing the result, so it is the
PI's decision.

Every number above comes from test sets now used three times (v13g, v13h, v13i), with settings changed after each
look. The test that would count is one no version has seen: the 4,289 test-routed reserve hard negatives
(`reserve_not_fetched.parquet`), or new alerts once the stream resumes.

## v13j and a fresh test (2026-10-05): written before any v13j output or fresh-set data existed

The PI asked for the best model rather than a ruling on G2 alone. Decisions:

1. **Candidate: v13i's models.** Its LSST α rule is justified on training data alone, and nothing about the models
   changes below.
2. **G2 is evaluated on P(Ia|SN)** (`--g2-metric p_ia_given_sn`, same thresholds and rows; both medians reported).
   G2 was written as an anti-Ia-regression check: the v10 failure was LSST P(Ia) suppressed. Since the 2026-10-04
   contract, P(SN) is the Rubin product, its level is a deliberate choice checked by criterion 3b, and P(Ia) is not
   shown on Rubin. On P(Ia|SN), G2 still catches the Ia axis collapsing, without also policing the P(SN) level. This
   change was made after seeing v13i fail the old form by 0.005, so it does not on its own justify deployment.
   Deployment is gated on the fresh test below. v13j = v13i + this G2 form (SCC job 7879724); its models and outputs
   are v13i's.
3. **Fresh hard negatives decide.** The test-routed half of `reserve_not_fetched.parquet` (4,289 objects, alert data
   never fetched, never scored) is split again by an independent hash: sha1("fresh20261005|" + id), first hex digit
   even → fresh-A (fetched and scored now), odd → fresh-B (not fetched, held for the next version).
   - All 528 ALeRCE-called confirmed objects were already used, so fresh-A is Fink-tag-only: catalogue stars, AGN and
     a few variables.
   - The positives are still the benchmark SNe. FPR on fresh-A is a fully clean measurement; AUC is clean on the
     negative side only.
   - v13f, v13g and v13j are each scored once on fresh-A (full inputs, the v13g gold recipe with the all-negative
     fallback).

v13j replaces v13f on the explorer site only if criteria 1 to 5 hold (G2 in its P(Ia|SN) form) and:
- **F1.** FPR at P(SN) > 0.5 on fresh-A (all objects) at n = 3, 5 and latest: not above v13f's, and ≤ 0.05.
- **F2.** AUC, fresh-A against the benchmark SNe, at n = 5 and latest: v13j − v13f paired 95% CI lower bound > −0.02.
- **F3.** The same against v13g: v13j − v13g lower bound > −0.02.

Reported, not gated:
- n = 3 and 10.
- The subset with a positive detection.
- Results by subtype (star / AGN / variable).
- Median P(SN) on fresh-A.

A failure is reported as such, and the site stays on v13f.

### v13j and fresh-A results (2026-10-05)

**v13j (SCC job 7879724, exit 0).** All guards pass. G2 on P(Ia|SN): median 0.524, max 0.979 over 39 rows; on
p_snia: 0.145 and 0.476, as for v13i. Its models and outputs are v13i's, so criteria 2 to 5 are v13i's (all pass).

**Fresh-A (SCC job 7880667).**
- Coverage: alert data for all 2,176 objects. ALeRCE classifier fields for 2,174, after a re-fetch.
  - The first fetch ran ALeRCE backfill and lightcurves side by side. The ALeRCE adapter recorded the throttled
    requests as "no data", and 10 lightcurve chunks got HTTP 429.
  - The bad ALeRCE output is kept in `bronze_superseded/`. Both steps were re-run one after the other.
- 1,965 of the 2,176 (90%) have no positive alert detection. Fink-tag-only "SN-like" objects are mostly stars and
  AGN with negative difference flux (the 10-04 test set's negative-only objects were also mostly Fink-only). The
  gated "all objects" numbers are therefore easy, and the 211 with a positive detection carry the information.
- Scored once by v13f, v13g and v13j (`data/hardneg_fresh_20261005/fresh_eval.json`).

| fresh-A | n = 3 | n = 5 | latest |
|---|---|---|---|
| all: n negatives; FPR v13f / v13g / v13j | 2,011; 0.002 / 0 / 0 | 1,228; 0 / 0 / 0 | 2,176; 0.003 / 0 / 0 |
| all: AUC v13f / v13g / v13j | 0.996 / 0.968 / 0.990 | 0.997 / 0.974 / 0.989 | 0.979 / 0.987 / 0.995 |
| all: v13j − v13f [95% CI] | −0.006 [−0.013, −0.000] | −0.008 [−0.017, −0.002] | +0.016 [+0.000, +0.039] |
| all: v13j − v13g | +0.022 [+0.007, +0.043] | +0.015 [+0.001, +0.034] | +0.008 [+0.001, +0.018] |
| detected: n negatives; FPR v13f / v13g / v13j | 72; 0.056 / 0 / 0 | 26; 0 / 0 / 0 | 211; 0.028 / 0.005 / 0 |
| detected: AUC v13f / v13g / v13j | 0.891 / 0.941 / 0.947 | 0.881 / 0.941 / 0.959 | 0.900 / 0.971 / 0.978 |
| detected: v13j − v13f | +0.056 [+0.004, +0.115] | +0.078 [+0.010, +0.164] | +0.078 [+0.030, +0.132] |
| detected: v13j − v13g | +0.006 [−0.009, +0.025] | +0.019 [−0.013, +0.057] | +0.007 [−0.003, +0.022] |

F1, F2 and F3 hold at every gated slice: **v13j passes**.
- On all objects, v13j is slightly below v13f at n = 3 and 5. v13f puts P(SN) ≈ 0.000 on negative-only lightcurves,
  v13j about 0.01.
- On the objects with a positive detection, v13j beats v13f at every slice (CIs exclude 0) and ties v13g.
- FPR at latest by subtype: v13f flags 14% of the 28 variables (v13g 4%, v13j 0%); AGN and stars ≈ 0 for all three.
- Median P(SN) on detected fresh negatives: 0.17 (v13f), 0.07 (v13g), 0.015 (v13j).

Limits:
- The positives are the benchmark SNe, used throughout.
- Fresh-A has no ALeRCE-called objects: the hardest negatives, where v13f was weakest (46% FPR on the 10-04 set),
  were all used before. Fresh-A shows the gain generalises to a new negative population, not to that one.

By the acceptance above, v13j replaces v13f on the explorer site. Rebuilding and pushing the site needs the
maintainer's go-ahead. Fresh-B (2,103) stays held.
