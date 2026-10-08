# fusion v13k — plan and pre-registered acceptance (2026-10-08)

Status: written before any v13k gold, model or score existed. Builds on v13j (`docs/fusion_v13g_plan.md`), deployed on
the explorer site since 2026-10-05.

## Why: a future-epoch leak in the gold tables

A review on 2026-10-08 found that `gold.select_events_asof` could attach a local expert's outputs from *later*
detections to an earlier row. Its step 3 ("local re-runs without a timed JD") returned every `rerun_exact` event of
the object, timed ones included. So when no output sat at or before the row's alert, the row got all the later ones.
That happens at every epoch before an expert's first output:

- `lc_features_bv` wrote empty probabilities (but `available = True`) below 4 detections;
- SALT3, SNGuess and the local SuperNNova fail or are unavailable at some early epochs.

This is the same family as the SALT3 averaging bug fixed in v13c, which was reached through NaN alert times instead.
Static broker context (Babamul, Sherlock) is dated at query time by design and is not affected.

Rows whose selected local-expert output postdates the row's alert (v13g golds):

| set | expert | n_det 1–2 | n_det 3–4 | n_det 5–6 |
|---|---|---|---|---|
| training, ZTF | `lc_features_bv` | 94% | 48% | 0 |
| training, ZTF | `salt3_chi2` | 95% | 3% | 1% |
| training, LSST | `lc_features_bv` | 79% | 34% | 0 |
| training, LSST | `salt3_chi2` | 66% | 6% | 5% |
| benchmark | `lc_features_bv` / `salt3_chi2` / `supernnova` / `ampel/snguess` | 70 / 69 / 23 / 40% | 41 / 16 / 18 / 14% | 0 / 15 / 17 / 12% |
| hard negatives (10-04) | `lc_features_bv` | 78% | 41% | 0 |

A second, smaller problem: the `lc_features_bv` head (artifact of 2026-04-20) was trained on every labelled object of
the April truth, every current ZTF fusion test object included. The local ALeRCE LC head was already retrained on the
train fold in April and has no test overlap.

### Measured effect on v13j (serving side only)

v13j re-scored on the benchmark and hard-negative golds with the future-dated local outputs blanked (what a correct
as-of join gives):

- **Benchmark, SN vs other.** AUC 0.945 → 0.937 / 0.936 → 0.937 / 0.905 → 0.905 / 0.955 → 0.954 at n = 3 / 5 / 10 /
  latest. The only shift is at n = 3: −0.008 [−0.019, +0.001].
- **Detected hard negatives vs benchmark SNe.** 0.904 → 0.909 / 0.885 → 0.885 / 0.836 → 0.836 / 0.950 → 0.946.
- **FPR at P(SN) > 0.5.** Unchanged except at n = 3: benchmark 0.007 → 0.021, detected hard negatives 0.005 → 0.010.

The Rubin results reported for v13j therefore stand on the serving side. What the heads learned from the leaked
training values can only be measured by retraining.

On ZTF the leak matters at early epochs. In a quick LightGBM on v13j's head-2 inputs (ZTF spectroscopic SNe, v13g
train vs test split), dropping `lc_features_bv` lowered Ia-vs-other-SN AUC by 0.053 [0.039, 0.072] at n = 3, with no
change at n = 5 or 10. As a single input, `lc_features_bv` scored 0.86 at n = 3 but 0.57 at n = 5.

## v13k recipe

Built by `jobs/run_fusion_v13k_gold.sh` and trained by `jobs/run_fusion_v13.sh` with `FUSION_V13_ARM=v13k`. v13j's
settings throughout, with three changes:

1. **As-of fix.** `select_events_asof` step 3 returns only untimed local re-runs. `scripts/build_snapshots_fusion.py`
   now refuses to write a gold in which any non-static expert's selected event postdates the row's alert
   (`gold.future_selections`). `lc_features_bv` marks an epoch without probabilities unavailable.
2. **`lc_features_bv` retrained on the v13g split's `train_ids` only** (`scripts/train_lc_features_head.py
   --train-split`). Same truth file and lightcurves as the April head. Its new outputs replace the old ones in copies
   of every silver; every other expert's outputs are unchanged.
3. **Golds rebuilt.** Training, benchmark, hard-negative test, explorer cohort, DP2 and fresh-A. The v13g split is
   passed as the locked split, and the job asserts that no train / cal / test assignment moves. Stage A is refitted,
   because its inputs changed.

Fresh-B stays unfetched.

## Pre-registered acceptance (v13k vs v13j, both scored on the v13k golds)

v13k is a correctness fix, so the gates are non-inferiority checks against v13j on clean inputs. A failure means
v13j's Rubin performance leaned on leaked training values. That will be reported, and v13k will not be deployed until
it is understood.

1. **Guards.**
   - G2 (P(Ia|SN) form), G3, G6 and G7 pass, and G8 ≤ 0.25.
   - The builder's as-of audit passes on every gold (it asserts).
   - The training split is identical to v13g's.
2. **Benchmark, SN vs other.** The paired AUC difference v13k − v13j at n = 3, 5 and latest has a 95% CI lower bound
   above −0.02.
3. **Detected hard negatives (10-04 test) vs benchmark SNe.**
   - At n = 5 and latest: AUC lower bound above −0.02.
   - FPR at P(SN) > 0.5 is not above v13j's + 0.02.
4. **Brier P(SN) on benchmark SNe plus detected hard negatives.** Not above v13j's + 0.01 at 3 of the 4 slices.
5. **Fresh-A (all objects).** FPR at P(SN) > 0.5 ≤ 0.05 at n = 3, 5 and latest.
6. **ZTF locked test.** Macro AUC@5 inside v13j's 95% CI [0.904, 0.940].
7. **Explorer cohort, typed ZTF at latest.** AUC ≥ v13j − 0.01.

Paired bootstraps resample objects (1000 resamples, seed 42) as in `docs/fusion_v13g_plan.md`.

Reported, not gated:
- ZTF macro and Ia AUC at n = 3 (expected lower than v13j's, which had the leak).
- Rubin Ia|SN.
- v13j on the v13g vs the v13k golds.
- `lc_features_bv` single-input AUCs by n_det.
- DP2 typed SNe (private notes).

### Amendment (2026-10-08, after the gold job started, before any v13k output existed)

A positional audit during the host-galaxy sizing (`data/hardneg_20261004/tools/lsst_twins.py`) found that one Rubin
source often has several diaObjectIds. These twins sit within 2" of each other, or share a TNS name.

The split builder groups LSST–ZTF associations and TNS internal names, but not LSST–LSST twins. As a result:
- Two benchmark supernovae (SN 2026fgl and SN 2026qrh, 6 benchmark diaObjectIds in all) have twins in the training
  split.
- 2 hard-negative test objects (catalogue stars) have training twins.
- Fresh-A has none.
- Inside the training data there are 357 train–train and 31 train–cal twin pairs. Calibration and out-of-fold
  estimates are therefore slightly optimistic.

v13k keeps v13g's split, so it shares this. Reported, not gated: criteria 2 to 4 recomputed without these 8 test
objects. Grouping LSST twins in the split is the next fix after v13k. It changes the locked split, so it needs its own
version.

### Amendment 2 (2026-10-08, still before any v13k output): SN light in the templates

When a difference-imaging template was built while a supernova was bright, the template holds SN light. Every later
difference flux in that band is then too low by a constant, and goes negative once the SN fades below its template
level. A data audit on the alert lightcurves:

- **Benchmark (frozen manifest).** 8 of 80 spectroscopic SNe have an alert detection below −5σ (contaminated). So do
  92 of the 152 catalogue others (variables and AGN dip naturally).
- **Training side.** 38% of the Rubin spectroscopic SNe are contaminated, and 33% have no positive alert detection at
  all. These are mostly SNe that were already bright when the Rubin templates were built, which is not the
  new-transient case metaDEBASS is for.
- **v13j on the benchmark.** Contaminated SNe score median P(SN) 0.10 at n = 3 (clean SNe 0.52), and their AUC
  against the benchmark others is 0.913 (clean 0.949; at latest 0.887 vs 0.963). The model reads negative detections as
  "variable".
- **DP2 is affected far more.** Numbers are in the private notes. Uncorrected DP2 difference photometry cannot be used
  for typing until a per-band template offset is fitted.

Reported, not gated: benchmark criteria 2 and 4, plus Ia|SN, split into clean and contaminated SNe
(`leakcheck_20261008/template_flag_bench.json`: an alert detection below −5σ).

### Second data finding for the next version: forced photometry counted as detections

The cached Rubin lightcurves mix alert detections with Fink forced photometry (`_source = fink_lsst_fp`). The gold
builder counts any point with psfFlux > 0 as a detection, so 25 to 31% of counted points are forced photometry. In
training, those points have median S/N 2.0, so "n_det" partly counts noise, and the n_det = 3 epoch moves earlier on
17% of objects.

`scripts/local_infer.py` filters on the ZTF-only `isdiffpos` field, so on Rubin every point reaches the local experts:
forced, negative and positive.

Not changed in v13k. The detection definition, the template offset and twin grouping go together into the next
version.

Replacing v13j on the explorer site needs the maintainer's go-ahead.

## v13k result (2026-10-08): fails criteria 2, 3, 4 and 7, not deployed

Jobs and checks:
- Gold job 7964833 and training job 7964834 on SCC, both exit 0.
- The builder's as-of audit passed on every gold, and the training split is identical to v13g's.
- On the benchmark gold, the rows where a local expert went from available to unavailable match the earlier masking
  test exactly. Training gold: `lc_features_bv` 34,432 rows, `salt3_chi2` 24,948, `ampel/snguess` 13,672,
  `supernnova` 3,107.

Acceptance: `data/hardneg_20261004/tools/accept_v13k.py v13k v13j`. Both models are scored on the v13k golds. In the
first run, criterion 7 compared v13k on its golds with v13j on the v13g golds, which is not the paired comparison
written above. v13j has since been scored on the v13k explorer golds, and the verdict is the same.

| # | criterion | v13k vs v13j | |
|---|---|---|---|
| 1 | guards | G2, G3, G6, G7 pass; G8 0.175 | pass |
| 2 | benchmark AUC, CI lower bound > −0.02 | n = 3: +0.003 [−0.013, +0.020]; n = 5: +0.001 [−0.017, +0.023]; latest: −0.020 [−0.038, −0.005] | **fail** (latest) |
| 3 | detected hard negatives, n = 5 / latest | −0.009 [−0.039, +0.020] / −0.021 [−0.037, −0.006] | **fail** |
| 4 | Brier, benchmark SNe + detected hard negatives | 0.099 / 0.137 / 0.212 / 0.095 vs 0.083 / 0.128 / 0.195 / 0.082 | **fail** |
| 5 | fresh-A FPR ≤ 0.05 | 0 / 0 / 0 | pass |
| 6 | ZTF locked macro AUC@5 in [0.904, 0.940] | 0.918 | pass |
| 7 | explorer typed ZTF, latest | 0.912 vs 0.952 | **fail** |

Criterion 7 rests on 11 spectroscopic non-SNe: 6 TDEs, 3 CVs and 2 AGN.

Where the loss is:
- **It is in head 1.** The LSST α is 1 for both models, and the anchors agree (benchmark anchor AUC 0.923 vs 0.920 at
  latest). Head-alone AUC at latest is 0.934 vs 0.954.
- **The retrained `lc_features_bv` is weaker.** As a single SN-vs-other input on the ZTF locked test it scores 0.860 at
  latest, against 0.924 for the old head, which had been trained on those test objects. On the benchmark it scores
  0.542 vs 0.589. It is also in-sample on training rows only: out-of-sample on cal and test rows, where before it was
  in-sample everywhere on ZTF.
- **v13j holds up on clean inputs.** On the v13k golds it stays at benchmark AUC 0.936 / 0.936 / 0.906 / 0.954. The
  leak in its training data does not affect its test inputs.
- **The leak inflated early ZTF Ia|SN.** On the explorer cohort at n = 3, v13j's Ia|SN AUC is 0.866 on the v13g golds
  and 0.816 on the v13k golds.

### v13ka (isolation arm), written before any v13ka output existed

v13k's settings, Stage A refitted. Golds are rebuilt with the as-of fix, but from the v13g-era silvers, so
`lc_features_bv` keeps its original head and outputs (`FUSION_V13K_OLD_LCF=1`, then `FUSION_V13_ARM=v13ka`). v13j
and v13k are scored on the same v13ka golds.

What the result would mean:
- **v13ka matches v13j** (criteria 2 to 7 pass): the loss comes from the `lc_features_bv` retrain. The fix is then
  cross-fitted (out-of-fold) `lc_features_bv` outputs for train and cal objects, not a train-only head.
- **v13ka loses like v13k:** the as-of fix itself changes what head 1 learns, and that needs its own diagnosis.

Either way the explorer site keeps v13j until the cause is understood.

**Explorer site (2026-10-08).** The site still shows the v13j model. With the maintainer's go-ahead it is now scored
on the v13k explorer golds (tns-edp2-explorer 90304b1), so early epochs no longer carry future-dated local outputs.
Those golds also carry the retrained `lc_features_bv` head, so ZTF P(Ia) moves at later epochs too. v13j holds its
benchmark on these golds (above).
