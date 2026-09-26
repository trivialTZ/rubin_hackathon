# fusion v13 — draft plan (2026-09-25)

Status: v13b trained and deployed to the explorer site (2026-09-25). Builds on fusion v12 (`docs/metadebass_rubin_review_20260924.md` §6).

## Why v13

v12 fixed the Rubin labels and train/serve skews, but on the frozen Rubin benchmark (232 ids; 222 scored at "latest":
78 SNe, 144 catalogue others) it depends on *which inputs happen to be present*:

Input-availability ablation, v12 (and v12w) on the benchmark gold, expert columns blanked
(`data/label_refresh_20260924/tools/mask_experts.py`; predictions in `data/label_refresh_20260924/ablate_v12/`).
Cells: SN-vs-other AUC / median P(SN) on SNe / median P(SN) on others.

| v12 | n = 3 | n = 5 | n = 10 | latest |
|---|---|---|---|---|
| full | 0.937 / 0.93 / 0.09 | 0.894 / 0.88 / 0.08 | 0.842 / 0.67 / 0.07 | 0.929 / 0.92 / 0.07 |
| no brokers (local experts kept) | 0.880 / 0.92 / 0.10 | 0.859 / **0.32** / 0.08 | 0.826 / **0.19** / 0.06 | 0.933 / 0.89 / 0.07 |
| no local experts (brokers kept) | 0.885 / **0.21** / 0.00 | 0.820 / **0.21** / 0.00 | 0.771 / 0.09 / 0.00 | 0.887 / 0.21 / 0.01 |
| no experts at all (lightcurve only) | 0.500 / 0.00 / 0.00 | 0.5 | 0.5 | 0.5 |

v12w (weak stamp labels kept) is less calibrated with everything present (others' median P(SN) 0.42) but degrades less
without brokers (0.89/0.86/0.80/0.90, SNe median 0.75).

On the DP2 TNS-typed objects (no broker outputs exist for DP2; local experts run on their lightcurves,
`jobs/run_edp2_v12_score.sh`), v12 does not separate SNe from non-SNe. DP2 is data-rights restricted, so its numbers
are kept in the private notes.

**Why it collapses (verified by a code review, 2026-09-25).** Not NaN-as-zero. Head 1's "LSST equalization"
(`hierarchical_followup.py::_fit_head1`) masks the whole ALeRCE family on every LSST weak or context row. That family
includes the local `alerce_lc` re-run, the Rubin stamp and Babamul. In v12 those rows are all ~18k catalogue "others",
while the ~460 LSST spectroscopic rows stay unmasked, so missingness is class-pure: v12's own availability audit
recorded max |corr(avail, is SN)| = 1.0 and nothing gated on it. On the benchmark, blanking `alerce_lc` alone drops
the SN median raw P1 from 0.999 to 0.39. With all inputs present, raw head 1 gives catalogue others 0.987; the
isotonic calibrator (51 LSST cal objects) does all the separating.

Further defects found:
- `q_prior__*` trust readouts are in-sample on train rows (no out-of-fold override, unlike `q__*`), and they carry
  74% of head-1's gain.
- `q__babamul` / `q__lasair__sherlock` exist in training but never at scoring.
- G2 is partly in-sample (it scores train rows with the fitted head).
- The ALeRCE stamps (Rubin, ZTF, 2025 beta) and BHRF-top project "SN" to "non-Ia SN" and are trained on
  "top class correct". So their trust heads learn P(non-Ia SN): v12's Rubin stamp trust scores AUC 0.21 against
  "is it a supernova" on the benchmark (inverted), and it feeds head 1.
- The context family (Sherlock, Babamul) is blanked by label-provenance masking on catalogue-context rows, i.e. on
  every LSST "other" row: `avail__babamul` / `avail__lasair__sherlock` correlate 1.0 / 0.996 with the class.
- ParSNIP runs as a stub (no model): a constant (1/6, 5/6, 0) marked available. In the v12w gold it is present on
  62% of LSST spectroscopic rows (the objects v12's job ran local experts on) and on no other row, but present on
  every row at scoring: a label proxy in training that is always on at serving. Fixed at the source (a stub is now
  unavailable) and dropped from v13.
- v12w's weak rows reached every non-ALeRCE trust head. For "top class correct" experts the weak SN label
  (always non-Ia) grades every correct Ia call wrong, which is why v12w's local-expert trust heads regressed.

Other v12 facts:
- Rubin Ia|SN: 0.55 [0.42, 0.67] at latest; no approach so far (DP2-trained, ZTF-trained, GRU) beats chance on Rubin.
- Trust heads: local experts much better than v11 (seq_v11 0.93, lc_features 0.92, alerce_lc 0.82). Fink trust flat
  (snn 0.75, cats 0.73); v12w's weak rows lift them (0.81/0.81) and the Rubin ALeRCE stamp head (0.75 → 0.85).
- LSST in the v12 split: 1,178 train objects, **58 cal**, 0 test (Rubin is evaluated on the external benchmark). The
  seq-train guard diverted 191 would-be cal objects to train because `seq_classifier_v11` trained on them.
- ZTF locked test macro AUC@5: v12 0.893 [0.869, 0.913].

## v13 changes (revised after review)

**Stage A (trust):**
1. The stamp classifiers and BHRF-top get the is-SN trust target. The SN-filter set is saved in the trust artifact,
   so old models score unchanged.
2. `q_prior` out-of-fold on train rows.
3. No `q__` columns for experts without a trust head.
4. LSST weak rows are used only for is-SN trust heads outside the ALeRCE family (Fink SNN, CATS, SNGuess);
   ZTF weak rows as in v12 (`--stage-a-weak-policy lsst_is_sn_only`). `q_prior` only for experts with a trust head
   (`--stage-a-q-prior-experts trained`), with training/scoring parity tested on cal rows.

**Heads:**
5. LSST weak rows out of head 1, its calibrators, the per-survey gate and α (`--head1-exclude-quality lsst:weak`);
   ZTF weak rows stay as in v12 (they carry Ia labels; excluding them would remove every ZTF "other").
6. LSST equalization off; the context family masked on all LSST rows (`--head1-context-mask survey`) at fit AND
   serve time; ParSNIP dropped everywhere (`--drop-expert parsnip`). Guard G8: weighted |corr(avail, is SN)| on the
   final head-1 frames ≤ 0.2 (dry-run on the full gold before the SCC run sets masks/threshold).
7. Availability dropout. LSST rows get copies in the no-broker, no-local and no-expert regimes; all rows get random
   drops. Copies share the object id, and each object's total weight is unchanged.
8. Head 1 cross-fitted (5-fold). Its calibrators, α and G2 use out-of-fold train rows plus cal. Conformal stays on
   cal.

**Scoring:**
9. Serving guard: `serving_regime` and input counts in the predictions. `--require-local-experts` refuses rows
   without local experts.

**G8 dry run on the full v12w gold** (all v13 settings; head-1 frames, weighted, after dropout): LSST max
|corr| 0.17 (lc_features_bv; 0.44 before dropout), fink_lsst cats 0.13, early_snia 0.12, seq_v11 0.11; ZTF salt3_chi2
0.21, lc_features_bv 0.21, snguess 0.19, supernnova 0.17. The ZTF ones are genuine (the same fits fail on the same
lightcurves at scoring) and were there in v12, so the SCC run uses G8 ≤ 0.25. v12's layout scores 1.0 on LSST.

SCC (2026-09-25 10:50): smoke 7733594, v13 7733595, v13nd 7733596. Code synced file by file (SCC copies of the
touched files were older Mac versions; backup `_backup_pre_v13_20260925/overwritten_files.tgz`).

No gold rebuild: v13 retrains on the v12w gold, helpfulness and split (`jobs/run_fusion_v13.sh`). The control arm
v13nd is identical without dropout. Deferred: letting seq out-of-fold objects into cal; a TNS label refresh.

## Pre-registered acceptance (vs v12, same benchmark inputs)

- **Guards:** G8 ≤ 0.2 on the LSST and ZTF head-1 frames. G2 (out-of-fold), G3, G6 and G7 pass.
- **Full inputs:**
  - Rubin SN-vs-other within v12's CI at every checkpoint.
  - Raw head-1 median on benchmark others ≤ 0.5 (no saturation).
  - ZTF macro@5 in [0.869, 0.913].
- **No brokers:** median P(SN) on benchmark SNe ≥ 0.6 at n = 5 and 10, and AUC ≥ v12's.
- **Lightcurve only:** AUC ≥ 0.75 at latest. The scorer still refuses rows without local experts by default.
- **DP2 typed SNe** (local experts, no brokers): median P(SN) ≥ 0.5.
- **Trust:**
  - Stamp heads ≥ 0.85 against is-SN (v12: 0.21).
  - Fink SNN and CATS ≥ 0.81 (v12w level).
  - Local-expert heads within 0.03 of v12's.

## Results (2026-09-25, SCC job 7733595, 36 min, exit 0)

Guards: G2 (out-of-fold, n=24 cells), G3, G6, G7 pass; G8 0.214 (ZTF salt3_chi2; LSST 0.174) passes the run's 0.25
but not the pre-registered 0.2. Control arm v13nd (no dropout) stopped at G8: LSST lc_features_bv 0.444, so dropout is
what removes the class-pure availability; rerun with `G8_ACK=1 FUSION_V13_REUSE_STAGE_A=1` (7734019).

Acceptance (`data/label_refresh_20260924/tools/eval_v13_accept.py`): 28 of 35 pass.
- Full inputs, Rubin SN-vs-other: 0.941 / 0.926 / 0.907 / 0.940 at n = 3 / 5 / 10 / latest (v12 0.937 / 0.894 / 0.842 /
  0.929). Raw head 1 on benchmark others 0.41 / 0.35 / 0.30 / 0.43 (v12 0.99): no saturation.
- No brokers: SNe median P(SN) 1.00 / 0.99 at n = 5 / 10 (v12 0.32 / 0.19); AUC 0.912 / 0.908 / 0.854 / 0.924
  (v12 0.880 / 0.859 / 0.826 / 0.933: latest misses by 0.009).
- Lightcurve only: AUC 0.912 at latest (v12 0.5).
- ZTF macro@5 0.919 [0.900, 0.937]: above the no-regression band. No leak found: the head-1 frame is the 99,668 train rows
  plus their dropout copies; test rows are never fitted.
- DP2 typed SNe: n = 3 and latest miss the 0.5 median (numbers in the private notes).
- Trust on the benchmark: Rubin stamp 0.895 vs is-SN (v12 0.206); Fink SNN 0.802, CATS 0.794 (miss 0.81 narrowly;
  v12 0.748 / 0.726); local heads within 0.03 of v12.

**Not pre-registered, blocks deployment: calibration.** On the benchmark (35% SNe) v13's calibrated P(SN) is too high
on non-SNe (mean 0.53 to 0.60; Brier 0.32 to 0.36 vs v12 0.105 to 0.130; `eval_v13_supp.json`). α = 1, so this is head
1's LSST calibrator:
1. Bug: the cross-fit calibrator (`_fit_head1_calibrators_crossfit`) is fitted with the training weights, where
   catalogue-context rows carry `context_weight` 0.15. The implied SN share is 23% by weight vs 3.6% of LSST head-1 objects
   (38 SN objects of 999). Object-normalized weights without tier factors: benchmark Brier 0.146 to 0.150 (full), 0.19 to
   0.22 (no brokers), but the DP2 SN medians fall well below 0.5.
2. Remaining gap: the no-local and no-expert regimes are well calibrated under every variant (Brier 0.10 to 0.14), while
   the full and no-broker regimes are not. The local-expert inputs make benchmark non-SNe look more SN-like than the
   training catalogue non-SNe (the out-of-fold raw score on training non-SNe is 0.005; benchmark 0.43). Not a prior
   problem: shifting the calibrator's prior to 0.20 or 0.35 makes the full regime worse. Not a seq leak: seq outputs on
   its own training objects are fold-routed (out of fold).
3. At n_det = 20, training non-SNe span 11 days and benchmark non-SNe 2 days; the 38 training SNe are the thin part.

Decision at the time: v12 stays on the site; next, v13b.

**Found for v13b: local SuperNNova is a label proxy on LSST.** Head-1 SHAP on the benchmark: SuperNNova adds about
+1.8 log-odds toward SN for non-SNe and SNe alike, through the shape of its output (entropy, top-1, margin). On LSST it
is uninformative (about 0.5 / 0.5, see its module docstring), and the wrapper marked its uniform stub outputs (model
missing, conversion failure, batch failure) as available. In the v12w gold every LSST training row is a stub except about
1% of SN rows (objects re-run by v12's job); every object scored at serving gets a real output. ZTF has no stubs
(Ia|SN AUC 0.59). The gold scan of all experts shows SALT3 degenerate (margin 1) on about half the benchmark rows vs ≤ 7%
in LSST training; its head-1 effect is small. The training side of that comparison was a bug (see "SALT3 on Rubin" below).

## v13b (SCC 7734744, 9.5 min; Stage A reused from v13)

Changes: `--head1-cal-weights object` (cross-fit calibrators without the label-quality factor) and
`--head1-survey-mask lsst:supernnova` (fit and serve; an explicitly named mask also blanks `q_prior__`). At the source,
SuperNNova stubs are now unavailable (`experts/local/supernnova.py`, `tests/test_supernnova_stub.py`), as for ParSNIP.

Guards all pass (G8 0.214 ZTF / 0.173 LSST; G2 out-of-fold). Acceptance: 29 of 35.
- Full inputs, Rubin SN-vs-other 0.930 / 0.895 / 0.875 / 0.952 (v12 0.937 / 0.894 / 0.842 / 0.929); raw head 1 on
  benchmark others 0.03 / 0.02 / 0.03 / 0.04 (v13 0.41 to 0.43).
- Calibration on the benchmark (Brier): full 0.090 / 0.088 / 0.090 / 0.082 (v12 0.105 / 0.130 / 0.112 / 0.115; v13
  0.32 to 0.36); no brokers 0.12 / 0.12 / 0.11 / 0.11 (v12 0.11 / 0.11 / 0.10 / 0.12); no local 0.09 to 0.10 (v12 0.12 to
  0.25); lightcurve only 0.11 to 0.13 (v12 0.15 to 0.35).
- No brokers: SNe median P(SN) 0.89 at n = 5 but 0.36 at n = 10 (21 SNe; fail); AUC ≥ v12 except latest (0.923 vs 0.933).
- Lightcurve only: AUC 0.908 at latest. ZTF macro@5 0.919 [0.900, 0.937] (above the band, as v13).
- Trust: Rubin stamp 0.895 vs is-SN; Fink SNN / CATS 0.802 / 0.794 (just under 0.81); local heads within 0.03 of v12.
- Rubin Ia|SN still chance (0.48 [0.34, 0.61] at latest).
- DP2 typed SNe (no brokers): median P(SN) ≥ 0.5 at every checkpoint (passes; numbers in the private notes).

Control arm v13nd (no dropout, G8 acknowledged at 0.444): Rubin full-input AUC 0.68 to 0.79 with P(SN) saturated at 1.00,
so dropout is what removes the class-pure availability.

TNS × EDP2 explorer cohort (public alert data): typed ZTF objects AUC 0.934 (v12 0.894), Brier 0.013 (v12 0.016); the
12 typed Rubin SNe median P(SN) 0.89 (v12 0.93).

**Deployed:** the explorer site uses v13b (tns-edp2-explorer 1251cb2); trust stays hidden (Fink trust on SN calls in the
cohort reads a median 0.33). Open: the no-broker n = 10 dip, SALT3 (below), Fink trust level, and more LSST SNe
in train and cal (TNS label refresh).

## SALT3 on Rubin (follow-up, 2026-09-25)

Re-fitting the benchmark slices locally (sncosmo 2.12; 506 of 616 rows reproduce the gold p(Ia) to 1e-3, the rest are
near-tied χ² pairs) and scanning the v12w gold:

- **Saturation is not Rubin-specific.** p(Ia) = sigmoid(Δχ²/2) with no model covariance saturates as points and S/N
  grow. In the v12w gold, ZTF rows are degenerate 51% (SNe) and 61% (others), rising from 10% at ≤ 3 detections to 80%
  at 11–20. Benchmark SNe: 29% / 58% / 84% / 84% at n = 3 / 5 / 10 / latest. That is faster than ZTF because Rubin
  S/N is higher (median 47 on SN rows with all-positive fluxes) and there are more bands. Those clean SN rows fit well
  (best reduced χ² median 1.8) and are still 57% degenerate. SALT3's Ia call on benchmark SNe is at chance either way
  (AUC 0.53; it calls Ia on 86% of rows, 61% are Ia).
- **LSST training rows had no event times.** The LSST rows in the `salt3_chi2` and `alerce_lc` silvers were appended
  with `alert_mjd` but a NaN `alert_jd`. `_local_record_to_events` fell back to `alert_mjd` only when `alert_jd` was
  None, so every LSST event was untimed and `select_events_asof` returned all of the object's rerun_exact events. The
  gold value was the mean over every epoch, later ones included: 36 events per SALT3 row and 160 per `alerce_lc` row,
  vs 2 and 16 on ZTF. Averaging pulls saturated values to the middle, hence 9–21% degenerate in LSST training vs 50–80%
  at serving. It is also a look-ahead inside LSST training rows for these two experts. `supernnova` and
  `lc_features_bv` silvers have no `alert_jd` column and were timed correctly. Serving golds (benchmark, explorer
  cohort) are timed and unaffected. Fixed in `ingest/gold.py` (`tests/test_asof_join.py`); takes effect at the next
  gold build, so v13b still carries it.
- **SN light in the templates: real, a minority.** Of 107 spectroscopic SNe in the benchmark, 16 (15%) have only
  negative alert detections and 9 (8%) mixed signs; one has negative g, r, i and y with positive z on the same nights.
  That is what a difference-imaging template containing the SN produces. The 7 SNe with any negative point in the
  refit slices are 93% degenerate (best reduced χ² median 12.6). In training, none of the 19 spectroscopic LSST SNe
  are negative-only, but 22% of the 1,749 weak-label (stamp) SNe are. The gold already carries `n_det_neg`, `frac_neg`
  and `lc_fallback_all_negative`. DP2 photometry shows the same effect against alerts (review doc, section 4).
- **SALT3 drops strong II preferences.** `math.exp(-Δχ²/2)` overflows for Δχ² < −1420; the collector catches the
  exception and writes no row, so the most II-favouring epochs are silently unavailable.

For a v13c: rebuild the gold with the fix; a stable sigmoid; scale Δχ² by the better fit's reduced χ² (or a cap on
|Δχ²| per point); then check whether the no-broker n = 10 dip moves, since local experts carry that regime.

## v13c (2026-09-25): bug fixes, all golds rebuilt, full retrain

Fixes (code, with tests):
- **Local-expert timing** (`ingest/gold.py`, `tests/test_asof_join.py`): rows with a NaN `alert_jd` are timed from
  `alert_mjd`. `scripts/local_infer.py` now writes `alert_jd` and `survey` on every row, so mixed silvers cannot
  produce NaN event times again.
- **SALT3 mapping** (`experts/local/salt3_fit.py`, `tests/test_salt3_fit.py`): p(Ia) = sigmoid(ΔlnL / (s · n)), with
  ΔlnL = (AIC_II − AIC_Ia) / 2, n the number of fitted points and s = max(1, χ²/ndof of the AIC-preferred fit). This is
  the mean per-point likelihood ratio with the errors rescaled so the better fit has reduced χ² ≤ 1. It does not
  saturate with n, charges SALT3 for its two extra parameters, and pulls fits that both templates fail (template light,
  reduced χ² 20 to 100) towards 0.5. The sigmoid is overflow-safe, a negative fitted amplitude counts as zero flux,
  and a failed fit is marked unavailable. The fit's χ² values are kept in the silver (`raw_summary`), so the mapping can
  be re-derived without refitting. On the benchmark refits, degenerate rows 60% → 13% (SNe at n = 3 / 5 / 10 / latest:
  31 / 58 / 84 / 82% → 22 / 24 / 11 / 35%), SNe called Ia 86% → 66%; Ia|SN stays at chance (AUC 0.52).
- `scripts/collect_epoch_history.py` counts and prints runner exceptions instead of dropping them silently.
- **DP2 flux keys**: DP2 catalogue lightcurves carry `psfFlux` / `psfFluxErr`; SALT3 and `lc_features_bv` read only
  `flux` / `fluxerr` / `magpsf`, so both gave no output on any DP2 row (rows marked available with empty
  probabilities). Both now fall back to the PSF-flux keys (nJy, as in the alerts); DP2 is re-run for both.

Recomputed on SCC (three chained jobs):
1. `jobs/run_v13c_salt3_array.sh` (48 shards; SCC ran 0–23, the Mac 24–47 and all of DP2 with identical inputs): SALT3 re-fitted for every set that carries it (training silver, 12,823
   objects; frozen benchmark + hold-out; explorer cohort LSST + ZTF; DP2) through `local_infer.py`, the serving path
   (the ZTF training rows came from `collect_epoch_history.py` before).
2. `jobs/run_fusion_v13c_gold.sh`: new silvers (real copies with the SALT3 rows swapped), then every gold rebuilt with
   the timing fix: training gold, DP1 and helpfulness `*_fusion_v13c` (the v12w recipe), `bench_v13c`, explorer
   `snapshots_{lsst,ztf}_v13c`, DP2 `snapshots_v13cloc`.
3. `jobs/run_fusion_v13.sh` with `FUSION_V13_ARM=v13c`: the v13b flags, Stage A retrained. Only v13c is scored on the
   v13c golds; v12 / v13b benchmark predictions stay as they were.
4. `jobs/run_v13c_dp2_lcf.sh`: `lc_features_bv` re-run on DP2 with the flux fix (it landed after job 2 was queued),
   DP2 gold rebuilt and re-scored.

### v13c results (SCC 7739792 gold 3 h 15 min, 7739793 train 62 min + scoring 11 min, 7740520 DP2 8 min; all exit 0)

Inputs after the rebuild: SALT3 degenerate rows 58.9% → 5.9% (training silver), 69% → 8% (benchmark), 44% → 10% and
36% → 3% (explorer LSST / ZTF); DP2 now has SALT3 and `lc_features_bv` output. No NaN `alert_jd` in any gold. The
train / cal / test split is identical to v12w (0 objects moved), so the comparison isolates the fixes.

Guards all pass (G8 0.214 ZTF / 0.173 LSST). Acceptance: 23 pass, 8 fail, 4 n/a (the no-broker AUC rows compare with
v12, which is not re-scored on the v13c golds; against the v13b ablation's v12 rows they pass at n = 3 and 5 and fail at
10 and latest). Benchmark, v13c vs v13b (v12):

| inputs | metric | n = 3 | n = 5 | n = 10 | latest |
|---|---|---|---|---|---|
| full | SN-vs-other AUC | 0.913 vs 0.930 (0.937) | 0.893 vs 0.895 (0.894) | 0.851 vs 0.875 (0.842) | 0.948 vs 0.952 (0.929) |
| full | Brier P(SN) | 0.091 vs 0.090 (0.105) | 0.097 vs 0.088 (0.130) | 0.087 vs 0.090 (0.112) | 0.089 vs 0.082 (0.115) |
| full | Brier 3-class | 0.283 vs 0.300 | 0.262 vs 0.258 | 0.211 vs 0.217 | 0.338 vs 0.356 |
| no brokers | SN-vs-other AUC | 0.889 vs 0.903 (0.880) | 0.859 vs 0.862 (0.859) | 0.806 vs 0.832 (0.826) | 0.921 vs 0.923 (0.933) |
| no brokers | Brier P(SN) | 0.102 vs 0.121 | 0.107 vs 0.116 | 0.097 vs 0.114 | 0.108 vs 0.114 |
| no local | Brier P(SN) | 0.133 vs 0.091 | 0.143 vs 0.099 | 0.137 vs 0.095 | 0.128 vs 0.098 |
| lightcurve only | Brier P(SN) | 0.101 vs 0.110 | 0.116 vs 0.116 | 0.111 vs 0.109 | 0.127 vs 0.131 |

- Every difference from v13b is inside the bootstrap CIs (benchmark n = 140 to 222). v13c is more hedged: full-input
  median P(SN) 0.80 to 0.82 on SNe and 0.13 to 0.20 on others (v13b 0.91 and 0.08 to 0.11; n = 10 excluded, where
  both drop to 0.4). The anchored blend dropped α in the
  main Rubin cell (`lsst|2+|>=0.25`) from 0.75 to 0.5 through its 1-SE rule (grid best still 0.75; the Rubin
  out-of-fold blend loss is 0.436 vs 0.417 in v13b).
- Regressions: brokers without local experts over-call SN on others (median P(SN) 0.34 to 0.40 vs 0.20 to 0.21; Brier above); the
  no-broker n = 10 dip is unchanged (median P(SN) on SNe 0.30, v13b 0.36); DP2 typed SNe (local experts only) are
  scored much lower, and v13c scores them lower with or without the new DP2 SALT3 / `lc_features_bv` inputs, so this is
  the retrained stack plus the smaller α, not the DP2 inputs (numbers in the private notes).
- ZTF macro@5 0.920 [0.900, 0.938]; Rubin Ia|SN still chance (0.46 [0.34, 0.59] at latest). Trust: Rubin stamp 0.898
  vs is-SN; Fink SNN / CATS 0.781 / 0.780; local heads within 0.03 of v12.
- v13b's LSST training rows carried the timing look-ahead (local experts averaged over all epochs); the serving golds
  were timed in both runs, so the benchmark comparison is fair, and v13c is the first stack without the look-ahead.

TNS × EDP2 explorer cohort: typed ZTF objects AUC 0.935 / 0.892 / 0.873 / 0.958 at n = 3 / 5 / 10 / latest (v13b 0.921
/ 0.896 / 0.891 / 0.934), Brier 0.016 / 0.013 / 0.015 / 0.012 (v13b 0.017 / 0.014 / 0.015 / 0.013); the 12 typed Rubin
SNe median P(SN) 0.82 (v13b 0.89). At the latest detection, v13c vs v13b P(SN) on the cohort: ZTF Spearman 0.94, 0.6%
of objects move by more than 0.2; Rubin Spearman 0.93, median −0.03, 17% move by more than 0.2.

## v13d (pre-registered 2026-09-25, before the run)

Diagnosis of v13c (two reviews; scripts in the session notes, SCC `data/v13d_diag/`):
- **The model stage is not worse than v13b.** v13c head 1 + calibrator on the benchmark: full inputs Brier 0.089 /
  0.089 / 0.095 / 0.084 (v13b 0.080 / 0.095 / 0.089 / 0.085), brokers-only others P(SN) 0.005 (v13b 0.007). The
  hedging and the brokers-only over-calling come from the blend.
- **The α 1-SE rule sits on a knife edge.** Cell `lsst|2+|>=0.25`, 3-class log-loss on OOF-train ∪ cal: v13b
  α 0.75 0.498 ± 0.017 vs α 0.5 0.516 (kept 0.75); v13c 0.508 ± 0.020 vs 0.521 (fell to 0.5). The 3-class loss in that
  cell is dominated by the Ia/non-Ia split of 45 LSST SN objects (loss 1.5 to 2.1, head 2 at chance on Rubin), which
  is not what the blend is for on Rubin, and the SE counts each row (epochs and dropout copies) as independent.
- **The anchor points the wrong way on brokers-only rows** (benchmark AUC 0.43 to 0.57). Trust heads of the SN-filter
  experts target `is_sn`, so their q estimates P(SN), not P(expert right): the Rubin stamp classifier, correct on
  others (P(SN) 0.03), gets q ≈ 0 there and drops out of the pool, while Fink CATS / SNN (P(SN) 0.86 / 0.72 on the same
  others) keep q ≈ 0.2. Unweighted, the same anchor reaches AUC 0.81 to 0.85.
- **v13b's Rubin confidence was partly the timing bug.** Head 1 split on `event_count__alerce_lc` at 12 / 60 / 144:
  LSST training rows had about 160 untimed events, every serving LSST row has 8, which lands on the ZTF-like branch
  (+0.8 to +1.3 log-odds on every Rubin row). v13c still splits at 40, which serving never reaches.
- The anchor's P(Ia|SN) fallback for LSST (0.35) is a row average over 38 objects; per object it is 0.47.
- Not addressed in v13d (next): only 38 LSST spectroscopic SNe in train and 7 in cal against 961 / 56 catalogue
  others, so the LSST calibrator sees 4.2% SNe; 23 Rubin SNe with only negative detections (SN light in the
  template, 15% of the benchmark SNe) are dropped by the gold builder; `lc_features_bv` reads DP2 SNe as "other"; the
  n = 10 no-broker dip is a population effect (the SNe still detected at n = 10 are faint, low-reliability ones and
  score low at every n_det), not an n_det effect.

v13d (`FUSION_V13_ARM=v13d`): v13c's golds and Stage A, heads and blend refitted with
- `--head-drop-feature-prefix event_count__ --head-drop-feature-prefix exact__` (both heads);
- `--anchor-call-weight-sn-filter`: experts whose trust head targets `is_sn` enter the anchor with the trust of their
  call (q if they say SN, 1 − q if not);
- `--alpha-objective sn_binary --alpha-se object`: α grid, 1-SE rule and the G3 per-survey verification on the
  SN-vs-other log-loss, SE clustered by object;
- `--anchor-base-rate-unit object`.
All options default to the v13c behaviour and are stored in the artifacts (`blend.json` key `v13d`, head metadata
`v13.feature_drop_prefixes`); v13b / v13c re-score to 1e-16 with the new code (`tests/test_anchor_blend_v13d.py`).

Acceptance (fixed before the run; benchmark slices n = 3 / 5 / 10 / latest):
1. Guards G2, G3, G6, G7 pass; G8 as in v13c (0.25 bound).
2. Full inputs: SN-vs-other AUC inside v13b's bootstrap CI at every slice; Brier P(SN) ≤ v13b + 0.005 at 3 of 4
   slices.
3. Brokers only: Brier P(SN) ≤ 0.10 at every slice.
4. No brokers: Brier P(SN) ≤ v13c + 0.005 at every slice.
5. Explorer cohort, typed ZTF: AUC ≥ v13c − 0.01 and Brier ≤ v13c + 0.002 at latest.
6. Reported, not gated: DP2 typed SNe median P(SN), Rubin Ia|SN, ZTF macro@5, trust AUCs.
If 1 to 5 pass, v13d replaces v13b on the explorer site. The benchmark was used to diagnose v13c, so these settings
were chosen from the failure analysis and are fitted on train / cal only; no α or setting is picked from benchmark
scores.

### v13d result (SCC 7744078, 38 min, exit 0): fails its acceptance (10 of 15)

Guards all pass. Benchmark, v13d (v13c, v13b), SN-vs-other AUC / Brier P(SN):

| inputs | n = 3 | n = 5 | n = 10 | latest |
|---|---|---|---|---|
| full | 0.933 / 0.096 (0.913 / 0.091, 0.930 / 0.090) | 0.902 / 0.102 (0.893 / 0.097, 0.895 / 0.088) | 0.864 / 0.090 (0.851 / 0.087, 0.875 / 0.090) | 0.949 / 0.095 (0.948 / 0.089, 0.952 / 0.082) |
| brokers only | 0.942 / 0.088 (0.863 / 0.133, 0.902 / 0.091) | 0.930 / 0.097 (0.814 / 0.143, 0.849 / 0.099) | 0.894 / 0.093 (0.760 / 0.137, 0.765 / 0.095) | 0.943 / 0.103 (0.886 / 0.128, 0.909 / 0.098) |
| no brokers | 0.883 / 0.117 (0.889 / 0.102, 0.903 / 0.121) | 0.862 / 0.118 (0.859 / 0.107, 0.862 / 0.116) | 0.826 / 0.098 (0.806 / 0.097, 0.832 / 0.114) | 0.918 / 0.111 (0.921 / 0.108, 0.923 / 0.114) |

Pass: full AUC inside v13b's CI at every slice; brokers-only Brier ≤ 0.10 at n = 3 / 5 / 10; no-broker Brier at 10 and
latest; explorer ZTF Brier. Fail: full Brier (0.096 / 0.102 / 0.090 / 0.095 vs v13b 0.090 / 0.088 / 0.090 / 0.082);
brokers-only Brier at latest (0.103); no-broker Brier at n = 3 / 5; explorer ZTF AUC at latest (0.943 vs v13c 0.958;
v13b 0.934). The ranking fixes worked (brokers-only AUC +0.06 to +0.13 over v13c; the no-broker n = 10 SN median
P(SN) 0.60, v13c 0.30), the calibration did not: others keep a median P(SN) of 0.16 to 0.22 on full inputs.

Cause (blend.json): the object-clustered SE is 0.046 in cell `lsst|2+|>=0.25` and 0.078 in `lsst|1|>=0.25` (about
45 LSST SN objects), so the 1-SE rule chose α 0.25 and 0 where the out-of-fold SN-vs-other loss is lowest at 0.75
(0.279; anchor alone 0.355) and 0.5. v13d is not deployed.

### v13e (declared after v13d's result, before its own)

v13d with `--alpha-rule best`: α is the grid minimum of the out-of-fold ∪ cal SN-vs-other loss per cell. The 1-SE
rule's preference for the anchor assumed the anchor is the safe default; on LSST its out-of-fold loss is 0.36 against
0.28, and a 5-point grid fitted on cross-fitted predictions of about 1,000 LSST objects has little room to overfit.
This choice follows v13d's benchmark failure, so the benchmark is no longer a clean test for the α rule; v13e keeps
the same acceptance (criteria 1 to 5 above) and its result will be read with that caveat.

### v13e result (SCC 7744203, 12 min, exit 0): 14 of 15, but not reproducible

Benchmark, v13e (v13b), SN-vs-other AUC / Brier P(SN): full 0.939 / 0.080 (0.930 / 0.090), 0.906 / 0.090 (0.895 /
0.088), 0.867 / 0.081 (0.875 / 0.090), 0.956 / 0.084 (0.952 / 0.082); brokers only 0.938 / 0.076, 0.922 / 0.087,
0.889 / 0.078, 0.949 / 0.089; no brokers 0.892 / 0.104, 0.871 / 0.107, 0.828 / 0.094, 0.925 / 0.105 (v13b Brier 0.121
/ 0.116 / 0.114 / 0.114). Explorer cohort, typed ZTF at latest AUC 0.939 (v13c 0.958, v13b 0.934; the one fail),
Brier 0.0125; typed Rubin SNe median P(SN) 0.88 (v13b 0.89).

v13e's α in cell `lsst|2+|>=0.25` was 0.5, not the 0.75 that v13d's loss curve gave, although both runs share the
same heads (identical `model.pkl`) and anchor. The cross-fitted out-of-fold head predictions differ on every row
(up to 0.80; same distribution), and so do the calibrators (calibrated P(SN) up to 0.18 apart; model-only AUC at
n = 10 0.924 vs 0.856). Cause: `GroupKFold` orders groups by size with `np.argsort`, which NumPy 2.2 runs on SIMD
kernels whose order among equal sizes depends on the CPU. v13d ran on a node without AVX-512 (scc-me8, E5-2650 v2),
v13e on one with it (scc-612, Gold 6526Y); on one node the same `GroupKFold` call gives a different fold map with
`NPY_DISABLE_CPU_FEATURES` set to the AVX-512 features. Every cross-fitted quantity since v13 (head calibrators, α,
G2, and Stage A's out-of-fold q) therefore depended on the node the job landed on.

Fix: `models/folds.py:StableGroupKFold` (same greedy balancing, stable sort, ties by object id; identical fold map
with and without AVX-512, `tests/test_folds.py`) replaces `GroupKFold` in Stage A, both heads and the multiclass
follow-up. Saved models are unaffected (folds are fit-time only).

### v13f (declared before its run)

v13e's settings with the stable folds (Stage A still v13c's, frozen). Same acceptance. The spread between v13d's and
v13e's calibrators (same heads, two fold maps) is a measure of how fragile the LSST isotonic calibrator is with about
45 SN objects; v13f fixes one fold map, it does not remove that fragility.

### v13f result (SCC 7744369, 11 min, exit 0): 14 of 15

Guards all pass. α per LSST cell is now the grid minimum of a fold map that no longer depends on the node: 0.75 in
`lsst|2+|>=0.25` (SN-vs-other loss 0.299; anchor alone 0.355, model alone 0.330) and 0.5 in `lsst|1|>=0.25`.
Benchmark, v13f (v13b), SN-vs-other AUC / Brier P(SN) at n = 3 / 5 / 10 / latest:

| inputs | n = 3 | n = 5 | n = 10 | latest |
|---|---|---|---|---|
| full | 0.945 / 0.075 (0.930 / 0.090) | 0.907 / 0.087 (0.895 / 0.088) | 0.868 / 0.078 (0.875 / 0.090) | 0.958 / 0.082 (0.952 / 0.082) |
| brokers only | 0.951 / 0.077 (0.902 / 0.091) | 0.929 / 0.087 (0.849 / 0.099) | 0.894 / 0.074 (0.765 / 0.095) | 0.951 / 0.088 (0.909 / 0.098) |
| no brokers | 0.903 / 0.101 (0.903 / 0.121) | 0.872 / 0.103 (0.862 / 0.116) | 0.828 / 0.097 (0.832 / 0.114) | 0.931 / 0.106 (0.923 / 0.114) |

Full inputs, median P(SN) 0.92 to 0.93 on SNe and 0.05 to 0.08 on others (v13b 0.91 / 0.08 to 0.11; n = 10 excluded,
0.36 on SNe). Explorer cohort: typed ZTF AUC 0.912 / 0.910 / 0.888 / 0.947 (v13b 0.921 / 0.896 / 0.891 / 0.934),
Brier 0.017 / 0.014 / 0.015 / 0.013 (as v13b); typed Rubin SNe median P(SN) 0.93 (v13b 0.89). The one fail is
criterion 5a, ZTF AUC at latest 0.947 against v13c − 0.01 = 0.948; the fold map alone moved that number by 0.008
(v13e 0.939) and the n = 3 value by 0.04 (v13e 0.954), because the isotonic calibrators' plateaus create ties.

Not solved: the no-broker SNe still at n = 10 (median P(SN) 0.24; a population of faint, low-reliability SNe, see
the v13d diagnosis); DP2 typed SNe (local experts only) stay low (private notes); Rubin Ia|SN at chance. Next
(v13g / label work): a smoother LSST calibrator than isotonic (plateau ties cost AUC and move with the fold map),
the negative-only Rubin SNe in the gold, more Rubin SN labels. Stage A is still v13c's (fitted with the old folds on
an AVX-512 node): reusing it is reproducible, refitting it on another node type would not be until it is rerun.
