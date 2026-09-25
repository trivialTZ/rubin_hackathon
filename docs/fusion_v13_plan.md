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
in training; its head-1 effect is small, left for later.

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
cohort reads a median 0.33). Open: the no-broker n = 10 dip, SALT3's serving skew, Fink trust level, and more LSST SNe
in train and cal (TNS label refresh).
