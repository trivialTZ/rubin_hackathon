# metaDEBASS on Rubin: where it stands, what EDP2 can add, and how to work (2026-09-24)

> Contains aggregate numbers derived from Rubin DP2 (EDP2) catalogues, and no per-object DP2 data. The per-object
> DP2 files behind it are in `data/edp2_train/` (gitignored). Follow-up on SALT3 and template light:
> `docs/fusion_v13_plan.md`, "SALT3 on Rubin".

## Summary

1. **On Rubin, metaDEBASS gets nearly all its skill from the brokers.** With broker inputs removed, v11 gives every
   Rubin object P(SN) ≈ 0.04 (SN-vs-other AUC 0.58–0.68 from the lightcurve alone). With brokers present it scores
   0.88–0.91 SN-vs-other on the live benchmark.
2. **The lightcurve helps the trust layer but not head 1.** Shuffling the lightcurve features between objects drops
   the trust-weighted anchor from 0.87 to 0.77 AUC (n_det = 5), while head 1's model arm goes *up*, 0.89 → 0.97.
3. **The cause is mostly the Rubin labels.** Of the ~3,950 Rubin objects v11 trained on, 3,733 carry weak labels copied
   from the ALeRCE Rubin stamp classifier's own class, 19 are spectroscopic (5 SN Ia), and 11 have catalogue-context
   labels. The test split has no Rubin objects, so no Rubin trust head has a test score.
4. **SN Ia vs other SNe on Rubin is at chance for everything tried**, including models trained on EDP2 (below). The
   ~146 typed supernovae in EDP2 are too few to fix that.
5. **EDP2 is most useful as a second real-Rubin test set, a source of non-circular "not a supernova" examples, and a
   pretraining corpus.** Its photometry is not a drop-in replacement for alert photometry: on the same visits, DP2
   difference fluxes are 11% lower in the median.
6. **Two things I got wrong or found broken today:**
   - The explorer site's metaDEBASS numbers came from the local smoke-scale v11 build (`models/*_v11`), not the SCC stack
     (`models_scc_v11/`). The scorer and the benchmark refresh script defaulted to the smoke build (fixed: the SCC stack now
     sits in `models/`). I re-scored with the SCC stack. On ZTF the smoke build was at chance and the SCC stack is not.
   - The CATS code bug (`docs/metadebass_cats_bug.md`) is still in the SCC code.

## 1. How metaDEBASS runs today

| What | Where |
|---|---|
| Active SCC clone | `/project/pi-brout/rubin_hackathon` (`.venv` = Python 3.10.12, lightgbm 4.6, torch 2.9.1+cu130). `/projectnb/pi-brout/rubin_hackathon_tmp` is stale. |
| Deployed stack | SCC `models/{trust,followup,anchor_blend,conformal}_fusion_v11` (07-05). Local copy: `models/*_v11` (moved there from `models_scc_v11/` on 2026-09-24, replacing a smoke-scale build that had been the scorer's default). |
| GRU | `seq_classifier_v12a`/`v12b` (07-07/08) and `seq_encoder_ssl_v1` (trained on the Mac). Not wired into the fusion stack. |
| v11 chain | `jobs/submit_fusion_v11_chain.sh` → build (16 cores, 6 h) → GPU pretrain (`gpu_c=8.0`, 12 h) → expert (16 cores, 10 h). `FUSION_V11_FORCE=1` forces rebuilds. |
| GRU retrain | `jobs/run_seq_v12a_retrain.sh`, `SEQ_V12_ENCODER=models/seq_encoder_ssl_v1` for v12b. |
| Live benchmark | `jobs/refresh_lsst_live.sh` runs **on the Mac**, not SCC. Last run 07-07. 232 frozen ids in `data/gold/lsst_live_locked_test.json`. |
| Nightly scoring | Not running (`submit_nightly.sh` is April-era and points at the stale path). |
| Code sync | rsync, not git. Everything on the v11/v12 path is identical on both sides. Local git has ~470 dirty entries and SCC ~138. |

SCC rules and traps:
- Never run training, scoring or big pandas on the login nodes (killed at ~15 CPU-min). Use `qsub -P pi-brout` or `qrsh`.
- `-hold_jid` does not check exit status: the v11 build "failed" G5b on float noise and the chain ran on regardless.
- `.qsub.out/.err` files append across resubmits, so old tracebacks stay above the real one.
- Home quota is at 9.54 of 10 GB.
- Nothing on SCC is newer than 07-05; the 07-07 benchmark and GRU reports exist only locally.

## 2. What the Rubin side does now (measured today, SCC v11 stack)

Live benchmark = frozen 232 ids; the spectroscopic SNe plus 142 catalogue-context "others" are scored.

| Test | n_det = 3 | n_det = 5 | latest |
|---|---|---|---|
| SN-vs-other, deployed blend | 0.888 | 0.886 | 0.905 |
| … model arm only | 0.933 | 0.897 | 0.761 |
| … trust-weighted anchor only | 0.871 | 0.872 | 0.902 |
| Brokers removed: SN-vs-other | 0.580 | 0.678 | 0.580 |
| Brokers removed: median P(SN), SNe and others | 0.04 / 0.04 | 0.04 / 0.04 | 0.04 / 0.04 |
| Lightcurve shuffled: anchor | 0.775 | 0.766 | 0.857 |
| Lightcurve shuffled: model arm | 0.963 | 0.965 | 0.795 |
| Trajectory features on (they were off at scoring) | 0.882 | 0.877 | 0.888 (vs 0.879 / 0.877 / 0.886 off) |

Other mismatches between training and scoring:
- Pitt-Google SuperNNova is a head-1 input and was available on 55% of the Rubin training rows. It is never fetched at
  scoring time because BigQuery bills.
- CATS "Periodic" is read as a non-Ia supernova vote (the CATS bug).

## 3. SN Ia vs other SNe on Rubin

AUC of P(Ia | SN). Benchmark = spectroscopic SNe in the frozen benchmark (49 at "latest", 26 Ia). DP2 = TNS-typed
supernovae with DP2 photometry. Brackets are 95% bootstrap intervals, resampling objects.

| Model | Tested on | n = 3 | n = 5 | n = 10 | latest |
|---|---|---|---|---|---|
| v11 deployed | benchmark | 0.47 [0.28, 0.67] | 0.39 [0.19, 0.60] | 0.56 [0.33, 0.80] | 0.53 [0.36, 0.69] |
| lightcurve head trained on DP2 | benchmark | 0.58 [0.39, 0.75] | 0.53 [0.32, 0.74] | 0.41 [0.18, 0.64] | 0.57 [0.40, 0.72] |
| lightcurve head trained on ZTF (2,297 SNe) | benchmark | 0.55 | 0.54 | 0.68 | 0.52 |
| ZTF + DP2 | benchmark | 0.55 | 0.53 | 0.62 | 0.59 |
| lightcurve head, DP2 5-fold CV | DP2 | 0.50 | 0.55 | 0.61 | 0.55 |
| v11 model arm (no brokers) | DP2 | 0.47 | 0.56 | 0.47 | 0.39 |
| GRU v12b | DP2 | 0.64 [0.51, 0.76] | 0.54 [0.39, 0.66] | 0.62 [0.49, 0.74] | 0.50 [0.40, 0.60] |

Every interval includes 0.5. The GRU does recognise the DP2 supernovae as supernovae (median P(SN) 0.67–0.91), which
GBM v11 without brokers does not.

## 4. What EDP2 has

- 1,742 TNS objects matched to a DP2 DiaObject. 1,537 have positive detections and give 11,143 gold rows through the
  normal builder.
- 161 are TNS-typed: 99 SN Ia, 53 other SNe, 9 not SNe. 110 are time-consistent with ≥ 3 positive S/N ≥ 5 detections
  (67 / 36 / 7). Only 44 of them are first detected within 5 days of the TNS discovery; the median first DP2
  detection is 11.5 days after discovery.
- No broker outputs: the public alert stream began 2025-10-25, and DP2 ends 2026-01-07. About 70 TNS objects have alert
  points inside the DP2 window.
- **Photometry vs alerts** (1,288 same-visit detections of 68 objects): the DP2/alert flux ratio has a median of 0.893
  (16–84%: 0.56–1.01). The pull median is −1.6σ with a scatter of 2.5σ. This is consistent with SN light in the DP2
  templates. Reliability on these detections is similar (median 0.93 in DP2, 0.98 in alerts).

What that means:
- **Can:** a second locked Rubin test set for SN Ia typing (146 typed SNe, 3× the benchmark's), independent in time
  from the live benchmark.
- **Can:** "not a supernova" lightcurves at Rubin cadence with labels that don't come from a broker, via catalogue
  cross-matches (Gaia DR3 stars and variables, AGN catalogues, SIMBAD), as was done for DP1 (`data/truth/dp1_truth_50k.parquet`).
- **Can:** a much larger SSL pretraining corpus than the 19k alert lightcurves behind `seq_encoder_ssl_v1`.
- **Cannot:** fix SN Ia typing with its own labels (the in-DP2 cross-validation shows it).
- **Cannot:** train trust heads, which need broker outputs, unless the open broker models (Fink SNN/CATS/EarlySNIa) are
  re-run on DP2 lightcurves.
- **Caution:** its fluxes are biased low against alerts, so mixing DP2 and alert rows risks the model learning the
  photometry source.

## 5. Plan

**Now (small):**
1. Push the explorer re-export built on the SCC stack (rebuilt locally; all site tests pass).
2. Done: the SCC stack now sits in local `models/*_v11`, so the scorer defaults are correct; the smoke builds are in the Trash.
3. Fix CATS (`docs/metadebass_cats_bug.md`), along with anything else needing a gold rebuild.

**Rubin labels (the main fix):**

4. Refresh the benchmark and backlog sweep. TNS now has about 870 typed objects at Dec < +10° discovered since
   2026-02-01; the July sweep matched 134 to Rubin alerts. These are labels in the deployment domain, with broker outputs.
5. Train head 1 on Rubin with spectroscopic and catalogue-context labels instead of the ALeRCE stamp class. The live-truth
   builder already makes context labels (3,147 rows for the eval cohorts), but only 11 reached v11 training.
6. Hold out a real Rubin test split, so the Rubin trust heads and G2 can be evaluated.
7. Decide on Pitt-Google: fetch it at scoring time, or drop it from head 1.

**EDP2:**

8. Freeze the 146 DP2 typed SNe as a second Rubin test set for SN Ia typing.
9. Harvest DP2 "not a supernova" lightcurves by catalogue cross-match, and test whether head 1 with lightcurve-only rows
   lifts the brokers-removed AUC (0.58–0.68) without hurting the deployed one.
10. Add DP2 to the GRU SSL corpus after correcting or checking the flux offset.
11. DP2 on SCC: keep it under the clone's `data/` (owner-only permissions). Whether other pi-brout members may read it is
    a data-rights decision.

**SN Ia typing on Rubin (longer):**

12. Fetch ELAsTiCC2 (the pretrain arm exists but was never run). Train on simulations and validate on the DP2 and
    benchmark typed SNe. Until something passes, keep showing only P(SN) for Rubin objects.

**Housekeeping:**

13. Commit the v7 → v12 work, then sync SCC by git.
14. Relax G5b or have the chain check exit codes.
15. Free some SCC home quota.
16. Wire v12b in only after it passes the benchmark and the DP2 test.

## Files

- `data/edp2_train/tools/make_edp2_lcs.py`: DP2 DiaSource → lightcurve JSON + truth (no RSP calls).
- `data/edp2_train/tools/exp_head2_lsst.py`: the SN Ia typing experiment in §3.
- `data/edp2_train/{gold,scores,ablate,silver_v12b}/`: gold, predictions and ablations behind §2–§4.
- `data/tnsx_eval_20260924/scores/predictions_tnsx_{lsst,ztf}_v11scc.parquet`: the TNS × EDP2 cohort on the SCC stack.
- `data/tnsx_eval_20260924/tools/eval_typed.py`: typed-object AUCs for that cohort.

## 6. fusion v12 results (2026-09-25)

**What changed from v11:**
- The CATS mapping is fixed.
- Rubin labels are now TNS spectroscopic types plus Gaia/SIMBAD catalogue context, instead of ALeRCE stamp classes and
  Sherlock.
- Rubin training inputs now match what scoring sees: Pitt-Google removed; Babamul and ALeRCE Rubin stamp outputs
  backfilled; local experts run everywhere.
- **v12w** is the control arm: identical to v12, but it keeps the stamp-class weak labels.

SCC jobs 7730998 (v12, 1.6 h on 8 cores, 30 GB) and 7730999 (v12w, 5.3 h). The benchmark was scored by 7731000 on
identical inputs, with v11 on gold built with its own pre-fix CATS mapping.

**Benchmark fixes found on the way.** 24 of the July benchmark ids leaked into v11's training, so they were moved to
`excluded_ids`:
- 10 sat in v11 train/cal as weak-labelled candidates;
- 14 had ZTF twins in the SCC locked ZTF train/cal (the July router checked a stale local copy of the locked split).

The benchmark now has 232 ids: 24 new spectroscopic objects in, 26 excluded.

**Guards and gates:**

| | v11 | v12 | v12w |
|---|---|---|---|
| G2 (LSST spec-Ia median p_snia) | unevaluable (n = 8) | **PASS** (n = 24, median 0.53) | PASS (median 0.34) |
| head-2 on LSST gate, Spearman | 0.26 (4 objects) | **0.50** (63 rows) | 0.13 |
| ZTF locked test, macro AUC @ n_det = 5 | 0.882 [0.859, 0.906] | **0.893 [0.869, 0.913]** | 0.886 |

**Rubin benchmark** (222 objects scored at "latest": 78 SNe, 144 catalogue others). 95% bootstrap intervals over objects.

| | n = 3 | n = 5 | n = 10 | latest |
|---|---|---|---|---|
| SN-vs-other, v11 | 0.948 | 0.930 | 0.871 | 0.936 [0.90, 0.97] |
| SN-vs-other, v12 | 0.937 | 0.894 | 0.842 | 0.929 [0.89, 0.96] |
| SN-vs-other, v12w | 0.933 | 0.911 | 0.879 | 0.936 [0.90, 0.97] |
| Ia \| SN, v11 | 0.55 | 0.46 | 0.43 | **0.39 [0.26, 0.51]** |
| Ia \| SN, v12 | 0.51 | 0.54 | 0.57 | **0.55 [0.42, 0.67]** |
| Ia \| SN, v12w | 0.55 | 0.55 | 0.27 | 0.49 [0.35, 0.63] |

Restricting the others to those a catalogue confirms changes nothing (within 0.01).

**Trust heads on the benchmark** (AUC of q against each head's own target; intervals are over rows, so they are
optimistic):

| Expert | v11 | v12 | v12w |
|---|---|---|---|
| seq_v11 (GRU) | 0.70 | **0.93** | 0.76 |
| lc_features_bv | 0.76 | **0.92** | 0.87 |
| alerce_lc | 0.47 | **0.82** | 0.78 |
| supernnova | 0.61 | **0.75** | 0.67 |
| salt3_chi2 | 0.63 | 0.70 | 0.62 |
| ampel/snguess | 0.75 | 0.81 | 0.78 |
| fink_lsst/snn | 0.77 | 0.75 | **0.81** |
| fink_lsst/cats | 0.75 | 0.73 | **0.81** |
| alerce stamp (Rubin) | no head | 0.75 | **0.85** |

**Without any broker or local-expert output**, every model collapses: median P(SN) is 0.04 for v11 and 0.00 for v12 and
v12w, including the 146 DP2 typed SNe. Every Rubin training row has these inputs, so a row without them is out of
distribution.

**Reading:**
- v12 is a modest, honest improvement: the inverted Rubin SN Ia ranking is gone, G2 passes, the local-expert trust heads
  are much better, and ZTF does not regress.
- It does not improve Rubin SN-vs-other: slightly lower at 5 and 10 detections, not significant.
- The weak labels still help the *broker* trust heads (more Rubin rows), but they bring back an inverted Ia ranking at
  n = 10.

**Next:**
1. Broker/local-expert dropout on Rubin training rows (robustness).
2. Keep weak rows for the Stage-A broker trust heads only, not for head 1.
3. Grow the Rubin spectroscopic set: only 51 Rubin SN objects reached calibration.
4. Run the local experts before scoring anything on the site (the TNS × EDP2 cohort was scored without them, which is
   out of distribution for both v11 and v12).
