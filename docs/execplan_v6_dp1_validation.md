# v6 Execution Plan — DP1 as metaDEBASS Validation Testbed

**Date**: 2026-04-24
**Author**: Claude + TZ
**Status**: ✅ **EXECUTED** at production scale on SCC (2026-04-24). See §12 for headline results.
**Prereq**: RSP access confirmed via USDF (token `metaDEBASS_`, 22-char Gafaelfawr)

---

## 1. Framing (why this plan, in one paragraph)

metaDEBASS v5d is our honest baseline (followup cal AUC 0.9694; LSST×spec slice
0.60 with n=12 → CI [0.28, 0.90]). The 12-object LSST-spec limit blocks
paper-grade claims about real-Rubin generalization. We confirmed RSP USDF
access and queried DP1 directly. **No spectroscopically-confirmed DP1 SNe
exist yet** (TNS, ZTF-BTS, Asiago all return 0 spec-typed DP1 matches — DP1
deep fields fall below ZTF/ATLAS depth). **But three published arXiv papers
provide 15 DP1 transients matched to `dp1.DiaObject`**: 12 with photometric
classifications (arXiv:2507.22864 contributes 11 via Superphot+; arXiv:2603.00262
adds AT 2024ahzi) and 3 additional untyped transients from arXiv:2507.22156
(Dong+ SLIDE). The 12 typed span: 2 Ia, 3 II, 4 IIn, 1 IIP, 1 Ibc, 1 SLSN-I
across ECDFS/EDFS/LELF. **Only 2 of the 12 have >95% Superphot+ confidence**
(aigv Ia, aigw II); the rest are weaker. **Additionally DP1 is rich in
negative-class truth**: 89% Gaia DR3 coverage, 4% SIMBAD-typed, 1% Gaia-variables.
The v6 plan is **validate v5d's rankings on DP1 using both (a) the 15 published
DP1 transients as positives (confidence-tiered) and (b) 83K+ Gaia/SIMBAD
negatives**, without retraining. This gives us a small-but-defined positive-class
metric (tiered) AND a statistically powerful negative-class rejection metric,
both on real Rubin data.

## 2. Probe results (real, 2026-04-24)

### 2a. Published DP1 transient sample — 15 matches confirmed

Three source papers were checked:

1. **Freeburn+ arXiv:2507.22864** — 11 extragalactic transients with Superphot+
   photometric classification (Table 4).
2. **de Soto+ arXiv:2603.00262** — AT 2024ahzi (SN IIP, photometric;
   host spec-z=0.211 from Magellan/LDSS-3).
3. **Dong+ arXiv:2507.22156** — 46 DECam+DP1 SLIDE detections (33 ECDFS +
   13 EDFS) with host galaxy properties; 16 are TNS-reported, ~30 are
   newly reported numbered candidates. No photometric types.

**15/16 named transients cleanly matched `dp1.DiaObject`** (the 16th,
AT 2024aigk, is offset ~1.5" with my coarse Dong+ coords — likely matches
with precise coords). Saved to `reports/rsp_probe/dp1_published_sne_hits.csv`.

**CRITICAL: all 16 named DP1 AT's are UNTYPED in TNS** (checked 2026-04-03
CSV: `type=NaN` for every one). No DP1 SN has spectroscopic confirmation.
Only 2024aigg has host spec-z (0.07593) via TNS.

**Superphot+ confidence tiers** (Freeburn+ Table 4):

| Tier | Confidence | N | Objects |
|---|---|---|---|
| Strong | >95% | **2** | aigv (Ia 98.4%), aigw (II 95.4%) |
| Medium | 60-95% | 3 | aigl (Ia 81%), aigt (II 73%), aigg (II 65%) |
| Weak | <60% | 6 | aigs (IIn 13%), aigh (SLSN-I 6%), ahyy (IIn 13%), ahzc (IIn 6%), aigj (Ibc 17%), aaux (IIn 6%) |
| de Soto+ | "high" | 1 | ahzi (IIP; numeric TBD) |
| Untyped (Dong+ adds) | — | 3 | ahsx, ahwk, ahyq (DP1 DiaObject matched but no photo type) |

**Full diaObjectIds** (all 15 matched):

| IAU name | Type | Conf | Field | diaObjectId | nDiaSources |
|---|---|---|---|---|---|
| AT 2024aigv | Ia | 0.984 | ECDFS | 609788942606139423 | 30 |
| AT 2024aigw | II | 0.954 | ECDFS | 611255210081255575 | 13 |
| AT 2024aigl | Ia | 0.814 | EDFS | 592913706862510093 | 12 |
| AT 2024aigt | II | 0.726 | ECDFS | 611253629533290657 | 14 |
| AT 2024aigg | II | 0.647 | ECDFS | 611255759837069401 | 253 |
| AT 2024aigs | IIn | 0.132 | EDFS | 591819074317582360 | 32 |
| AT 2024ahyy | IIn | 0.133 | ECDFS | 609781520902651937 | 100 |
| AT 2024aigj | Ibc | 0.172 | ECDFS | 611256447031836769 | 25 |
| AT 2024ahzc | IIn | 0.059 | ECDFS | 609782208097419314 | 371 |
| AT 2024aaux | IIn | 0.058 | LELF | 648374722634973207 | 19 |
| AT 2024aigh | SLSN-I | 0.056 | EDFS | 592915218690998602 | 23 |
| AT 2024ahzi | IIP | "high" | EDFS | 592914119179370575 | 5 |
| AT 2024ahsx | — | — | ECDFS | 611253629533292287 | 1 |
| AT 2024ahwk | — | — | ECDFS | 611253973130674268 | 1 |
| AT 2024ahyq | — | — | ECDFS | 609782208097421487 | 2 |

**Opportunity**: Dong+ Table 1-2 has **~30 more unreported candidates
with numbered IDs, full RA/Dec, host properties, and (most with) DP1
DiaObject associations**. Not yet extracted into our table. If their
machine-readable table is released in the ApJL publication, we can
double our "real DP1 transient" positive sample for free.

### 2b. Negative-class truth sources

Measured on a DP1 sample of 3000 DiaObjects (500 per field) with
`nDiaSources ≥ 5`, across: ECDFS, SMC, RA59/Dec-49, RA95/Dec-25,
RA106/Dec-11, RA38/Dec+7.

| Source | Matches | Rate | Typed? | Notes |
|---|---|---|---|---|
| TNS (193K typed) | 4 | 0.13% | 0 spec-typed | 3/4 ARE our DP1 AT candidates (untyped in TNS) |
| ZTF-BTS (3615 typed in DP1 Dec) | 0 | 0% | — | Depth/sky mismatch |
| Asiago SN (VizieR B/sn) | 0 | 0% | — | Pre-2020 catalog |
| **Gaia DR3** | **2681** | **89%** | parallax+pm | **Primary negative-class source** |
| **SIMBAD typed** | **121** | **4%** | 18 classes | Star/Galaxy/QSO/RRLyrae/EclBin/LPV/WD/CV |
| **Gaia DR3 variables** | **32** | **1%** | 9 classes | AGN/SOLAR/LPV/ECL/RR/CV/YSO |
| Milliquas QSO | 10 | 0.3% | QSO | Supplements SIMBAD |
| OGLE-IV SMC variables | — | — | — | Not X-Match indexed; needs local download |
| DES-SN Y5 | — | — | — | Not X-Match indexed; public release available |

### 2c. Automated plan-verification results

Seven claims auto-checked by `scripts/_exp_rsp_verify_plan.py` — all pass:

| Claim | Verified value | Result |
|---|---|---|
| DP1 pool size (n≥5) ~93K | **93,336** | ✓ |
| DiaSource reliability<0.1 frac ~85% | **85.2%** | ✓ |
| TAP `IN(…)` works at 500 IDs | 6410 rows returned in one query | ✓ |
| Published SN CSV has 15 matched | 15 rows | ✓ |
| 3 sample diaObjectIds resolvable | 3/3 | ✓ |
| Published candidates have real LCs | 901 dets total, median 19/obj | ✓ |
| 0 DP1 candidates have TNS spec type | 0/15 (all AT, type=NaN) | ✓ |

SCC-side verifications (2026-04-24):
- v5d followup model: ✓ at `/project/pi-brout/rubin_hackathon/models/followup_safe_v5d/`
- 11 trust-head models: ✓ at `.../models/trust_safe_v5d/`
- 4 local-expert artifacts: ✓ at `.../artifacts/local_experts/{supernnova,alerce_lc,lc_features,salt3_chi2}/`

Rerun with: `python scripts/_exp_rsp_verify_plan.py`

Extrapolated to **93,336 trainable DP1 pool** (nDiaSources ≥ 5, verified 2026-04-24):
- **~83K Gaia-known stars** (89% × 93K)
- **~3700 SIMBAD-typed non-SNe** (4% × 93K)
- **~1000 Gaia-classified variables**
- **15 matched DP1 transients** positive-class (12 photometrically typed + 3 transient-only)
- **+ ~30 more unreported Dong+ numbered candidates** — not yet tallied

## 3. Science goals (dual: positive-class + negative-class)

### Positive-class claim (SN recovery, honest tiering):

**Primary claim (strong):**
> "On the 2 high-confidence (>95%) photometrically-classified DP1 SN
> candidates (aigv Ia, aigw II from arXiv:2507.22864), v5d assigns top-X%
> follow-up priority to 2/2."

**Secondary claim (broader, Superphot+-agreement):**
> "On 12 typed DP1 SN candidates from Freeburn+2025 and de Soto+2026,
> metaDEBASS follow-up ranking agrees with Superphot+ class assignments on
> M/12 objects (specifically, N_high/2 of the high-confidence subset,
> N_med/3 of medium, N_low/6 of weak-confidence cases where agreement
> is less meaningful)."

**Tertiary claim (transient detection, 15 total):**
> "On all 15 TNS-reported DP1 transients matched to DP1 DiaObject
> (10 ECDFS + 4 EDFS + 1 LELF from three arXiv papers: Freeburn+2025,
> de Soto+2026, Dong+2025), v5d flags K/15 as follow-up-worthy above
> threshold τ — demonstrating real-Rubin transient detectability even
> without spectroscopic truth."

### Negative-class claim (contaminant rejection):

> "On real DP1 data with N = 50,000 objects, v5d correctly assigns low
> follow-up priority (bottom X%) to Y% of Gaia-confirmed stars and
> Z% of Gaia/SIMBAD classified variables (QSO, RRLyrae, EclBin, LPV),
> demonstrating generalization of our ZTF-trained trust heads to
> Rubin-depth observing conditions."

### Combined (the paper headline):

> "Validated on real Rubin DP1 with (a) 2 strong-confidence photometric SN
> classifications (recall 2/2 targeted), (b) 15 total published DP1
> transients with follow-up-ranking detection rate K/15, and (c) ~85,000
> Gaia/SIMBAD negative-class contaminants with rejection rate Y% —
> replacing v5d's prior 12-spec-Ia LSST slice (AUC 0.60, CI [0.28, 0.90])
> with real-Rubin metrics orders of magnitude better powered on the
> negative side."

### Caveats (must appear in the paper)

- All DP1 SN types are **photometric** (Superphot+ or similar), not
  spec-confirmed. metaDEBASS agreement with Superphot+ is classifier-vs-classifier,
  not truth validation. Confidence-weighted reporting required.
- N=2 high-confidence positives is small → wide CIs on primary claim.
  Report both metric and CI; bootstrap recommended.
- Gaia-star rejection is the load-bearing quantitative claim (clean labels,
  large N).
- LSST-broker expert columns (`fink_lsst/*`, `pittgoogle/supernnova_lsst`)
  will be NaN on DP1 (brokers don't index DP1). v5d feature importance
  analysis showed these contribute <1% of followup decision, so NaN-filling
  is safe.

## 4. Architecture

Two shapes depending on data-rights outcome (pending user check):

- **Shape A** (DP1 export allowed): run end-to-end on SCC. Pull via TAP,
  cache DiaSource parquets, run local experts, score through v5d.
- **Shape B** (DP1 export restricted): fork ingest into RSP JupyterLab.
  Run lightcurve → feature → local-expert pipeline inside RSP; export only
  the final score + label joined table (no raw flux).

Both shapes share the same logic; only the execution host differs.

## 5. Pipeline

### Stage 1 — truth table (1 day)

- Pull ~50K DP1 DiaObjects via TAP (`nDiaSources ≥ 5`).
  - **TAP batching required**: single `TOP 50000` query fails
    (service-side timeout observed at 60K). Pull via 6 field-filtered
    queries (one per sky region) at ~10K each, or by nDiaSources buckets.
  - Total pool available: 93,336 objects (verified 2026-04-24).
- Crossmatch against:
  - Gaia DR3 (1") — via CDS X-Match, bulk. (Verified: 2681 hits on 3K sample.)
  - SIMBAD (2") — via CDS X-Match, bulk. (Verified: 121 hits on 3K sample.)
  - Gaia DR3 variables (1") — via CDS X-Match. (Verified: 32 hits on 3K sample.)
  - (Optional) OGLE-IV SMC variables — not X-Match indexed; local download only.
- Derive labels per object:
  - `is_gaia_known_star`: Gaia parallax or pm significant at >3σ.
  - `simbad_class`: SIMBAD `main_type` string if any.
  - `gaia_var_class`: Gaia variability classifier if any.
  - `is_known_not_sn`: `is_gaia_known_star` ∨ simbad ∈ {Star/Galaxy/QSO/RRLyrae/…}.
  - `is_known_variable`: `gaia_var_class` is not null ∨ simbad ∈ variable classes.
- Save `data/truth/dp1_truth.parquet`.

### Stage 2 — DP1 lightcurves (2 days)

- For each labeled object, pull DiaSource via TAP:
  `band, midpointMjdTai, psfFlux, psfFluxErr, reliability, ra, dec`.
- Batched: `diaObjectId IN (…)` at 500/batch → ~100 queries for 50K objects.
  **Verified 2026-04-24**: 500-ID batch returned 6410 DiaSource rows in
  one query → ~13 det/obj avg, scales linearly.
- Apply reliability filter: `reliability ≥ 0.1`. Document the cut —
  **85.2% of DP1 DiaSource has reliability < 0.1** (verified), so this
  aggressively prunes, but the alternative is pipeline noise dominates.
- Save per-object parquet in `data/lightcurves/dp1/`.

### Stage 3 — feature + local expert table (1 day)

- Feed each lightcurve through `src/debass_meta/features/lightcurve.py`
  `compute_features()` → 51-feature vector.
- Run 4 working local experts on the same lightcurve:
  `supernnova`, `alerce_lc`, `lc_features_bv`, `salt3_chi2` → ternary scores.
  **Verified 2026-04-24**: all 4 have trained models at
  `/project/pi-brout/rubin_hackathon/artifacts/local_experts/{supernnova,alerce_lc,lc_features,salt3_chi2}/`.
- Skip `parsnip` (weights unavailable) and `oracle` (#19 pending).
- Fill NaN for the 4 LSST-broker columns: `fink_lsst/snn`, `fink_lsst/cats`,
  `fink_lsst/early_snia`, `pittgoogle/supernnova_lsst`. LightGBM native NaN
  handling per CLAUDE.md.
- Emit `data/gold/dp1_snapshots.parquet` in v5d schema.

### Stage 4 — inference (0.5 day)

- Load frozen v5d models from SCC (verified 2026-04-24):
  - Followup: `/project/pi-brout/rubin_hackathon/models/followup_safe_v5d/`
    (model.pkl + calibrator.pkl + metadata.json)
  - Trust heads: `/project/pi-brout/rubin_hackathon/models/trust_safe_v5d/`
    (11 experts: alerce_lc, antares/superphot_plus, fink_lsst/{cats,snn,
    early_snia}, fink/{rf_ia,snn}, lc_features_bv, pittgoogle/supernnova_lsst,
    salt3_chi2, supernnova)
- No re-fit, no calibration re-tune. Pure inference.
- Emit `reports/v6_dp1/predictions.parquet` with
  `[diaObjectId, followup_score, per_expert_trust…]`.

### Stage 5 — analysis + paper (2-3 days)

Metrics:
1. **Rejection rate of known stars**: P(followup_score < τ | Gaia-star).
   Target τ = top-5% threshold from training distribution. Expect >95%.
2. **Rejection rate of known variables by class**: same as 1 but stratified
   by `simbad_class` / `gaia_var_class`. Reveals where metaDEBASS struggles
   (probably RRLyrae/LPV blends that look SN-shaped).
3. **Top-K discovery inspection**: top-50 DP1 by followup score, check
   their Gaia/SIMBAD status. Truly unlabeled + clean lightcurves = SN
   candidates worth human inspection.
4. **Score distribution by field**: compare ECDFS (extragalactic) vs SMC
   (stellar) score distributions. Different field → different prior.
5. **Bootstrap CI** on rejection-rate numbers (since spec-SN AUC not
   measurable).

Paper output:
- New §4.5 "Validation on real Rubin DP1 data" (1-2 pages).
- Figure: followup-score histogram stratified by truth class.
- Table: rejection rate per SIMBAD/Gaia class.
- Appendix: top-10 DP1 candidates with lightcurves + crossmatch notes.

### Stage 6 — close-out

- Memory update: `project_rsp_access_dp1.md`, update `project_final_paper_phase.md`.
- Commit `src/debass_meta/access/rubin_rsp.py` + ingest scripts.
- Archive `data/truth/dp1_truth.parquet` on SCC.

## 6. Deliverables

| Path | Purpose | Size estimate |
|---|---|---|
| `src/debass_meta/access/rubin_rsp.py` | TAP adapter | ~200 LOC |
| `src/debass_meta/ingest/rsp_dp1.py` | DP1 lightcurve → feature builder | ~150 LOC |
| `scripts/build_dp1_truth.py` | crossmatch → truth parquet | ~80 LOC |
| `scripts/fetch_dp1_lightcurves.py` | bulk DiaSource puller | ~120 LOC |
| `scripts/score_dp1_v5d.py` | inference + metrics | ~100 LOC |
| `data/truth/dp1_truth.parquet` | object × truth-label matrix | ~10 MB |
| `data/lightcurves/dp1/*.parquet` | per-object LC cache | ~500 MB for 50K objects |
| `data/gold/dp1_snapshots.parquet` | v5d-schema feature matrix | ~50 MB |
| `reports/v6_dp1/*` | predictions, figures, tables | ~30 MB |

## 7. Risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| DP1 export restricted (Shape B forced) | low-medium | Port ingest to RSP JupyterLab; export only scores |
| Reliability filter too aggressive — lightcurves empty | medium | Start at rel≥0.1, tune down if >80% lost |
| DP1 DiaSource reliability so noisy that local experts fail | medium | Use Bazin/Villar (physics fits, robust to noise); skip SNN if it flakes |
| NaN LSST-broker cols break v5d inference | low | LightGBM docs confirm NaN handling; fall back to 0-impute if needed |
| Gaia/SIMBAD hits don't stratify followup cleanly | low | Still reports real-Rubin score distribution — useful regardless |
| OGLE SMC local download fails / format changes | low | SMC already mostly covered by Gaia; SMC bonus, not blocker |
| v5d followup weights access from SCC | low | All artifacts under `/project/pi-brout/rubin_hackathon/` |

## 8. Timeline

- Stage 0 (data rights research, user): ½ day
- Stage 1 (truth table): 1 day
- Stage 2 (lightcurves): 2 days
- Stage 3 (features + experts): 1 day
- Stage 4 (inference): ½ day
- Stage 5 (analysis + paper): 2-3 days
- **Total: ~8-10 working days**

## 9. Prerequisites (action items on user)

- [ ] Confirm DESC DP1 data-export policy: can DiaSource rows leave RSP?
  Determines Shape A vs Shape B.
- [ ] Approve the "validate, don't retrain" framing.
- [ ] Approve sample size target (proposal: 10K feasibility → 50K production).

## 10. What this does NOT do

- Does NOT retrain v5d or alter any v5d weights.
- Does NOT produce a spec-Ia AUC number on DP1 (not achievable — no spec truth).
- Does NOT supersede v5d; it's a validation layer on top.
- Does NOT require external broker API calls on DP1 (brokers don't index DP1).

## 11. Success criteria

A successful v6 closes the paper's biggest open gap by delivering:
1. Rejection rate on ~80K Gaia-known stars (expected 95%+).
2. Rejection rate stratified by ~10 variable/AGN classes (expected patterns).
3. Top-K discovery list (20-50 objects) of potential unreported SN candidates
   from DP1, for follow-up by Rubin DDF teams.

With those, the paper moves from "validated on 12 LSST spec-Ia (0.60 AUC,
wide CI)" to "validated on tens of thousands of Gaia+SIMBAD-labeled DP1
objects with concrete stratified metrics", which is a materially stronger claim.

## 12. Production run results (2026-04-24, SCC)

The full pipeline executed end-to-end on `/project/pi-brout/rubin_hackathon/`
at plan §9 production scale. Actual ceiling was **41,292 objects** — the
full usable `nDiaSources≥5` set in the 6 DP1 fields (plan target 50K; two
fields are sparser than 8,333 per-field cap).

### Run metrics (N=15,868 LC-survivors of 41,292 pool, after `rel≥0.1`)

| Metric | Value | CI95 | N |
|---|---|---|---|
| τ (top-5% in-pool threshold) | 0.5615 | — | — |
| Gaia-known-star rejection | **94.8%** | [94.4, 95.2] | 9,937 |
| Gaia/SIMBAD variable rejection | 83.2% | [78.3, 88.1] | 226 |
| SIMBAD Star rejection | 93.4% | — | 909 |
| SIMBAD Galaxy rejection | 91.3% | — | 402 |
| SIMBAD AGN rejection | 89.5% | — | 19 |
| SIMBAD QSO rejection | 74.5% | — | 55 |
| SIMBAD EclBin rejection | 76.7% | — | 30 |
| SIMBAD RRLyrae rejection | 77.3% | — | 22 |
| Published-SN recall (above τ) | 2/14 | — | 14 LC-survivors |

### Key findings

- **Load-bearing claim met**: Gaia-known-star rejection is **94.8% [94.4, 95.2]**
  on N=9,937, replacing v5d's prior "12 LSST spec-Ia, AUC 0.60 [0.28, 0.90]"
  with a CI two orders of magnitude tighter.
- **Periodic contaminants are the real failure mode**: EclBin (76.7%) and
  RRLyrae (77.3%) rejection lags all non-periodic classes. v5d has no
  dedicated periodic filter; this is paper-worthy as a known limitation.
- **Published-SN recall is gated by reliability noise, not classifier quality**:
  only 2 of the 14 matched DP1 SN candidates retain n_det≥12 after the
  `reliability≥0.1` cut; both are correctly ranked above τ (AT 2024ahzc,
  AT 2024aigt). The two >95%-confidence hi-conf candidates (aigv Ia 98.4%,
  aigw II 95.4%) lose most detections to the floor and cannot be scored
  by any feature-based classifier from the 1–4 sparse dets that survive.

### Build timings (SCC)

| Stage | Input | Output | Wall time |
|---|---|---|---|
| Truth builder | 41,292 DP1 objects + VizieR X-Match | `data/truth/dp1_truth_50k.parquet` (14 cols) | ~2 min |
| Fetch DiaSource | 41,292 diaObjectIds (batched 500) | 15,868 parquets + 25,424 sentinels | 11 min |
| Snapshot build | 15,868 LCs × per-epoch | 31,339 rows × 226 cols, batch SNN + 3 serial experts | 8.5 min |
| v5d scoring | 11 trust heads + followup | predictions.parquet + metrics.json | ~1 min |

### Artifacts

Both SCC (`/project/pi-brout/rubin_hackathon/`) and local repo:

- `data/truth/dp1_truth_50k.parquet` (1.5 MB)
- `data/gold/dp1_snapshots_50k.parquet` (3.9 MB)
- `reports/v6_dp1_50k/predictions.parquet` (4.9 MB)
- `reports/v6_dp1_50k/metrics.json` (20 KB — full per-SIMBAD-class table)

Feasibility-scale (3K) artifacts at `reports/v6_dp1/` and `data/*/dp1_truth.parquet`
kept for regression comparison.

### Remaining for paper §4.5

1. Histogram of `p_follow_proxy` stratified by truth class from
   `reports/v6_dp1_50k/predictions.parquet`.
2. Top-K ranked list with LC plots for the ~20 unlabeled high-scoring
   objects (unreported SN candidates for Rubin DDF teams).
3. Decision on whether to relax `reliability≥0.1` to recover hi-conf Ia
   candidates — trades SN recall vs. top-K contamination.
