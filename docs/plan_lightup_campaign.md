# DEBASS Light-up Campaign — Strategic Plan

**Created**: 2026-04-20
**Goal**: Go from 3 → ~15+ live classifiers on BU SCC, covering every broker at least once, plus local physics experts. Strengthen the "heterogeneous broker fusion wins" paper claim.

---

## Current baseline (honest)

| Metric | Value |
|---|---|
| Registered classifiers | **26** |
| Classifiers actually contributing to ensemble | **3** (fink/snn, fink/rf_ia, supernnova) |
| Trust-weighted AUC at n_det=3 | 0.674 |
| Trust-weighted AUC at n_det=10 | 0.823 |
| Best-single AUC at n_det=10 | 0.647 (Δ +0.18) |
| Followup head AUC | 0.942 |
| Truth labels (ZTF-heavy) | 10,684 spec / 15,428 total |
| SCC disk free on /project | **193 GB** (only 2.5 GB used) |

**Structural gaps this plan closes:**
1. Coverage — 23/26 experts silent
2. Per-broker representation — several brokers have zero live experts
3. Orthogonality — no physics-based signal (SALT3, Bazin) yet contributing
4. Robustness — single 80/10/10 split, no cross-val, no per-class breakdown

---

## Pipeline code audit (findings)

| Component | Status | Notes |
|---|---|---|
| `scripts/backfill.py` | ✓ | 5 brokers wired (alerce, fink, lasair, pitt_google, antares). **AMPEL defined but missing from `ALL_ADAPTERS`** in `src/debass_meta/access/__init__.py:7` |
| `scripts/normalize.py` | ✓ | Auto-discovers new broker parquets via glob |
| `scripts/build_object_epoch_snapshots.py` | ✓ | Iterates `ALL_EXPERT_KEYS` from registry |
| `scripts/build_expert_helpfulness.py` | ✓ | Auto-iterates registry |
| `scripts/train_expert_trust.py` | ✓ | Auto-discovers from helpfulness_df |
| `scripts/collect_epoch_history.py` | Partial | Runs `supernnova`, `alerce_lc`. **Missing SALT3, Bazin/Villar, ORACLE-local runners** |
| `src/debass_meta/experts/local/oracle_local.py:150` | ✗ | Inference stubbed — returns "TODO after repo clone" |
| LSST↔ZTF xmatch | Available | `scripts/crossmatch_lsst_to_ztf.py`, not yet run |

---

## Phase map

Eight phases, F0 through F8. F0–F6 are the critical path. F7 is robustness polish. F8 is optional (multi-week LSST-native effort).

### F0 — Register AMPEL + pre-flight *(30 min)*
**Code change**: add `AmpelAdapter` import + entry to `ALL_ADAPTERS` in `src/debass_meta/access/__init__.py`. Commit.
**Verify**: dry-run each broker adapter with 1-object query — confirm HTTP/auth works.
**Output**: all 7 remote brokers reachable from SCC.

### F1 — Backfill 5 idle remote brokers *(6–10 hrs compute)*
**Brokers to run**: `antares`, `ampel`, `pitt_google` (with UPSILoN flag).
**Command**: `scripts/backfill.py --broker antares --parallel 8` etc.
**Output**: `data/bronze/{antares,ampel,pitt_google}_*.parquet` populated.
**Expected delta**: +500 MB – 1.2 GB bronze.
**Risk**: ANTARES returns 0 for pre-2019 ZTF IDs — acceptable, expected behaviour.
**Fallback**: if any broker is unreachable, mark that expert "external-dependency-blocked" in metadata.

### F2 — Silver + gold rebuild *(1 hr)*
**Commands**:
- `scripts/normalize.py` → silver
- `scripts/collect_epoch_history.py` → local rerun outputs
- `scripts/build_object_epoch_snapshots.py` → gold
**Output**: regenerated `data/gold/object_epoch_snapshots_trust.parquet` with all 26 expert projection columns populated.

### F3 — Trust-head training, all live experts *(2–4 hrs)*
**Commands**:
- `scripts/build_expert_helpfulness.py` → `expert_helpfulness.parquet`
- `scripts/train_expert_trust.py --min-rows 300`
- `scripts/train_followup.py`
**Min-rows threshold**: 300 training rows per expert — skip experts with too little data instead of training noisy heads.
**Output**: `models/trust/<expert>/{model.pkl,calibrator.pkl,metadata.json}` for 10–15 experts. Updated `models/followup/`.
**Expected**: n_det=3 AUC moves from 0.67 → **0.72–0.80** range.

### F4 — Local physics experts *(2 days code + 1–2 days compute)*
**Code to write**:
- `scripts/run_salt3_batch.py` — iterate labelled objects, run SALT3 Ia+non-Ia template fits, write per-object Δχ² to `data/silver/local_expert_outputs/salt3_chi2/part-latest.parquet`. Follow `collect_epoch_history.py` schema.
- `scripts/run_lc_features_batch.py` — Bazin+Villar via `light-curve-python`, write features + LightGBM predictions.
**Compute**: SALT3 ~30 s/object → 5918 × 30 = ~50 hrs single-thread; parallelise as SGE array job, 20 workers → 2.5 hrs. Bazin ~2 s → ~3 hrs single-thread.
**Output**: 2 new "expert" streams in silver; re-run F2 + F3 to train trust heads on them.
**Why this matters**: physics-based signal is **orthogonal** to ML classifiers → better ensemble diversity than adding more RNNs.

### F5 — ORACLE local inference *(1 day)*
**Fill stub** at `src/debass_meta/experts/local/oracle_local.py:150`:
- Build SNANA-table adapter (lightcurve JSON → ORACLE's expected Pandas format)
- Call `ORACLE1_ELAsTiCC.classify(lc_df)`
- Map 19-class output → ternary (projector already exists)
- Add to `collect_epoch_history.py`
**Compute**: ~1 s/object on CPU → ~1–2 hrs for 5918 objects.
**Output**: `oracle_lsst` expert goes live (local, independent of ANTARES). Gives us 2 ORACLE signals (remote + local, useful for robustness study).

### F6 — Regenerate analyze + figures *(30 min)*
**Commands**:
- `scripts/analyze_results.py` → updated `reports/metrics/latency_curves.parquet`
- `scripts/make_paper_figures.py` → 4 updated PDFs
- New figure: `fig_ablation.pdf` — top-K ensemble sweep (K=1..15)
- New figure: `fig_per_class_confusion.pdf` — Ia/nonIa-SN/other triangle at n_det=3,5,10,15
**Output**: updated paper figures ready for methods section.

### F7 — Honesty pass *(2–3 days)*
- **F7a**: 5-fold grouped cross-validation in `train_expert_trust.py` + `train_followup.py` (grouped by `object_id` to prevent leakage). Replaces single-split point estimates with mean ± std.
- **F7b**: Per-class breakdown — confusion matrix at each n_det, not just Ia-vs-rest AUC.
- **F7c**: Failure-mode analysis — which objects does the ensemble still miss at n_det=10? What's their host context, redshift, discovery broker?
- **F7d**: Ablation — ensemble with top-K most helpful experts (K=1..N). Shows marginal value curve.

### F8 — ELAsTiCC2 LSST training *(1–2 weeks, DEFERRED)*
- Selectively pull DUMP truth files from NERSC portal (not full 7 GB tarball)
- Wire `build_truth_multisource.py --elasticc2` to parse `gentype` → ternary
- Retrain LSST-exclusive trust heads (`fink_lsst/*`, `alerce/stamp_classifier_rubin_beta`)
- Headline Rubin-era figure for paper
**Decision**: execute only if paper timeline permits. Otherwise, defer and cite as future work.

---

## Per-broker coverage target

After F6, "at least one live classifier per broker" means:

| Broker | Live classifier (target) | Source |
|---|---|---|
| Fink ZTF | fink/snn, fink/rf_ia | Already live |
| Fink LSST | fink_lsst/snn (at least) | Trust head trained in F3 |
| ALeRCE | alerce/lc_classifier_BHRF_forced_phot_transient + top | F3 retrain (data exists in silver) |
| Pitt-Google | pittgoogle/supernnova_ztf | F3 retrain |
| Lasair | lasair/sherlock | F3 (might fail min-rows) |
| ANTARES | antares/oracle, antares/superphot_plus | F1 backfill → F3 train |
| AMPEL | ampel/snguess, ampel/parsnip_followme | F0 fix → F1 backfill → F3 train |
| Local ML | supernnova (live), parsnip | F3 retrain |
| Local physics | salt3_chi2, lc_features_bv | F4 |
| Local deep | oracle_lsst | F5 |

**Expected live count after F0–F6**: 15–18 experts (depends on min-rows cut + ANTARES/AMPEL hit rate).

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| ANTARES index misses old ZTF IDs | Low coverage for antares/* | Already expected — 0 pre-2019 matches is fine |
| AMPEL REST rate limits | Slow backfill | Throttle to 4 parallel; log failures to retry |
| SALT3 sncosmo fit divergence on low-S/N | Missing rows | Catch exception, write NaN — trust head handles via LightGBM NaN support |
| ORACLE lightcurve format mismatch | Expert stays offline | Compare against ORACLE's example notebook; fall back to marking unavailable |
| Disk pressure on /project | Job crash mid-run | 193 GB free, 2.5 GB used → 35× headroom; no risk |
| Calibrator NaN for low-positive-rate experts | Skip calibrator, use raw | `ExpertTrustArtifact.calibrator = None` path already handled |
| New trust heads overfit on small n | Noisy test AUC | F3 min-rows 300 cut; F7a cross-val |

---

## Success criteria

- [ ] ≥ 13 trust heads trained with calibrator (up from 3)
- [ ] At least one live classifier from every broker in the registry
- [ ] `fig_latency_curve.pdf` shows n_det=3 AUC ≥ 0.70 (up from 0.67)
- [ ] `fig_ensemble_gain.pdf` shows Δ-AUC vs best-single widens or stays ≥ 0.15
- [ ] `fig_ablation.pdf` published — shows marginal value per added expert
- [ ] `fig_per_class_confusion.pdf` shows non-Ia-SN and other are separable (not just Ia-vs-rest)
- [ ] 5-fold cross-val reported alongside single-split numbers
- [ ] Memory file updated with final live-expert count + metrics

---

## Decision points

- **After F1**: if ANTARES+AMPEL return < 200 objects each, consider whether Tier-2 (data-but-no-head) experts are worth retraining first as a proof.
- **After F3**: if new ensemble n_det=3 AUC is < 0.70, F4/F5 become necessary (not optional) for paper claim.
- **After F6**: decide whether to do F7 honesty pass or F8 ELAsTiCC2 based on paper deadline.

---

## Artefact paths on SCC (post-campaign)

- Gold: `/project/pi-brout/rubin_hackathon/data/gold/object_epoch_snapshots_trust.parquet`
- Models: `/project/pi-brout/rubin_hackathon/models/trust/<expert>/`
- Metrics: `/project/pi-brout/rubin_hackathon/reports/metrics/latency_curves.parquet`
- Figures: `/project/pi-brout/rubin_hackathon/paper/debass_aas/figures/fig_*.pdf`
- Live expert inventory: `/project/pi-brout/rubin_hackathon/reports/expert_inventory.json` (new — emitted by F6)

---

**Updated**: 2026-04-20 — initial plan. Update as phases complete.
