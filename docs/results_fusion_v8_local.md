# fusion_v8 — Local Results & SCC Handoff (2026-06-11)

The advanced-ML upgrade of the metaDEBASS meta-layer: per-classifier confidence (trust) +
science-goal-conditioned follow-up priority, for SN Ia AND non-Ia AND other users.
Design: judge-panel-selected GBM++ (`tmp/fusion_designs/design_boosted.md`,
`tmp/fusion_v8_implementation_spec.md`). Everything is additive (`_fusion_v8` suffix);
the locked v6e2 artifacts are untouched.

## What changed vs v6e2

1. **Training-set expansion 1,920 → 8,668 objects** (29.5k → 137.3k epoch rows): every cached
   lightcurve with truth (BTS/TNS/object_truth) + catalog-'other' harvest. Locked test set
   preserved **byte-identical** (383 objects); all new objects went to train/cal only.
2. **Label honesty fix**: 3,149 BTS-unclassified ('-') objects carried fabricated
   `nonIa_snlike/spectroscopic` labels in `object_truth.parquet` (pre-existing bug, 15 of them
   inside the locked test set). Demoted to `label_quality='bts_unclassified'`, class cleared —
   excluded from all training losses and eval slices for BOTH models.
3. **50 EXT lightcurve features** (`features/lightcurve_ext.py`): cadence, magerr/SNR,
   detection significance, variability/periodicity (von Neumann η, flip rate, Stetson,
   Lomb–Scargle at n_det≥8), shape/rise, reference-source context (distnr).
4. **66 broker score-trajectory features** (`features/trajectory.py`) from 4.17M silver events
   (as-of alert_jd, leakage-tested; gold parity 1e-16).
5. **Pooled trust model** (`models/pooled_trust.py`): ONE LightGBM over all experts
   (expert-ID categorical) replaces isolated per-expert heads; q__ = NaN when absent (no more
   "missing = untrustworthy"); hierarchical calibration isotonic→Platt→global; per-expert
   fallback gate vs dedicated heads; `q_prior__` = expected helpfulness if a broker were
   fetched (new capability: broker-query prioritization).
6. **Multiclass follow-up head** (`models/multiclass_followup.py`): 3-class (snia / nonIa /
   other) replaces the binary Ia proxy; quality-weighted, object-weight-normalized loss;
   provenance masking (structural anti-circularity); Dirichlet calibration ladder.
7. **Conformal sets + selection** (`models/conformal.py`, `models/selection.py`): Mondrian APS
   (class × survey × n_det bucket), utility-vector ranking s = u·p̂, budget-aware top-K with
   abstention demotion, FDR-controlled threshold variant.
8. **Honest protocol baked in** (`scripts/eval_fusion_v8.py`): object-level, locked-test-only,
   spec-only headline, 1,000-resample paired object bootstrap, best-single chosen on cal,
   pre-registered headline + guards, component gates decided on cal before test is scored.

## Pre-registered headline (locked test, spec-only, object-level, n_det=5)

| Quantity | Value |
|---|---|
| fusion_v8 macro OvR AUC | **0.975** [0.950, 0.994] (n=103 objects) |
| fusion_v8 snia OvR AUC | 0.968 [0.932, 0.994] |
| fusion_v8 nonIa OvR AUC | **0.982** (new capability — v6e2 has no non-Ia output) |
| Δ snia AUC vs re-scored v6e2 (same frame, paired) | **+0.064 [+0.017, +0.126]** — significant |
| Δ snia AUC vs v6e2 on its own-era frame (audit's toughest variant) | **+0.039 [+0.005, +0.083]** — still significant |
| At n_det=3 | fusion 0.934 vs v6e2 0.816 (snia OvR) |

Guards: DP1 EclBin+RRLyrae EF non-regression **PASS** (3.9 vs 13.6); seed spread **PASS**
(0.003 < 0.02); LSST-spec slice N/A locally (no LSST spec test objects — SCC gate).

## DP1 50k operational ranking (EF @ top-1%, p_follow_proxy, N=15,868)

| Class | fusion_v8 | v6e2 |
|---|---|---|
| Published SNe (want HIGH) | **14.3** [0, 35.3] | **0.00** (found none) |
| EclBin+RRLyrae (want LOW) | **3.9** | 17.9 |
| Gaia variables (want LOW) | 2.2 | 5.5 |
| Gaia stars (want LOW) | 0.87 | 0.85 |
| SIMBAD Galaxy (want LOW) | 2.2 [0.95, 3.7] | 0.00 |

The periodicity/variability features fixed the documented periodic-variable failure while the
ranker went from finding zero of the 14 published SNe to enriching them 14×. The small galaxy
EF uptick (0→2.2) is the one regression to watch on SCC.

## Component-gate ledger (decided on cal, then frozen)

| Component | Decision | Cal Δ macro-AUC@5 |
|---|---|---|
| EXT features | **kept** | +0.036 (CI-positive) |
| Trajectory features | dropped (locally) | +0.003 — only ZTF brokers locally; re-test on SCC where fink_lsst's 1.7M-event trajectories live |
| q__ as Stage-B features | dropped | −0.061 — trust q remains a first-class deliverable, just not a follow-up feature |
| Expert-dropout augmentation | dropped | n.s. with ~12-20% local expert coverage; re-test on SCC |

Trust heads (pooled, calibrated test AUC): fink/snn 0.961, fink/rf_ia 0.880, supernnova 0.979,
alerce/stamp 0.949 — at parity or better vs dedicated v6e2-style heads (no fallback fired).

## Honest caveats (from the independent audit — CONFIRMED-WITH-CAVEATS)

- The spec-only local test slice is small (103 objects; 91 Ia / 12 non-Ia / **0 'other'**) and
  100% ZTF. Local numbers gate correctness and direction; **SCC (~800 spec test objects,
  LSST slices) gates the publishable magnitudes.**
- ~40% of the +0.064 same-frame delta reflects v6e2 degrading on the richer-coverage frame;
  the conservative paired estimate is +0.039 (still CI-positive).
- "v6e2" locally = the 3-trust-head local stack, not the 12-head SCC version.
- Local ranking metrics (purity@K) saturate at the 88% Ia base rate — uninformative locally.

## Reproduce locally (~50 min total)

```bash
source ~/.venvs/debass_py313/bin/activate
python scripts/build_snapshots_fusion.py --n-jobs 8 --dp1        # ~10 min
python scripts/build_helpfulness_fusion.py \
  --snapshots data/gold/object_epoch_snapshots_fusion_v8.parquet \
  --output data/gold/expert_helpfulness_fusion_v8.parquet --parity-check
python scripts/train_fusion_v8.py --n-jobs 8                     # ~30 min (Stage A grid dominates)
python scripts/score_fusion_v8.py --snapshots data/gold/object_epoch_snapshots_fusion_v8_trust.parquet
python scripts/score_fusion_v8.py --dp1
python scripts/eval_fusion_v8.py --snapshots data/gold/object_epoch_snapshots_fusion_v8_trust.parquet \
  --split data/gold/split_fusion_v8.json
# Tables → reports/fusion_v8/; tests: python -m pytest tests/test_lightcurve_ext.py \
#   tests/test_trajectory_asof.py tests/test_pooled_trust.py tests/test_followup_fusion.py \
#   tests/test_snapshots_fusion_g5.py   (75 tests)
```

## SCC scale-up (the real performance gate)

```bash
qsub -N debass_fusion_v8 -cwd -V -P pi-brout -l h_rt=08:00:00 -l mem_per_core=8G -pe omp 16 \
     -o logs/fusion_v8.qsub.out -e logs/fusion_v8.qsub.err jobs/run_fusion_v8_pipeline.sh
```

What SCC adds that local cannot: 12,772 objects / 12+ experts incl. fink_lsst + pittgoogle
LSST trajectories (the trajectory-feature and expert-dropout gates should be re-decided there),
~800 spec test objects (powered headline), LSST-spec slice guard, context-tier catalog-'other'
labels, and the v6e2 12-head comparator. Expect the SCC silver/truth to need the same
BTS-unclassified demotion — it is in the builder, applied automatically.

## Follow-ups (deferred by design)

- v9 deep add-on (gated): frozen GRU sequence embeddings + seq_surprisal as Stage-B features,
  SSL corpus excluding cal/test (recipe in `tmp/fusion_designs/design_sequence.md`).
- ELAsTiCC2 sim-truth LSST benchmark (execplan task #22) remains the path to an honest
  LSST headline.
- Galaxy-EF uptick on DP1: add host-separation features (LSST analogue of distnr) if it
  persists at SCC scale.
