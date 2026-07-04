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

- ELAsTiCC2 sim-truth LSST benchmark (execplan task #22) remains the path to an honest
  LSST headline.
- Galaxy-EF uptick on DP1: add host-separation features (LSST analogue of distnr) if it
  persists at SCC scale.
- LSST gold rows: SCC has all 3,998 LSST lightcurves as JSONs, but `object_truth.parquet`
  carries no LSST IDs, so the truth-driven fusion builder excludes them. Wiring labels.csv
  LSST rows in as weak-tier truth (train/cal only) is the next data extension.

---

# fusion_v9 — sequence arm (2026-06-11, same session)

Implementation: `src/debass_meta/features/sequence_dataset.py` (causal per-detection
tensors, gold-identical truncation), `src/debass_meta/models/seq_encoder.py` (2-layer GRU,
45,578 params, heteroscedastic next-Δmag SSL, 16-dim frozen embeddings + seq_surprisal),
`scripts/train_seq_encoder.py` (SSL corpus EXCLUDES all cal/test objects — transduction
hygiene), `scripts/build_seq_features.py` (frozen export + join). 12 unit tests including
end-to-end prefix causality. The seq columns enter the SAME pre-registered component-gate
chain as every other block (`seq_features` gate in train_fusion_v8.py).

**Local results:** SSL pretraining converged (val NLL 0.899, 160 s on MPS; corpus = 7k
objects after excluding 2,117 cal/test). Export covered 100% of the 137,281 gold rows with
zero truncation mismatches. **Gate verdict: seq_features DROPPED on cal**
(Δ macro-AUC = −0.004 on top of EXT features; rule requires CI-positive improvement).
The deep arm did not pay rent at the 562-spec-label local scale — consistent with the design
panel's prediction that the EXT sequence statistics already capture most raw-photometry
signal. The local v9 model therefore equals v8 (ledger: `reports/metrics/fusion_v9_train.json`).

**SCC v8 result (completed 2026-06-12, job 6060088):** on the powered test —
**765 spectroscopic test objects** with the real 11-trust-head v6e2 comparator —
fusion_v8 macro OvR AUC @ n_det=5 = **0.922 [0.903, 0.939]**, and the pre-registered
headline delta vs re-scored v6e2 = **+0.132 [+0.105, +0.157] snia OvR AUC — decisively
significant**. Guards: DP1 EclBin+RRLyrae EF non-regression PASS, seed-spread PASS,
LSST-spec N/A (gold still ZTF-only pending the LSST weak-label rebuild in v9c).

---

# fusion_v9c — the sequence classifier as a registered expert (2026-06-12)

v9c promotes the sequence model from frozen features to a STANDALONE TERNARY CLASSIFIER
integrated through the standard expert contract ("especially for what we did"):

- `src/debass_meta/models/seq_classifier.py` — SSL encoder + per-step classification head
  (one forward = P(class | first n dets) for every prefix; causal, leak-free, ~57k params),
  temperature-calibrated on cal.
- `scripts/train_seq_classifier.py` — fine-tune on TRAIN objects only (cal = early stop +
  temperature; test never read); weak LSST labels from `lsst_candidates.csv` (ALeRCE Rubin
  stamp: SN → nonIa_snlike NEVER snia; AGN/VS/asteroid/bogus → other;
  label_source='alerce_self_label' so provenance masking applies).
- Registered as expert `seq_v9` (EXPERT_REGISTRY + projector + ALL_LOCAL_EXPERTS): scored
  per epoch into silver by `local_infer.py`, picked up by the gold builder, trust-headed by
  pooled Stage A, gated into Stage B. 19 unit tests (causality, loss masking, temperature,
  artifact round-trip, registry wiring, LSST label mapping).
- Builder extension: `--lsst-candidates` harvest (3,998 LSST objects with cached lightcurves
  on SCC enter train/cal as weak truth — inert locally, active on SCC).

**Local results (562-spec-label scale):**
- Standalone skill (cal, object-level OvR AUC, weak labels included): snia 0.74/0.79/0.87
  and other 0.78/0.86/0.91 at n_det = 3/5/10 — competitive with the dedicated local experts,
  using nothing but photometry. Fine-tunes in 82 s on MPS.
- Trust head: pooled Stage A assigned seq_v9 a calibrated test AUC of **0.868 on 5,529 test
  rows — the widest coverage of any expert** (needs only a lightcurve, no broker).
- Fusion gate: **seq_v9_expert DROPPED on cal** (Δ macro-AUC −0.008). Diagnosis: the first
  self-trained expert exposes a stacking trap — its train-row projections are partially
  in-sample (it saw those labels), so Stage B overweights them; cal (out-of-sample for the
  classifier) catches it. The gate machinery was extended so self-trained experts are gated
  components by default. Proper v10 fix if wanted: K-fold OOF inference for seq_v9 on train
  rows.

**SCC chain (rerunning, jobs 6061741 → 6061742 → 6061743):** GPU SSL pretrain + classifier
fine-tune (queue unpinned with `-l gpu_c=8.0` — the first attempt landed on a Tesla P100
sm_60 that modern torch wheels cannot run; `resolve_device` now also falls back to CPU on
unsupported GPUs) → frozen-embedding arm re-test → full v9c integration (seq_v9 silver →
gold rebuild WITH LSST weak labels → Stage A retrain incl. seq_v9 trust head → gates →
score → eval into `reports/fusion_v9c`). At SCC scale the classifier trains on ~4k spec +
weak ZTF + 3,998 weak LSST objects with mixed-band sequences — the honest LSST-capability
test, and the rebuilt gold finally carries LSST rows (LSST-spec guard may become
computable).

Monitor: `ssh scc 'qstat -u tztang'`; logs `logs/fusion_v9_pretrain.qsub.out`,
`logs/fusion_v9_arm.qsub.out`, `logs/fusion_v9c.qsub.out`; resubmit:
`bash jobs/submit_fusion_v9_chain.sh [V8_JOB_ID]`.

## v9c FINAL — SCC results (2026-06-12, job 6064722, 54 min wall)

Two SCC incidents en route, both fixed and regression-tested: (1) the dedicated-head
fallback gate crashed on experts whose rows are all train/cal (the LSST cohort has zero
locked-test rows) — empty-mask guard added to `pooled_trust.py`; (2) the rerun was
accelerated by pinning Stage A's deterministic grid winner via
`DEBASS_POOLED_TRUST_PARAMS` (result-identical; the crashed run had already computed it)
plus idempotent artifact-reuse in the job script.

**Headline (765 spec test objects, locked):** macro OvR AUC@5 = **0.921 [0.901, 0.939]**,
snia 0.923, **Δ vs re-scored v6e2 = +0.129 [+0.102, +0.156]** — the v8-scale win is fully
preserved in the integrated stack. Guards: DP1 EclBin EF PASS, seed-spread PASS,
LSST-spec still N/A (the LSST cohort is weak-labeled; spec LSST truth remains the
ELAsTiCC2 follow-up).

**seq_v9 as an expert at scale:** trust head calibrated test AUC **0.824 on 27,323 test
rows — the widest coverage of any of the 13 trust-headed experts** (it needs only a
lightcurve). The LSST broker experts (fink_lsst/*, pittgoogle) trained trust heads from
the new LSST rows (cal-only cohort, so no test-row entry) and emit q on LSST objects at
scoring time.

**Gates at 2,555-cal-object power:** ext_features kept (+0.0087 CI-positive);
seq_v9_expert dropped (−0.0009 — neutral, the local stacking-trap concern resolved as
no-harm/no-help); traj_features dropped (+0.0005 n.s. even with fink_lsst trajectories on
LSST rows); pooled-q and expert-dropout dropped; Dirichlet kept. The deployed Stage B is
base + EXT; seq_v9 contributes through its calibrated trust-headed projections in the
per-expert payload and ensemble, not as Stage-B feature columns.

Artifacts: `models/{trust,followup,conformal}_fusion_v9c`, `reports/fusion_v9c/`,
`data/scores/predictions_fusion_v9c*.parquet` + per-goal priority lists (all on SCC;
mirrored locally under `models_scc_backup/fusion_v9c_20260612/`, `reports_from_scc/`,
`data/scores/scc_fusion_v9c/`).

**DP1 enrichment @ top-1%, SCC models (v6e2 → v8 → v9c):** EclBin+RRLyrae 17.88 → 5.77 →
9.62; Gaia variables 5.46 → 3.54 → 3.98; Published SNe 0.00 → **0.00 → 7.14**. The
notable finding: the ZTF-only v8 still found zero published LSST SNe in its top-1% — it
was the v9c LSST weak-label training rows that switched published-SN recall on, at a
modest periodic-contamination cost (still ~2× better than baseline). The often-quoted
"EclBin 3.9 / published-SN 14.3" figures are the LOCAL v8 model's DP1 scores — correct in
their own context, not the SCC numbers. Presentation materials
(`presentations/hackathon_20260612/`) use the SCC v9c numbers via `facts.json`.
