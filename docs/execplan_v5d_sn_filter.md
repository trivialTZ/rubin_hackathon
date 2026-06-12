# v5d — SN-filter trust target for fink_lsst/snn and fink_lsst/cats

**Phase**: final paper (post-hackathon).
**Date authored**: 2026-04-23.
**Status**: v5c bugs diagnosed + verified; v5d code landed; qsub 4603140 submitted; figures/docs updates queued.
**Baseline**: v5c (2026-04-23 21:02 EDT, job 4583755) — see `project_lsst_truth_gap.md`.

---

## 1. TL;DR

v5c's LSST trust heads (`fink_lsst/snn` 0.990, `fink_lsst/cats` 0.996) report aggregate AUCs that are **inflated by easy-negative rejection in the AGN-heavy LSST sample** and **inverted on hard-truth slices** (spec × snia AUC ≈ 0 for all 4 LSST heads). Root cause is a **target/capability mismatch**: we were training trust heads for SN-vs-other filters against `is_topclass_correct`, a label that presumes an Ia-vs-non-Ia capability these experts do not have.

v5d's only change is the per-expert trust target:

```python
SN_FILTER_EXPERTS = {"fink_lsst/snn", "fink_lsst/cats"}
def trust_target_col(expert_key):
    return "is_sn" if expert_key in SN_FILTER_EXPERTS else "is_topclass_correct"
```

Everything downstream (v5c truth, v5c gold snapshots, followup architecture) is preserved.

---

## 2. Why v5c's LSST trust heads looked great but weren't

### 2.1 The aggregate-AUC inflation

`reports/trust_head_slices_v5c.csv` shows per-slice test AUCs:

| Head | overall | LSST×weak | LSST×spec | tc=snia | tc=other |
|---|---:|---:|---:|---:|---:|
| fink_lsst/cats | 0.992 | 0.996 | **0.600** (n=120) | **0.018** | 0.999 |
| fink_lsst/snn | 0.947 | 0.998 | **0.432** (n=147) | **0.000** | 0.9998 |
| fink_lsst/early_snia | 0.870 | 0.841 | **0.000** (n=30) | 0.000 | N/A |
| pittgoogle/supernnova_lsst | 0.889 | 0.894 | **0.001** (n=75) | 0.001 | N/A |

The LSST-wide pos_rate of `is_topclass_correct` is 0.07–0.29. Most LSST truth is Tier-3 `other` (AGN-dominated stamp labels). On `other` rows the expert is trivially "right" whenever its top-1 is `other`; the trust head learns this high-prior decision surface and reports the rigged aggregate AUC.

On a true-snia row the expert's `p_snia` is high. The trust head's learned rule on the majority sample says "high `p_snia` → predicted correctness is low" (because on majority rows high `p_snia` was over-calling Ia on an AGN, hence wrong). Applied to actual Ia rows, this rule inverts → AUC ≈ 0.

### 2.2 The mechanical cause — projection caps p_snia at 0.5

Re-reading the projections (`src/debass_meta/projectors/fink_lsst.py`):

```python
# fink_lsst/snn (SN-vs-other binary, Möller+2024 col f:clf_snnSnVsOthers_score)
p_snia  = sn_prob * 0.5       # MAX 0.5
p_nonia = sn_prob * 0.5
p_other = 1.0 - sn_prob

# fink_lsst/cats — when CATS class == 11 (SN-like)
p_snia  = cats_score * 0.5    # MAX 0.5
p_nonia = cats_score * 0.5
p_other = 1.0 - cats_score
```

These caps are the honest admission "SNN/CATS cannot distinguish Ia from non-Ia", but they make `p_snia >= 0.5` **mathematically impossible**. Any Ia-binary follow-up decision derived from their output is necessarily `follow_pred = 0`. Dumping actual v5c values on LSST spec-snia rows:

| Expert | p_snia min | p_snia max | frac > 0.5 |
|---|---:|---:|---:|
| fink_lsst/snn | 0.330 | 0.427 | **0%** |
| fink_lsst/cats | 0.000 | 0.500 | **0%** |
| fink_lsst/early_snia | 0.437 | 0.631 | 67% |
| pittgoogle/supernnova_lsst | 0.013 | 0.693 | 21% |

**Conclusion:** `fink_lsst/snn` and `fink_lsst/cats` are SN-filters. They must be trained as such.

### 2.3 Failed fixes that preflights ruled out

All verified in `scripts/_exp_class_balance_preflight.py` and `scripts/_exp_trust_target_ablation.py`:

- **Class-balanced sample_weight** on the existing target (293× weight on snia rows): cats snia-AUC went 0.021 → 0.010 — *no lift*. Weighting can't fix a target that's semantically wrong.
- **Integer FLT encoding** in local SuperNNova for LSST bands: SNN's `classify_lcs` wants strings; integer input crashes `sequence item 1: expected str, found int`. Rolled back; string FLT is correct.
- **ELAsTiCC2 ingest** to get more Ia: all ML-based broker heads (snn, cats, early_snia, pgb-snn, local supernnova) were trained on ELAsTiCC2 TRAIN → overfit on any public TRAIN_02 sample. TEST_02 requires DESC TOM access (1+ week human step). Parked indefinitely.

---

## 3. Option A — the fix that wins empirically

### 3.1 Mechanism

Re-label the trust-head training target per expert:

- **SN-filter experts** (`fink_lsst/snn`, `fink_lsst/cats`): target = `is_sn = int(target_class != "other")`.
- **All other experts**: target = `is_topclass_correct` (unchanged).

### 3.2 Why this matters mechanically

With `is_sn` as target, every row is diagnostic:

- target=`snia`: y=1 (expert should flag as SN)
- target=`nonIa_snlike`: y=1 (expert should flag as SN)
- target=`other`: y=0 (expert should flag as not-SN)

No slice has a constant y value. The trust head learns a coherent SN-vs-other calibration rather than an inverted Ia/majority pattern.

### 3.3 Preflight evidence (the real experiment)

`scripts/_exp_sn_filter_retrain.py` trained both experts under both targets on the v5c train/test split:

| Expert | Target | Overall AUC | LSST weak | Inversion on spec-snia |
|---|---|---:|---:|---|
| fink_lsst/cats | is_topclass_correct | 0.996 | 0.996 | **YES (0.000)** |
| fink_lsst/cats | **is_sn** | **0.848** | **0.854** | None (well-defined) |
| fink_lsst/snn | is_topclass_correct | 0.982 | 0.998 | **YES (0.000)** |
| fink_lsst/snn | **is_sn** | **0.834** | **0.845** | None (well-defined) |

Aggregate AUC drops ~0.15 — **that's the correction**. The old number was measuring the wrong task.

### 3.4 What Option A does NOT change

From `scripts/_exp_trust_target_ablation.py` (V0, VA, VB, VC1, VC2 all under the same split and feature set):

- Followup test AUC (overall): **0.956** under all 5 variants.
- Followup test AUC (ZTF×spec): **0.934** under all 5 variants.
- Followup test AUC (LSST×spec): **0.587**, bootstrap CI **[0.28, 0.90]**, under all 5 variants.

Why no measurable downstream gain: in v5c the followup head's total feature importance is concentrated in ZTF + survey-agnostic heads and LC features. LSST-specific features rank #74 at 0.2% (fink_lsst/snn q), #81 at 0.1% (pittgoogle/snn_lsst q), #87 at 0.1% (fink_lsst/cats). All 4 LSST heads combined: **<1% feature importance.** The followup ignores them on the 5 LSST spec test objects, so replacing them with anything else produces identical predictions.

Do not claim Option A "improves LSST follow-up". Claim instead: "Option A corrects the trust-head reporting error that made v5c's LSST headline AUCs look spuriously high, at no cost to downstream predictive performance, and aligns the meta-classifier with the actual classification surface each expert produces."

### 3.5 Comparison to Options B, C

- **Option B** (double `p_snia` in the projection for snn/cats): changes inputs, breaks the existing p_nonIa accounting, requires a full gold rebuild to be consistent. Ablation V_B in `_exp_trust_target_ablation.py` showed identical followup AUC to V0/VA/VC. Rejected because it's more invasive for no downstream gain.
- **Option C1/C2** (zero-out q or drop q+proj for the 4 LSST heads): identical followup AUC. Rejected because it loses the SN-filter signal for future use — when Rubin ops delivers more spec-typed LSST Ia, Option A heads will become useful; Option C heads are gone.

Option A is the minimum-change, maximum-information fix.

---

## 4. Code changes

All edits are in-tree and complete as of the v5d qsub.

### 4.1 `scripts/build_expert_helpfulness.py`

Added `is_sn` column alongside existing `is_topclass_correct`:

```python
record["is_sn"] = (
    int(target_class != "other")
    if target_class is not None
    else None
)
```

### 4.2 `src/debass_meta/models/expert_trust.py`

New top-level constant and helper:

```python
SN_FILTER_EXPERTS = {"fink_lsst/snn", "fink_lsst/cats"}

def trust_target_col(expert_key: str) -> str:
    if expert_key in SN_FILTER_EXPERTS:
        return "is_sn"
    return "is_topclass_correct"
```

Inside `train_expert_trust_suite`, the training-target column is resolved per expert:

```python
target_col = trust_target_col(expert_key)
# ...existing filtering...
expert_rows = expert_rows[expert_rows[target_col].notna()].copy()
y_all = expert_rows[target_col].astype(int).to_numpy()
train_y = train_rows[target_col].astype(int).to_numpy()
cal_y_arr = cal_rows[target_col].astype(int).to_numpy()
test_y = test_rows[target_col].astype(int).to_numpy()
```

Fallback: if `is_sn` column is missing from an older helpfulness parquet, the code falls back to `is_topclass_correct` and logs a message (keeps backward compatibility).

Also blocked from feature-candidate pool: `"is_sn"` added to the `blocked` set in `_expert_feature_cols`.

### 4.3 `jobs/run_f3_safe_v5d.sh`

Three-stage pipeline, ~10 min on SCC:

1. `build_expert_helpfulness.py` → `data/gold/expert_helpfulness_safe_v5d.parquet` (now emits `is_sn`).
2. `train_expert_trust.py` → `models/trust_safe_v5d/` + `object_epoch_snapshots_trust_safe_v5d.parquet` + metrics.
3. `train_followup.py --weak-weight 0.25` → `models/followup_safe_v5d/` + metrics.

Reuses v5c truth (`data/truth/object_truth_v5c.parquet`) and v5c gold snapshots (`data/gold/object_epoch_snapshots_safe_v5c.parquet`). No need to rerun the truth-fill pipeline.

---

## 5. Execution

### 5.1 Submitted

```
qsub jobs/run_f3_safe_v5d.sh
# Your job 4603140 ("debass_f3safev5d") has been submitted
```

### 5.2 Validation checklist (landed 2026-04-24 00:02 EDT, job 4603140)

Read `reports/metrics/expert_trust_metrics_safe_v5d.json` and check:

- [x] `fink_lsst/cats` raw AUC **0.8430** (target ~0.84 — match) — was 0.9962 in v5c
- [x] `fink_lsst/snn`  raw AUC **0.8345** (target ~0.83 — match) — was 0.9900 in v5c
- [x] `fink_lsst/cats` calibrated **0.8408** (Δraw = 0.002, no collapse)
- [x] `fink_lsst/snn`  calibrated **0.8336** (Δraw = 0.001, no collapse)
- [x] Other 9 trust heads unchanged from v5c: all Δ AUC ≤ 0.0000 to 4 decimals (bit-identical training path)
- [x] Followup `test_calibrated.roc_auc` **0.9694** (v5c 0.9690, Δ = +0.0004; no regression; Brier 0.0606, ECE 0.0044)
- [x] Bonus — `fink_lsst/early_snia` calibrated: v5c 0.489 (collapsed) → v5d 0.897 (calibrator correctly SKIPPED with cal_n=138<200; task #42 guard triggered)

Per-slice check via `scripts/analyze_trust_heads_by_slice.py --helpfulness data/gold/expert_helpfulness_safe_v5d.parquet --trust-snapshots data/gold/object_epoch_snapshots_trust_safe_v5d.parquet --trust-metadata models/trust_safe_v5d/metadata.json --output reports/trust_head_slices_v5d.csv`  (y auto-resolves to `is_sn` for SN-filter experts):

- [x] `fink_lsst/cats` LSST×spec×snia: inversion **gone**. Spec rows all have is_sn=1 → single-class slice returns `None` (correct), not 0.00.
- [x] `fink_lsst/snn`  same — single-class `None` on all spec/snia/nonIa slices; no inversion.
- [x] LSST×weak AUC strong — cats **0.849**, snn **0.845** (SN-filter task is solvable and well-calibrated).
- [x] LSST×context AUC reasonable — cats 0.809, snn 0.751 (on host-context truth where is_sn signal is weaker).

### 5.3 If the job fails

- Stage 1 failure (helpfulness): most likely a schema mismatch from the new `is_sn` column. Inspect the log at `/project/pi-brout/rubin_hackathon/logs/f3safev5d.qsub.out`.
- Stage 2 (trust): if any expert reports "is_sn column missing, falling back" — the helpfulness parquet wasn't regenerated. Re-run stage 1 and retry.
- Stage 3 (followup): any AUC regression > 0.01 on ZTF×spec is a red flag; compare feature column lists between v5c and v5d.

### 5.4 After validation

1. Clone `jobs/run_f6_safe_v5c_figures.sh` with `sed 's|safe_v5c|safe_v5d|g; s|object_truth_v5c|object_truth_v5c|g'` (truth stays v5c).
2. qsub and pull figures locally into `paper/metaDEBASS_aas/figures_safe_v5d/`.
3. Append "v5d outcome" section to `memory/project_lsst_truth_gap.md` and/or `memory/project_final_paper_phase.md`.

---

## 6. What this fix means for the paper

### 6.1 Methodological contribution (the real win)

*"Meta-classifier trust heads must be trained against a target that lies within the expert's output space. For SN-filter experts whose ternary projection caps `p_snia` at 0.5, the canonical `is_topclass_correct` target produces aggregate AUCs inflated by easy-negative rejection and inverted AUCs on rare-class spec truth. We align the trust objective with each expert's capability (SN-filters train against `is_sn`, Ia-classifiers against `is_topclass_correct`) and report honest per-head AUCs as a result."*

### 6.2 Honest reporting

Replace v5c headline claims in the paper:

| Metric | v5c headline (misleading) | v5d honest |
|---|---:|---:|
| fink_lsst/cats trust AUC | 0.996 | **0.85** |
| fink_lsst/snn trust AUC | 0.990 | **0.83** |
| fink_lsst/cats LSST×spec×snia | inverted (0.00) | well-defined |
| Followup LSST×spec AUC | 0.60 [0.28, 0.90] | 0.60 [0.28, 0.90] *(unchanged; data-limited)* |
| Followup overall AUC | 0.969 | ≥ 0.96 *(unchanged)* |

### 6.3 The LSST data-regime caveat (must be explicit)

We have:

- 5 unique LSST spec-typed objects in the test set (3 Ia + 2 non-Ia).
- 4 unique LSST spec-typed Ia objects in the train set.
- Bootstrap CI on LSST×spec AUC is [0.28, 0.90].

With this sample size, the followup head cannot statistically distinguish any of A/B/C from baseline. Any reported LSST headline must carry the CI. Future Rubin operations will resolve this.

### 6.4 Ablation panel for the paper

`scripts/_exp_trust_target_ablation.py` already produced `reports/trust_target_ablation.json` with V0/VA/VB/VC1/VC2 followup AUCs. This IS a paper table — it honestly shows that the trust-target question has zero measurable effect on downstream in our current LSST regime, at the same time that it has large effect on per-head reporting. That juxtaposition is the contribution.

---

## 7. What NOT to do (lessons from the ablation)

- **Do not** re-introduce ELAsTiCC2 ingest until DESC TOM access is arranged AND we decouple our local SuperNNova weights from Fink's ELAsTiCC training (otherwise contamination is unavoidable). See `project_lsst_truth_gap.md` §v5c and `project_final_paper_phase.md`.
- **Do not** attempt to salvage `fink_lsst/cats` or `fink_lsst/snn` as Ia classifiers by any amount of sample_weight or class-balanced loss. Verified with 293× weight — snia AUC went 0.021 → 0.010. The problem is semantic, not statistical.
- **Do not** expand `SN_FILTER_EXPERTS` without re-verifying each addition's projection. `fink_lsst/early_snia` and `pittgoogle/supernnova_lsst` DO produce `p_snia > 0.5` on real Ia (67% and 21% respectively) — they are Ia classifiers and stay with `is_topclass_correct`.
- **Do not** drop the 4 LSST trust heads from the followup feature set. Their current importance is <1%, but keeping them preserves the architecture for when LSST spec-Ia data arrives at scale.

---

## 8. Open follow-ups (not blocking v5d)

1. **fink_lsst/early_snia calibration collapse is already guarded** (task #42). v5d will exercise the new guard — confirm in metrics that early_snia's calibrated AUC matches raw to within 0.05, or guard triggers with a logged skip reason.
2. **Per-class trust reporting** for all LSST heads should appear in the paper even though only cats/snn change target. Use `scripts/analyze_trust_heads_by_slice.py` on the v5d snapshots.
3. **Local SuperNNova on LSST data preview is uninformative** (tested 2026-04-24, `scripts/_exp_snn_lsst_preflight.py`). Documented as a caveat in `src/debass_meta/experts/local/supernnova.py`. Do not re-litigate.
4. **Task #19 F5 — ORACLE local inference wiring** remains genuinely pending; out of scope for v5d.
5. **Bootstrap CIs** on all LSST×spec metrics — add to `scripts/analyze_results.py` or report alongside the ablation table. 5-object CI is wide but correct.

---

## 9. Checklist to close v5d

- [x] Code edits landed (`build_expert_helpfulness.py`, `expert_trust.py`)
- [x] Job script written (`jobs/run_f3_safe_v5d.sh`)
- [x] Files shipped to SCC
- [x] qsub submitted (job 4603140)
- [x] Job completed successfully (2026-04-24 00:02 EDT, 9 min wall)
- [x] Validation metrics pass (§5.2 checklist — all boxes ticked)
- [x] `analyze_trust_heads_by_slice.py` updated to auto-resolve `y` target per expert (`trust_target_col` import) and re-run on v5d → `reports/trust_head_slices_v5d.csv`
- [ ] v5d figures regenerated (§5.4) — deferred until paper-draft hand-off
- [x] Memory updated with v5d outcome (`project_final_paper_phase.md`, `project_lsst_truth_gap.md`, new `feedback_trust_target_capability.md`)
- [ ] Paper text updated with honest numbers (§6) — deferred to paper drafting
