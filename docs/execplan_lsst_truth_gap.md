# DEBASS — LSST Truth Gap: execution plan (verified)

_Authored 2026-04-23 after F3-SAFE-v5 job 4572623 finished with 7 trust heads
(same as v4). All CLI flags, column names, and snapshot paths below were
verified against the actual code and data on 2026-04-23._

---

## 1. Verified facts

### 1.1 What exists on SCC (don't touch — baseline for comparison)

| Artifact | Path | Size / rows |
|---|---|---|
| Gold snapshot v5 | `data/gold/object_epoch_snapshots_safe_v5.parquet` | 211,392 rows (72,331 LSST + rest ZTF) |
| Helpfulness v5 | `data/gold/expert_helpfulness_safe_v5.parquet` | 985,541 rows |
| Trust-scored snapshot v5 | `data/gold/object_epoch_snapshots_trust_safe_v5.parquet` | same rows as gold v5 |
| Trust models v5 | `models/trust_safe_v5/{alerce_lc, antares__superphot_plus, fink__rf_ia, fink__snn, lc_features_bv, salt3_chi2, supernnova}/` | 7 heads |
| Trust metrics v5 | `reports/metrics/expert_trust_metrics_safe_v5.json` | — |
| Followup v5 | `models/followup_safe_v5/` + `reports/metrics/followup_metrics_safe_v5.json` | cal AUC **0.9249** |

### 1.2 Snapshot has truth columns INLINE (verified 2026-04-23)

`object_epoch_snapshots_safe_v5.parquet` contains these four columns directly
(populated from `object_truth.parquet` at snapshot-build time via
`src/debass_meta/ingest/gold.py` lines 163–166):

| Snapshot column | Source in `object_truth.parquet` |
|---|---|
| `target_class` | `final_class_ternary` |
| `target_follow_proxy` | `follow_proxy`  (`int(final_class_ternary == "snia")`) |
| `label_source` | `label_source` |
| `label_quality` | `label_quality` |

For LSST rows (72,331 of them) **all four are `None`**. That is the entire
bug: no truth in → no `is_topclass_correct` out → trust trainer skips the
LSST-only experts.

### 1.3 Helpfulness builder reads `target_class` from snapshot row, not truth file

`scripts/build_expert_helpfulness.py:47` does `"target_class": row.get("target_class")`.
So **patching the snapshot's four truth columns is sufficient** — we do not
need to rebuild the snapshot to propagate new LSST truth.

### 1.4 Truth-source usage in snapshot

`scripts/build_object_epoch_snapshots.py --truth data/truth/object_truth.parquet`
is how the v5 snapshot was built. `data/truth/object_truth.parquet` today has
**15,428 rows, all ZTF** (verified). We can either:
- **Path A** — patch the snapshot parquet directly (fast), OR
- **Path B** — append LSST rows to truth + rerun snapshot builder (slow, ~2h).

The plan uses Path A. Path B is the fallback if the patch path corrupts the
parquet (unlikely — it's pandas I/O on known columns).

### 1.5 Existing `label_source` patterns (for naming consistency)

Current `object_truth.parquet` uses source strings like:
- `ztf_bts`, `tns_spectroscopic`, `tns_unconfirmed`, `labels_csv`, `alerce_self_label`
- `fink_xm_tns (<TNS_NAME>)`, `fink_xm_tns_ambiguous (<name>)`, `fink_xm_host_context`

Our new LSST sources will follow the same pattern:
- `fink_lsst_xm_tns_spec`, `fink_lsst_xm_host_context`
- `alerce_stamp_lsst` (safe: `alerce/stamp_classifier_rubin_beta` has **0 events in silver**, so it is never a trust-trained expert; using its class as a truth source is not circular for any current or imminent head)
- `broker_consensus_lsst_loo` (per-expert leave-one-out; never same-expert)

### 1.6 `label_quality` vocabulary (existing + new)

Used by the trust trainer implicitly (`exclude_circular_consensus` only
filters `broker_consensus` literal). Current values: `spectroscopic`,
`tns_untyped`, `context`, `weak`. We'll reuse: `spectroscopic`, `weak`, `context`.

### 1.7 ALeRCE Rubin stamp classifier provenance check

Queried `GET https://api-lsst.alerce.online/probability_api/probability?oid=<id>`
on 47 sample LSST objects 2026-04-23:

- 46 / 47 returned a ranked list
- Top-class distribution: **AGN 36 · SN 7 · asteroid 3 · VS 1**
- Confidence: median **0.94**, p10=0.65, **83 %** of objects at ≥ 0.85
- Classifier name: `stamp_classifier_rubin_beta_20260421` (latest) or `stamp_classifier_rubin_beta`

The sample is AGN-dominated — this is a real signal (same conclusion as
broker consensus: 0 objects have 2+ experts agreeing Ia; 250 have 2+
agreeing non-Ia).

### 1.8 Broker consensus measurement (from gold snapshot latest epoch)

Per-object latest `proj__*__p_snia` availability among 3,998 LSST objects:

| Expert | Non-null | % |
|---|---:|---:|
| `fink_lsst/snn` | 3,424 | 85.6 % |
| `fink_lsst/cats` | 3,173 | 79.4 % |
| `pittgoogle/supernnova_lsst` | 2,713 | 67.9 % |
| `fink_lsst/early_snia` | 106 | 2.7 % |

86 % of objects have ≥ 2 of the first three experts scoring them. Only 250
show consensus non-Ia at p < 0.3; **0 show consensus Ia at p > 0.7**.

### 1.9 Fink xm fields coverage (from silver `fink_lsst/crossmatch`, 3,997 objects)

| Field | Non-null | % |
|---|---:|---:|
| `fink_xm_tns_type` (spec) | **8** | **0.2 %** (useless alone) |
| `fink_xm_tns_fullname` | 100 | 2.5 % |
| `fink_xm_simbad_otype` | 935 | 23.4 % |
| `fink_xm_legacydr8_zphot` | 2,433 | 60.9 % |
| `fink_xm_gaiadr3_Plx` | 935 | 23.4 % (stellar indicator) |

### 1.10 DP0.2 / DP1 simulation truth

Bit-decode of `170019695454847192` as DP0.2 format yields `tract=151`,
far below the DP0.2 valid-tract range (~8,000–10,000). These IDs are **not
DP0.2**; may be DP1 but Rubin Science Platform auth is required to query
TAP catalogs. **Not a near-term unlock; plan stays stamp-driven.**

---

## 2. Truth tier rule (applied once per LSST object)

Evaluated top-to-bottom. First match wins.

```
TIER 1 — SPECTROSCOPIC (~8 objects, 0.2 %)
  if fink_xm_tns_type matches /^SN\s*Ia/i          → final_class_ternary=snia
  elif fink_xm_tns_type matches /^SN\s*(II|Ib|Ic|IIn|Ibc|SLSN)/i
                                                    → final_class_ternary=nonIa_snlike
  elif fink_xm_tns_type matches /^(AGN|QSO|TDE|CV)/i
                                                    → final_class_ternary=other
  label_source="fink_lsst_xm_tns_spec", label_quality="spectroscopic"

TIER 2 — STAMP + BROKER AGREEMENT  (~3,000 expected)
  if stamp_prob >= 0.85 AND stamp_class is given:
    if stamp_class == "SN" AND max(p_snia over LSST SN experts at latest epoch) >= 0.5:
        final_class_ternary="snia"
    elif stamp_class == "SN" AND consensus-non-Ia (2+ experts with p_snia < 0.3):
        final_class_ternary="nonIa_snlike"
    elif stamp_class == "SN":          # stamp says SN, brokers undecided
        final_class_ternary="nonIa_snlike"    # our sample leans non-Ia; safer default
    elif stamp_class in {"AGN","VS","asteroid","star"}:
        final_class_ternary="other"
  label_source="alerce_stamp_lsst+consensus", label_quality="weak"

TIER 3 — STAMP ONLY  (~700 expected)
  if 0.70 <= stamp_prob < 0.85:
    map stamp_class → ternary as above (without broker cross-check)
  label_source="alerce_stamp_lsst", label_quality="weak"

TIER 4 — CONTEXT  (~200 expected)
  if gaiadr3_Plx > 1.0:                           # mas → Galactic star
    final_class_ternary="other"                   label_source="gaia_parallax_exclusion"
  elif simbad_otype in {"QSO","Sy1","Sy2","AGN"}:
    final_class_ternary="other"                   label_source="fink_lsst_xm_simbad_agn"
  elif simbad_otype in {"G","rG","GiG","GiC","BiC"} AND legacydr8_zphot is not None:
    final_class_ternary="nonIa_snlike"            # extragalactic transient, subtype unknown
    label_source="fink_lsst_xm_host_context"
  elif simbad_otype in {"*","RR*","EB*","CV"}:
    final_class_ternary="other"                   label_source="fink_lsst_xm_stellar"
  label_quality="context"

DROP
  else → skip (leaves target_class=None; expert trust skips the row as before)
```

In all tiers:
```
follow_proxy = int(final_class_ternary == "snia")   # matches build_truth_multisource.py L83
```

Expected yield: ≥ 97 % of 3,998 objects get a non-None `final_class_ternary`.

---

## 3. Implementation — 5 concrete steps

All new artifacts use the `_v5b` suffix; nothing in v5 is overwritten.

### STEP 1 — Backfill ALeRCE Rubin stamp for 3,998 LSST objects

**New file**: `scripts/backfill_alerce_lsst_stamp.py` (~80 lines)

```
GET https://api-lsst.alerce.online/probability_api/probability?oid=<id>
8-way ThreadPoolExecutor (no auth; no rate limit observed).
Write out: data/truth_candidates/alerce_stamp_lsst.parquet
  columns: object_id, stamp_class, stamp_prob, stamp_classifier_version, query_time
Retry on 502 with exponential backoff (2s/5s/12s) — same pattern as
    scripts/fetch_lightcurves.py:_request_with_retry
```

Wall time ~10 min (based on 9.2 s for 47 oids at 8× parallel).

### STEP 2 — Build LSST truth candidate DataFrame

**New file**: `scripts/build_lsst_truth.py` (~150 lines)

**Inputs** (all already on SCC):
- `data/silver/broker_events.parquet` (for `fink_lsst/crossmatch` raw JSON)
- `data/gold/object_epoch_snapshots_safe_v5.parquet` (for latest-epoch consensus)
- `data/truth_candidates/alerce_stamp_lsst.parquet` (from Step 1)
- `data/labels.csv` (LSST id list)

**Output**: `data/truth_candidates/object_truth_lsst.parquet`

**Columns** — must match existing `object_truth.parquet`:
```
object_id, final_class_ternary, follow_proxy, label_source, label_quality,
bts_type (None), tns_name, redshift, final_class_raw, truth_timestamp,
tns_prefix, tns_type, tns_has_spectra, tns_redshift, tns_ra, tns_dec,
tns_discovery_date, consensus_experts, consensus_n_agree, consensus_n_total
```

For LSST rows, the TNS/BTS columns stay None unless Tier 1 fires; consensus
columns populated from the 3 LSST experts' latest p_snia. Apply §2 rule.

### STEP 3 — Merge LSST truth into a new truth table

**New file**: `scripts/merge_lsst_truth.py` (~30 lines)

```
ztf_truth = pd.read_parquet("data/truth/object_truth.parquet")   # 15,428 rows
lsst_truth = pd.read_parquet("data/truth_candidates/object_truth_lsst.parquet")
out = pd.concat([ztf_truth, lsst_truth], ignore_index=True, sort=False)
out.to_parquet("data/truth/object_truth_v5b.parquet")            # new file, not overwrite
```

Assert no `object_id` duplicates.

### STEP 4 — Surgical snapshot patch

**New file**: `scripts/patch_snapshot_truth_lsst.py` (~50 lines)

```
snap = pd.read_parquet("data/gold/object_epoch_snapshots_safe_v5.parquet")
lsst_truth = pd.read_parquet("data/truth_candidates/object_truth_lsst.parquet")

# Build lookup
lookup = lsst_truth.set_index("object_id")[
    ["final_class_ternary", "follow_proxy", "label_source", "label_quality"]
]

# Fill only where target_class is None (LSST rows)
mask = snap["target_class"].isna() & snap["object_id"].isin(lookup.index)
for src_col, dst_col in [("final_class_ternary","target_class"),
                         ("follow_proxy","target_follow_proxy"),
                         ("label_source","label_source"),
                         ("label_quality","label_quality")]:
    snap.loc[mask, dst_col] = snap.loc[mask, "object_id"].map(lookup[src_col])

snap.to_parquet("data/gold/object_epoch_snapshots_safe_v5b.parquet")
```

Verify: `snap.loc[snap["id_kind"]=="lsst_dia_object_id", "target_class"].notna().mean()` should be ≥ 0.97.

### STEP 5 — Rebuild helpfulness + retrain trust + retrain followup

**New file**: `jobs/run_f3_safe_v5b_truth_fill.sh` (SGE-wrapped, clones v5 exactly)

```bash
#!/bin/bash
#$ -N debass_f3safev5b
#$ -l h_rt=01:00:00
#$ -l mem_per_core=16G
#$ -pe omp 4
#$ -cwd
#$ -j y
#$ -o /project/pi-brout/rubin_hackathon/logs/f3safev5b.qsub.out

set -euo pipefail
cd /project/pi-brout/rubin_hackathon
source .venv/bin/activate
source .env 2>/dev/null || true
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

echo "$(ts) F3-SAFE-v5b — LSST truth filled from ALeRCE stamp + Fink xm + consensus"

python3 -u scripts/build_expert_helpfulness.py \
    --snapshots data/gold/object_epoch_snapshots_safe_v5b.parquet \
    --output    data/gold/expert_helpfulness_safe_v5b.parquet

python3 -u scripts/train_expert_trust.py \
    --snapshots       data/gold/object_epoch_snapshots_safe_v5b.parquet \
    --helpfulness     data/gold/expert_helpfulness_safe_v5b.parquet \
    --models-dir      models/trust_safe_v5b \
    --output-snapshots data/gold/object_epoch_snapshots_trust_safe_v5b.parquet \
    --metrics-out     reports/metrics/expert_trust_metrics_safe_v5b.json \
    --n-jobs 4

python3 -u scripts/train_followup.py \
    --snapshots        data/gold/object_epoch_snapshots_trust_safe_v5b.parquet \
    --trust-models-dir models/trust_safe_v5b \
    --model-dir        models/followup_safe_v5b \
    --metrics-out      reports/metrics/followup_metrics_safe_v5b.json

echo "$(ts) F3-SAFE-v5b DONE"
echo "  trust heads (safe-v5b): $(ls -1 models/trust_safe_v5b/ | grep -v 'metadata.json$' | wc -l)"
ls -1 models/trust_safe_v5b/ | grep -v 'metadata.json$' | sort
python3 -c 'import json; d=json.load(open("reports/metrics/followup_metrics_safe_v5b.json")); print("followup cal AUC:", d["test_calibrated"]["roc_auc"])'
```

Wall time budget: ~15 min (helpfulness ~5 min · trust ~5 min · followup ~5 min).

### STEP 6 (optional, post-v5b) — Propagate fix upstream

If v5b looks healthy, add stamp-based LSST truth to
`scripts/build_truth_multisource.py` as a first-class source so future
retrains include it automatically. Not required for v5b itself.

---

## 4. Expected v5b outcome

| Metric | v4 | v5 (today) | v5b (target) |
|---|---:|---:|---:|
| Trust heads | 7 | 7 | **11** (7 legacy + 4 LSST exact_alert) |
| LSST rows in gold | 0 | 72,331 | 72,331 (same, truth filled) |
| LSST rows with non-None `target_class` | 0 | 0 | **≥ 3,850 / 3,998** (≥ 97 %) |
| Follow-up cal AUC (test) | 0.9250 | 0.9249 | ≥ 0.92 floor; higher if LSST heads are informative |

New trust heads expected (in sorted order matching the v5 listing convention):
- `fink_lsst__cats`
- `fink_lsst__early_snia` *(may skip if < ~300 usable rows — currently 786 helpfulness rows, will depend on the tier-2/3 survival rate)*
- `fink_lsst__snn`
- `pittgoogle__supernnova_lsst`

`lasair/sherlock` stays excluded — `prediction_type=context_only` by design
(correct behavior; it's a static host classifier, not a SN-Ia predictor).

---

## 5. Risks & contingencies

| Risk | How it manifests | Mitigation |
|---|---|---|
| Weak labels noisy → LSST heads underfit | Trust AUC < 0.6 per head | Tag `label_quality` strictly; we can add `--min-label-quality weak` if noisy tier-3/4 rows hurt |
| Snapshot patch misses a column | LSST rows still None after Step 4 | Fallback Path B: run Step 3 → rerun `build_object_epoch_snapshots.py --truth object_truth_v5b.parquet` (2 h) |
| ALeRCE API down during Step 1 | Script fails | Retry wrapper with 3× exponential backoff (already in `fetch_lightcurves.py`); if still down, fall back to tier-1 + tier-4 only (≤ 25 % coverage — will yield fewer heads) |
| AGN-dominated sample → class imbalance in followup | Followup cal AUC drops | Report per-class precision; revisit `target_follow_proxy` weighting if Ia fraction is < 10 % |
| `fink_lsst/early_snia` has too few rows to pass trust's internal 10-cal-row gate | Head not created | Expected — log will say skipped; fine |
| Stamp API returns different `classifier_name` for different objects | Mixed vintages | We record `stamp_classifier_version` but don't discriminate in Step 2; fine for weak label |

---

## 6. Validation checklist (run after Step 5)

```python
# 1. LSST truth coverage
snap = pd.read_parquet("data/gold/object_epoch_snapshots_safe_v5b.parquet")
lsst = snap[snap["object_id"].astype(str).apply(infer_identifier_kind) == "lsst_dia_object_id"]
assert lsst["target_class"].notna().mean() > 0.95

# 2. Helpfulness sees non-null is_topclass_correct for LSST experts
h = pd.read_parquet("data/gold/expert_helpfulness_safe_v5b.parquet")
for k in ["fink_lsst/snn", "fink_lsst/cats", "pittgoogle/supernnova_lsst"]:
    nn = h[h["expert_key"]==k]["is_topclass_correct"].notna().sum()
    assert nn > 500, f"{k} has only {nn} usable rows"

# 3. Trust models directory has 4+ new heads
expected = {"fink_lsst__snn","fink_lsst__cats","pittgoogle__supernnova_lsst"}
got = set(os.listdir("models/trust_safe_v5b")) - {"metadata.json"}
assert expected.issubset(got), f"missing: {expected - got}"

# 4. Followup AUC not regressed by more than 0.01
v5  = json.load(open("reports/metrics/followup_metrics_safe_v5.json"))
v5b = json.load(open("reports/metrics/followup_metrics_safe_v5b.json"))
assert v5b["test_calibrated"]["roc_auc"] >= v5["test_calibrated"]["roc_auc"] - 0.01
```

---

## 7. Rollback

All v5b outputs live in `_v5b`-suffixed files / dirs. If v5b is worse than
v5, `rm -rf` the `_v5b` artifacts — v5 is untouched. No `git revert` needed.

---

## 8. Execution-order summary

```
 (0:00) Step 1 launch in background on SCC   — alerce stamp backfill (~10 min)
 (0:10) Step 2 build_lsst_truth.py           (~2 min)
 (0:12) Step 3 merge_lsst_truth.py           (~10 s)
 (0:12) Step 4 patch_snapshot_truth_lsst.py  (~1 min; writes v5b snapshot)
 (0:14) Step 5 qsub run_f3_safe_v5b_*.sh     (~15 min: helpfulness + trust + followup)
 (0:30) Validation checklist                 (~1 min)
 (0:31) v5b is trained; compare vs v5
```

Wall time: **~30 min** from green-light to validated v5b.

---

## 9. v5c — circularity post-mortem and fix (2026-04-23)

### The trap discovered in v5b

v5b's headline +3.5pp AUC gain (0.9249 → 0.9598) was inflated by a **direct
circular label**. The Tier-2 rule set `final_class_ternary = snia` whenever
`stamp_class == SN` AND `max(p_snia from {fink_lsst/snn, cats, pittgoogle_lsst})
>= 0.5`. Those three `p_snia` scores are also features in the followup head.
The model was therefore trained to predict a target that its own inputs define —
text-book leakage.

### Evidence

Per-slice test AUC (v5b, 40,465 test rows):

| Slice | n | pos rate | Test AUC |
|---|---:|---:|---:|
| OVERALL (reported 0.9598) | 40,465 | 0.303 | 0.9602 |
| Weak-label rows (LSST-heavy) | 25,161 | 0.190 | 0.9635 |
| Spec-only (hard truth) | 13,924 | 0.536 | **0.9303** |
| **LSST spec (hard truth)** | **60** | **0.333** | **0.3562** ← below chance |

The LSST spec AUC **below 0.5** is the unambiguous proof: the model has not
learned LSST physics, it has learned the labeling rule.

### The principle

**Label lineage must be disjoint from feature lineage.** A feature that is used
(directly or transitively) to assign a label cannot also be used to predict that
label at the same level of the stack. This is the rule every LSST
early-classification project must honour; broker outputs are so information-rich
that they will quietly re-use themselves as truth if you let them.

### The v5c fix — four edits

1. **`scripts/build_lsst_truth.py`** — new `--tier2-mode {stamp_only, consensus}`
   flag (default `stamp_only`). In stamp_only mode, SN-class stamps at prob ≥ 0.70
   map to `nonIa_snlike` — we **never fabricate `snia` positives from weak labels**.
   All `snia` positive labels originate from Tier-1 TNS spec only. Tier-3 is
   absorbed into Tier-2 (threshold lowered to 0.70). Old consensus mode
   retained only for A/B reproducibility.

2. **`src/debass_meta/models/followup.py`** — new `weak_weight` parameter
   (default 1.0). Applied as `sample_weight = np.where(label_quality in
   {weak, context}, weak_weight, 1.0)` on `LGBMClassifier.fit`. Weak rows still
   teach "not-Ia" but with 1/4 the leverage at the decision boundary.

3. **`scripts/train_followup.py`** — `--weak-weight` CLI flag passes through.

4. **`jobs/run_f3_safe_v5c.sh`** — end-to-end SGE job: rebuild truth (stamp_only)
   → merge → patch snapshot → helpfulness → trust → followup with
   `--weak-weight 0.25`. All outputs suffixed `_v5c`; v5 and v5b untouched for
   comparison.

### Preflight experiments (all PASSED before trusting the full retrain)

- **E1** — `build_lsst_truth.py` in both modes, real LSST sample (N=3,426):
  consensus mode → **172 snia weak labels**; stamp_only → **0**.
- **E2** — patched v5c snapshot: LSST rows with `label_quality=weak` **and**
  `target_class=snia` → **0**. (v5b had ~15k such rows.)
- **E3** — synthetic LightGBM with misleading-weak + informative-spec:
  `weak_weight=0.25` strictly improves spec-subset AUC vs `weak_weight=1.0`,
  confirming `sample_weight` reaches `LGBMClassifier.fit` and downweighting is
  effective. Also exercises the `debass_meta.models.followup.train_followup_model`
  integration path.

### v5c results

| Slice | v5b | **v5c** | Δ |
|---|---:|---:|---:|
| Overall test AUC | 0.9602 | 0.9692 | +0.009 (legit: LSST weak rows are now pure negatives) |
| Spec-only (hard truth) | 0.9303 | **0.9359** | **+0.006** (honest gain) |
| ZTF spec (n≈14.7k) | 0.9319 | **0.9398** | **+0.008** |
| **LSST spec (n=100)** | **0.3562** | **0.5988** | **+0.2426** (sign flip) |

Trust-layer pos_rate sanity check — drops to honest values without circular
labels:

| Head | v5b pos_rate | v5c pos_rate |
|---|---:|---:|
| `fink_lsst/snn` | 0.051 | **0.012** |
| `fink_lsst/cats` | 0.097 | **0.072** |
| `pittgoogle/supernnova_lsst` | 0.291 | **0.239** |

### Why v5c overall AUC rose (not a new circularity)

LSST weak rows now have pos_rate ≈ 0 — all `nonIa_snlike` or `other`. Broker
`p_snia` on genuine AGN/stellar/asteroid LSST objects is correctly low, so the
model predicts ~0 for them. That is **legitimate signal**, not label leakage.
The honesty test is spec-only AUC (0.936, +0.6 pp over v5b) and the LSST spec
sign-flip (0.36 → 0.60) — both moved in the right direction.

### Upstream TODO (for future `submit_retrain_v2.sh` runs)

`scripts/build_truth_multisource.py` does not yet invoke the stamp-only LSST
tier. For automated end-to-end retrains, either:
- call `build_lsst_truth.py --tier2-mode stamp_only` from inside the pipeline
  and merge its output into `object_truth.parquet`; or
- inline the stamp-only logic directly into `build_truth_multisource.py`.

Until then, v5c must be produced via the surgical `run_f3_safe_v5c.sh` job.

### Long-term answer — ELAsTiCC2 sim truth (pending, Task #22)

Simulated LSST truth is the only way to measure LSST meta-classification AUC
on n>>100 Ia labels before Rubin spec data exists. Sketch: pull ≥5k balanced
ELAsTiCC2 sims, run local experts (SuperNNova, SALT3, Bazin/Villar, alerce_lc),
slot in broker scores from published ELAsTiCC2 benchmarks, train a v6 trust +
followup on sim truth. Report sim-AUC alongside ZTF-spec-AUC as the two honest
benchmarks in any paper/slide.

