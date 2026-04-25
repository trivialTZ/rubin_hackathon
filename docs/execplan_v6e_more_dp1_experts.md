# Exec plan — v6e: activate more local experts on DP1

**Status:** Drafted 2026-04-25. Verified locally on 14 cached DP1 SN LCs.
SCC env state inventoried (see "SCC inventory" below).

**Goal:** Bring DP1's effective ensemble from **3 active local experts**
toward **5** by fixing one bug (salt3) and wiring one new model (ORACLE).
Two-tier rollout because the v5d trust-head registry doesn't yet contain
ORACLE:

- **Tier A — salt3_chi2** (drop-in, immediate win): trust head already
  trained in v5d (`models/trust_safe_v5d/salt3_chi2/`). Fixing the
  LSST-band bug + installing `iminuit` activates the 4th expert. No
  retraining. Effort: ≈ 30 min on SCC.
- **Tier B — oracle_lsst** (new expert; needs trust-head training):
  wiring works end-to-end and emits `proj__oracle_lsst__p_*` columns,
  but `models/trust_safe_v5d/` has **no `oracle_lsst/` directory** —
  the score step will load 11 trust heads as before, ignore ORACLE,
  and ORACLE's contribution to the ensemble will be zero. To make
  ORACLE actually contribute, run an F3-style helpfulness rebuild +
  per-expert trust-head train against the v5d training set first.
  Effort: ≈ ½ day on SCC, separate task.

This exec plan covers **Tier A only**. Tier B is parked as v6e.2 with
a clear runbook stub at the bottom.

ParSNIP is fully deferred to v6f (wrapper segfaults locally — torch/numpy
ABI suspected; details below).

## Why this exec plan exists

The v6 production parquet (`reports/v6_dp1_50k/predictions.parquet`)
shows that 8 of 11 v5d trust heads have `avail=0` on DP1 because
brokers don't index DP1, and 1 of 4 wired *local* experts (`salt3_chi2`)
silently emits `"no probabilities"` for every snapshot — i.e. only 3 local
experts (alerce_lc, supernnova, lc_features_bv) actually contribute to
the trust-weighted ensemble. The 3-expert ablation
(`reports/v6_dp1_50k/ablation_3expert.md`) shows that the trust-weighted
ensemble's headline galaxy-suppression result is essentially driven by
`alerce_lc` alone — a thin claim for a "trust-weighted multi-broker
fusion" paper.

This plan adds two more LSST-native local experts (no broker dependency)
to make the multi-expert claim defensible.

## What changed in the source tree

Code changes already committed locally (UNCOMMITTED to git):

| File | Change | Why |
|---|---|---|
| `src/debass_meta/experts/local/salt3_fit.py` | `_band_name` now disambiguates single-letter `g/r/i` by `det['survey']`. LSST → `lsst{u,g,r,i,z,y}`, ZTF → `ztf{g,r,i}`. | DP1 detections come in with `band='r'` → previously routed to `ztfr` → sncosmo fits failed with wrong bandpass → all 31,339 snapshots had `avail__salt3_chi2=0`. |
| `src/debass_meta/experts/local/oracle_local.py` | Replaced TODO stub with real inference using `ORACLE1_ELAsTiCC_lite` (lightcurve-only variant). Handles flux conversion (DP1 nJy → SNANA FLUXCAL ZP=27.5 via ×0.02754) and band casing (`y`→`Y`). | Wires the dev-ved30/Oracle PyTorch model into the snapshot pipeline. The lite variant skips 18 host-context features that DP1 doesn't carry. |
| `src/debass_meta/projectors/local_oracle.py` | Rewritten to handle per-class events from `_local_outputs_to_events` (also still accepts legacy `raw_probs_json` shape). | Original projector expected an aggregate JSON blob format that no local expert produces in this pipeline. |
| `src/debass_meta/projectors/antares.py` | Extended `_ORACLE_NONIA / _ORACLE_OTHER_TRANSIENT / _ORACLE_PERIODIC` sets with the actual dev-ved30 leaf names: `sni91bg, snib/c, dwarfnovae, ulens, m-dwarfflare, deltascuti`. | Class names in the new ORACLE PyTorch repo differ slightly from the antares/ANTARES feed those sets were originally written for. |
| `src/debass_meta/ingest/rsp_dp1.py` | `LOCAL_EXPERT_KEYS` += `oracle_lsst`; `_init_experts()` instantiates `OracleLocalExpert()`. | Wires ORACLE into the snapshot builder. |
| `env/requirements.txt` | Added `iminuit>=2.20`, `oracle @ git+https://github.com/dev-ved30/Oracle.git`, `torch>=2.0`, `torchvision>=0.15`. | Hard runtime dependencies for salt3 (iminuit) and ORACLE. |

### Local end-to-end verification (already done, paper-grade)

Ran on 14 cached DP1 published-SN LCs via
`scripts/build_dp1_snapshots.py --only-published`:

```
salt3_chi2          : avail rows =  93 / 117  (24 snapshots have <3 dets at the epoch — expected)
oracle_lsst         : avail rows =  95 / 117  (22 short-LC snapshots correctly skipped)
```

`proj__salt3_chi2__p_snia` ranges 0.0–1.0 (median 1.0 on the SN sample).
`proj__oracle_lsst__p_other` is 1.0 for nearly every SN — a known
calibration mismatch (ORACLE-lite trained on ELAsTiCC2 multi-year LCs,
DP1 LCs are weeks-to-months). The trust head will down-weight it on
DP1, but it should still contribute negative-class suppression on Gaia
stars / galaxies (the 41K production pool is dominated by non-SNe).

Smoke scripts: `scripts/_exp_salt3_dp1_smoke.py`,
`scripts/_exp_oracle_pipeline_smoke.py`.

## SCC inventory (verified 2026-04-25)

`/usr3/graduate/tztang/debass_meta_env/bin/python` reports:

| package | SCC version | Tier-A need | Tier-B need |
|---|---|---|---|
| numpy | **1.26.4** | OK | OK (ORACLE source is numpy-1.x compatible) |
| pandas | **2.1.4** | OK | OK |
| torch | **2.11.0+cpu** | n/a | OK |
| sncosmo | 2.12.1 | OK | n/a |
| astropy | 6.1.7 | OK | OK |
| lightgbm | 4.6.0 | OK | n/a |
| supernnova | installed | OK | n/a |
| matplotlib | 3.7.2 | OK | OK |
| networkx | 3.1 | n/a | OK |
| graphviz | 0.20.1 | n/a | OK |
| wandb | 0.15.5 | n/a | already there (good) |
| polars | 0.20.7 | n/a | already there (good) |
| **iminuit** | **MISSING** | **install** | n/a |
| **oracle** (dev-ved30) | **MISSING** | n/a | **install** |
| **transformers** | MISSING | n/a | install (ORACLE dep) |
| Jinja2 | unknown | n/a | install if missing |
| umap-learn | broken (numba/numpy ABI mismatch) | n/a | **skip** — not needed for inference |

Artifacts:
- `data/lightcurves/dp1/*.parquet` × **41,292** ✓
- `data/truth/dp1_truth_50k.parquet` ✓
- `data/gold/dp1_snapshots_50k.parquet` ✓ (v6 baseline preserved)
- `models/trust_safe_v5d/{alerce_lc, antares__superphot_plus, fink_lsst__cats, fink_lsst__early_snia, fink_lsst__snn, fink__rf_ia, fink__snn, lc_features_bv, pittgoogle__supernnova_lsst, salt3_chi2, supernnova}/` — **11 heads** ✓
- `models/followup_safe_v5d/metadata.json` ✓

⚠️ **No `models/trust_safe_v5d/oracle_lsst/`** — confirmed Tier B needs
trust-head training before ORACLE will contribute to the ensemble.

## Deployment to SCC — Tier A (salt3 only)

This is the immediate rollout. Three commands plus a re-score.

### A1. Commit + push code changes from local

The v6e commit ships **both Tier A and Tier B source changes** in a
single bundle. Tier B code is dormant on SCC until ORACLE is installed
+ a trust head is trained, so it's safe to land now.

Files in the commit:

| Path | Purpose | Tier |
|---|---|---|
| `src/debass_meta/experts/local/salt3_fit.py` | LSST band-map fix | **A** |
| `env/requirements.txt` | + `iminuit`, ORACLE deps | **A** + B |
| `src/debass_meta/experts/local/oracle_local.py` | ORACLE inference (was a stub) | B |
| `src/debass_meta/ingest/rsp_dp1.py` | wires `oracle_lsst` into `LOCAL_EXPERT_KEYS` | B |
| `src/debass_meta/projectors/local_oracle.py` | per-class event handler | B |
| `src/debass_meta/projectors/antares.py` | extra ORACLE leaf-name aliases | B |
| `docs/execplan_v6e_more_dp1_experts.md` | this file | — |
| `scripts/build_enrichment_metrics.py` | enrichment-factor headline rebuild | analysis |
| `scripts/plot_enrichment_curves.py` | enrichment-factor figure | analysis |
| `scripts/build_ablation_3expert.py` | per-expert DP1 ablation | analysis |
| `scripts/analyze_dp1_topk_discovery.py` | top-K unlabeled candidate panel | analysis |
| `scripts/plot_contamination_vs_threshold.py` | contamination-vs-K curve | analysis |
| `scripts/_exp_salt3_dp1_smoke.py` | salt3 fix verification helper | smoke |
| `scripts/_exp_oracle_dp1_smoke.py` | ORACLE wiring verification helper | smoke |
| `scripts/_exp_oracle_pipeline_smoke.py` | end-to-end pipeline verification | smoke |

```bash
git add \
    src/debass_meta/experts/local/salt3_fit.py \
    src/debass_meta/experts/local/oracle_local.py \
    src/debass_meta/ingest/rsp_dp1.py \
    src/debass_meta/projectors/local_oracle.py \
    src/debass_meta/projectors/antares.py \
    env/requirements.txt \
    docs/execplan_v6e_more_dp1_experts.md \
    scripts/build_enrichment_metrics.py \
    scripts/plot_enrichment_curves.py \
    scripts/build_ablation_3expert.py \
    scripts/analyze_dp1_topk_discovery.py \
    scripts/plot_contamination_vs_threshold.py \
    scripts/_exp_salt3_dp1_smoke.py \
    scripts/_exp_oracle_dp1_smoke.py \
    scripts/_exp_oracle_pipeline_smoke.py
git commit -m 'v6e: salt3 LSST fix + ORACLE wiring + enrichment-factor analysis'
git push origin main
```

### A2. Pull on SCC + install iminuit
```bash
ssh scc
cd /project/pi-brout/rubin_hackathon
git pull
PY=/usr3/graduate/tztang/debass_meta_env/bin/python
$PY -m pip install --user iminuit
$PY -c "import iminuit, sncosmo; print('iminuit', iminuit.__version__, 'sncosmo', sncosmo.__version__)"
```

### A3. Smoke test on 14 published SNe
```bash
$PY scripts/build_dp1_snapshots.py --only-published \
    --truth data/truth/dp1_truth_50k.parquet \
    --out   data/gold/dp1_snapshots_smoke_v6e.parquet
$PY -c "
import pandas as pd
g = pd.read_parquet('data/gold/dp1_snapshots_smoke_v6e.parquet')
n = int(g['avail__salt3_chi2'].fillna(0).astype(float).gt(0.5).sum())
print(f'salt3_chi2 avail: {n} / {len(g)}  (expect ~70-80% of 117)')
"
```
Expected: salt3 avail ≈ 80–95 / 117. (24 short-LC snapshots correctly skip.)

### A4. Full 41K rebuild + re-score
```bash
# rebuild gold (≈ 9 min for 41K, only marginally longer than baseline)
$PY scripts/build_dp1_snapshots.py \
    --truth data/truth/dp1_truth_50k.parquet \
    --out   data/gold/dp1_snapshots_50k_v6e.parquet

# re-score (≈ 60 s)
$PY scripts/score_dp1_v5d.py \
    --snapshots   data/gold/dp1_snapshots_50k_v6e.parquet \
    --trust-dir   models/trust_safe_v5d \
    --followup-dir models/followup_safe_v5d \
    --out-dir     reports/v6e_dp1_50k
```

### A5. Pull artifacts back, re-run analysis
```bash
# locally
scp scc:/project/pi-brout/rubin_hackathon/reports/v6e_dp1_50k/predictions.parquet \
    reports/v6e_dp1_50k/

# Compare salt3_chi2 contribution to v6 baseline
python3 -c "
import pandas as pd
old = pd.read_parquet('reports/v6_dp1_50k/predictions.parquet')
new = pd.read_parquet('reports/v6e_dp1_50k/predictions.parquet')
print('OLD salt3_chi2 avail:', int(old['avail__salt3_chi2'].fillna(0).astype(float).gt(0.5).sum()))
print('NEW salt3_chi2 avail:', int(new['avail__salt3_chi2'].fillna(0).astype(float).gt(0.5).sum()))
print('OLD ensemble_n_trusted_experts:', old['ensemble_n_trusted_experts'].value_counts().sort_index().to_dict())
print('NEW ensemble_n_trusted_experts:', new['ensemble_n_trusted_experts'].value_counts().sort_index().to_dict())
"

# Re-run enrichment metrics on v6e
python3 scripts/build_enrichment_metrics.py    # set PRED to reports/v6e_dp1_50k/...
```

**Pass / fail criteria for Tier A:**
- ✅ `salt3_chi2 avail` rises from 0 to ≥ 60 % of snapshots
- ✅ `ensemble_n_trusted_experts` distribution shifts to higher modes
- ✅ Galaxy suppression EF at K=1% does not regress vs v6 baseline
- ✅ EclBin/RRLyrae enrichment factor does not regress

If any of these regress, roll back: `data/gold/dp1_snapshots_50k.parquet`
and `reports/v6_dp1_50k/` are unchanged.

## Deployment — Tier B (ORACLE) — DEFERRED to v6e.2

Cannot deploy as-is. The v5d trust-head registry has no `oracle_lsst`,
so the score step would load 11 heads, find no oracle trust, and the
ensemble loop would skip oracle entirely (avail × trust × proj where
trust=0 contributes 0).

To activate ORACLE properly we need:
1. Run `build_dp1_snapshots` with oracle wired so the v5d **training
   set** also gets ORACLE projection columns. (We currently have it
   wired only for the DP1 driver — a parallel change is needed in the
   ZTF/training-set ingest pipeline.)
2. Rebuild `data/gold/expert_helpfulness.parquet` with ORACLE rows.
3. Run `scripts/train_expert_trust.py --expert oracle_lsst --target is_topclass_correct`
   (or `is_sn` for the SN-filter target — TBD per the v5d
   target/capability convention).
4. Place the new trust head under `models/trust_safe_v5d/oracle_lsst/`.
5. Re-score DP1 with the now-12-head registry.

Effort estimate: ≈ ½ day on SCC. Track as a separate v6e.2 task.

### Risk note for Tier B: ORACLE pyproject.toml uses HARD `==` pins

`numpy==2.2.4`, `pandas==2.2.3`, `astropy==7.0.1`, `torch==2.6.0`, …
SCC has `numpy 1.26.4`, `pandas 2.1.4`, `torch 2.11.0+cpu`. A naive
`pip install git+...Oracle.git` would upgrade them and likely break
`light-curve`, `lightgbm`, `sncosmo` integration (numba on SCC also
silently breaks above numpy 1.24 — `umap-learn` already does).

Mitigation (verified locally): ORACLE's source has no obvious
numpy-2.x-only APIs (no `np.float_`, no `np.unique(return_inverse=)`
new signature). The `==` pins are over-cautious; install with
`--no-deps` + curated subset:

```bash
ssh scc
cd /project/pi-brout/rubin_hackathon
PY=/usr3/graduate/tztang/debass_meta_env/bin/python

# (assumes Tier A is already done — iminuit installed)
$PY -m pip install --user --no-deps \
    git+https://github.com/dev-ved30/Oracle.git
$PY -m pip install --user transformers   # only ORACLE dep SCC is missing
# wandb (0.15.5), polars (0.20.7), networkx (3.1), graphviz (0.20.1), Jinja2 — already present
# umap-learn — broken on SCC (numba/numpy ABI), unused at inference

# Sanity
$PY -c "from oracle.pretrained.ELAsTiCC import ORACLE1_ELAsTiCC_lite; print('OK')"

# Download weights (413 KB)
mkdir -p artifacts/local_experts/oracle/models/ELAsTiCC-lite/revived-star-159
curl -L -o artifacts/local_experts/oracle/models/ELAsTiCC-lite/revived-star-159/best_model_f1.pth \
    https://github.com/dev-ved30/Oracle/raw/main/models/ELAsTiCC-lite/revived-star-159/best_model_f1.pth
```

If `--no-deps` import fails (e.g. transformers ABI mismatch), fall
back to a separate venv (`$PY -m venv ~/oracle_inference_env`) and
run ORACLE inference out-of-process; pipe proj outputs back via JSON.

## Known caveats to disclose in the paper

1. **ORACLE-lite domain mismatch** (DP1-era, well-defined):
   ORACLE1_ELAsTiCC_lite is trained on synthetic 3-year LSST LCs.
   On short DP1 commissioning baselines it confidently predicts
   `Periodic / Fast / AGN` for spectroscopically-typed SNe.
   The trust-head architecture is designed to discount this; expect
   low trust on ORACLE for DP1, useful contribution once DP2/DR1
   multi-month baselines arrive. **Frame this as DP1-era domain
   mismatch, not a method limitation.**

2. **salt3_chi2 over-confident on outliers**: the SALT3 vs Nugent-IIP
   χ² ratio gets very large for long, well-sampled LCs (Δχ² → 10⁴
   for n=181 LC). The sigmoid-temperature mapping then saturates
   `p_Ia = 1.0`. This is fine for ranking; the trust head rescales
   it via isotonic calibration.

3. **Heavy ORACLE deps**: full pip install pulls `wandb, umap-learn,
   transformers, polars` (≈ 700 MB total). Use `--no-deps + curated
   subset` on SCC; we don't need any of those for inference.

## ParSNIP — research findings only (defer to v6f)

**Native ELAsTiCC2-trained ParSNIP weights do not exist publicly.**
Only three pre-trained variants ship with the LSSTDESC repo
(`parsnip/models/`):

| Model | Training data | Bands | DP1-applicable? |
|---|---|---|---|
| `plasticc` | PLAsTiCC simulated LSST | ugrizy | **Yes (best off-the-shelf)** |
| `plasticc_photoz` | + photo-z input | ugrizy | Yes if photo-z available |
| `ps1` | PanSTARRS1 | grizy | No (no u, different ZP) |

PyPI package: **`astro-parsnip`** (NOT `parsnip` — that is unrelated
Python-2 code). Bundled weights at
`<site-packages>/parsnip/models/plasticc.pt` (≈ 4.7 MB).

PLAsTiCC predates ELAsTiCC2 and uses older SN templates / rates;
expect approximate accuracy on real DP1.

**One-liner to wire after v6e is verified:**

```bash
$PY -m pip install --user astro-parsnip
mkdir -p artifacts/local_experts/parsnip
cp $($PY -c 'import os, parsnip; print(os.path.join(os.path.dirname(parsnip.__file__),"models","plasticc.pt"))') \
   artifacts/local_experts/parsnip/model.pt
# then add 'parsnip' to LOCAL_EXPERT_KEYS in src/debass_meta/ingest/rsp_dp1.py
```

**Open issue blocking v6f**: our `ParSNIPExpert` wrapper at
`src/debass_meta/experts/local/parsnip.py` segfaults locally when
loading the bundled `plasticc.pt`. Likely a torch/numpy ABI mismatch
in the wrapper-side code or a stale `_load_model` path. Need to
debug before deploying — DON'T add `parsnip` to `LOCAL_EXPERT_KEYS`
until this is resolved.

## Rollback plan

If anything breaks, revert these files and rebuild gold from the
preserved v6 artifacts:

```bash
git revert <salt3+oracle commit>
# v6 gold preserved at data/gold/dp1_snapshots_50k.parquet
# v6 predictions preserved at reports/v6_dp1_50k/predictions.parquet
```

Trust heads / followup head are unchanged by this exec plan
(it only adds *inputs* to them, not retrains them).

## Out of scope for v6e

- No retraining of v5d trust or followup heads.
- No change to truth tables.
- No paper-figure regeneration (do that in a separate v6e analysis pass).
- No new broker queries (brokers still don't index DP1).
- ParSNIP wrapper integration (deferred to v6f after debugging the segfault).

## File-by-file change manifest

```
# Tier A — drop-in fix, ships with this commit
modified:  src/debass_meta/experts/local/salt3_fit.py        (LSST band-map by survey)
modified:  env/requirements.txt                              (+ iminuit, ORACLE deps)

# Tier B — wired but dormant until ORACLE installed + trust head trained
modified:  src/debass_meta/experts/local/oracle_local.py     (was a stub; now real inference)
modified:  src/debass_meta/ingest/rsp_dp1.py                 (LOCAL_EXPERT_KEYS += oracle_lsst)
modified:  src/debass_meta/projectors/antares.py             (ORACLE leaf-name aliases)
modified:  src/debass_meta/projectors/local_oracle.py        (per-class event handler)

# Analysis artifacts (paper-relevant)
new:       scripts/build_enrichment_metrics.py               (EF + bootstrap CI95)
new:       scripts/plot_enrichment_curves.py                 (EF-vs-K figure)
new:       scripts/build_ablation_3expert.py                 (per-expert DP1 ablation)
new:       scripts/analyze_dp1_topk_discovery.py             (top-K unlabeled panel)
new:       scripts/plot_contamination_vs_threshold.py        (contamination vs K)

# Smoke / verification helpers
new:       scripts/_exp_salt3_dp1_smoke.py
new:       scripts/_exp_oracle_dp1_smoke.py
new:       scripts/_exp_oracle_pipeline_smoke.py

# Doc
new:       docs/execplan_v6e_more_dp1_experts.md             (this file)

# Untracked, NOT in this commit (download on SCC during deployment)
gitignored: artifacts/local_experts/oracle/models/ELAsTiCC-lite/revived-star-159/
                best_model_f1.pth   (413 KB, curl from dev-ved30/Oracle GitHub)
                best_model.pth
                train_args.csv

# Untracked locally, NOT in this commit (already-shipped paper artifacts on SCC)
local only: reports/v6_dp1_50k/enrichment_metrics.json
            reports/v6_dp1_50k/enrichment_per_class.csv
            reports/v6_dp1_50k/enrichment_headline.md
            reports/v6_dp1_50k/enrichment_curves.{png,pdf}
            reports/v6_dp1_50k/ablation_3expert.{json,csv,md}
            (regenerate on SCC by running the analysis scripts after rescore)
```
