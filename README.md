# metaDEBASS — Trust-Aware Early-Epoch Transient Meta-Classifier

metaDEBASS is a **meta-layer**, not another classifier. Its inputs are an object's
early lightcurve (3–5 detections) plus the outputs of every available broker and
local classifier; its outputs are (1) a **calibrated per-expert trust** score at
each epoch and (2) a **follow-up priority** for all three science goals
(`snia` / `nonIa_snlike` / `other`), for spectroscopic target selection with
Rubin/LSST and ZTF.

It trains on both surveys through one pipeline (union feature set, LightGBM
native-NaN handling, per-survey heads and calibration) and is **validated
against the live LSST alert stream** on a frozen, never-trained-on benchmark of
real 2026 transients with TNS spectroscopic truth.

**Current version: `fusion_v13b`** (2026-09-25), the version behind the
[TNS × EDP2 explorer](https://trivialtz.github.io/tns-edp2-explorer/). Plan, defects found and
results: `docs/fusion_v13_plan.md`; job: `jobs/run_fusion_v13.sh` (`FUSION_V13_ARM=v13b`).
The architecture is fusion_v11's (`docs/fusion_v11_design.md`, `docs/fusion_v11_spec.md`).

## Status (2026-09-25): fusion_v13b vs fusion_v12

Frozen live-Rubin benchmark (222 objects scored at the latest detection, 78 SNe; never trained on)
and the ZTF locked spectroscopic test. Checkpoints are n_det = 3 / 5 / 10 / latest.

| Evaluation | fusion_v13b | fusion_v12 |
|---|---|---|
| Rubin SN-vs-other AUC, all inputs | 0.930 / 0.895 / 0.875 / 0.952 | 0.937 / 0.894 / 0.842 / 0.929 |
| Rubin calibration (Brier), all inputs | 0.090 / 0.088 / 0.090 / 0.082 | 0.105 / 0.130 / 0.112 / 0.115 |
| Rubin, no broker outputs: AUC at latest | 0.923 | 0.933 |
| Rubin, no broker outputs: median P(SN) on SNe at n = 5 | 0.89 | 0.32 |
| Rubin, lightcurve only (no experts): AUC at latest | 0.908 | 0.500 |
| Rubin ALeRCE stamp trust vs "is it a supernova" | 0.895 | 0.206 (inverted) |
| Rubin Ia given SN, AUC at latest | 0.48 [0.34, 0.61] | 0.55 [0.42, 0.67] |
| ZTF locked test, macro OvR AUC @ n_det = 5 (765 obj) | 0.919 [0.900, 0.937] | 0.893 [0.869, 0.913] |

What changed since v12 (details in the plan): the stamp classifiers are trust-trained on
"is it a supernova"; out-of-fold `q_prior`; LSST weak labels kept out of head 1; availability
dropout (no-broker / no-local / no-expert copies) plus a guard (G8) on availability-vs-class
correlation; head 1 cross-fitted, with calibrators weighted by object mix; ParSNIP dropped and
the local SuperNNova masked on Rubin, since their stub outputs had been marked available and
tracked which objects were re-run. Ia vs non-Ia on Rubin is still at chance, so the explorer shows
only P(supernova) for Rubin objects.

## Earlier status: fusion_v11 (2026-07-05)

| Evaluation | fusion_v11 | fusion_v10 (same frame) |
|---|---|---|
| ZTF locked spec test, snia OvR AUC @ n_det=5 (765 obj) | 0.881 [0.857, 0.905] | 0.923 [0.903, 0.941] |
| ZTF Δ vs v6e2 baseline (snia @5) | **+0.092 [+0.063, +0.120]** | +0.131 |
| Live-LSST benchmark, SN-vs-other AUC (187 obj) | **0.935 [0.875, 0.979]** | 0.515 [0.385, 0.638] |
| Live-LSST benchmark, Ia-vs-rest AUC (45 spec obj) | 0.517 [0.35, 0.70] (null) | 0.383 |
| median p_snia on live spectroscopic SNe Ia | 0.171 | 0.011 (suppressed) |

Read this table honestly:

- **On live LSST, v11 works and v10 did not.** v10 had a structural anti-Ia bug
  (weak LSST labels were force-mapped SN→nonIa while `survey_is_lsst` was a
  feature, so the model learned "LSST ⇒ not Ia") and its fused SN-vs-other was
  a coin flip. v11 removes the bug by construction and its deployed output is
  the best SN-vs-other scorer measured on this benchmark.
- **Ia-vs-nonIa on live LSST is still statistically unresolved** — by us and by
  every broker. LSST brokers only give SN-vs-not this early; the discriminative
  Ia signal awaits the growing pool of survey-era spectroscopic labels
  (the weekly refresh below accumulates them automatically).
- **On ZTF, v11's hierarchical head trades ~0.04 snia AUC vs v10's flat head**
  on identical data and corrected labels (blend and calibration ruled out as
  causes). Both artifact stacks are kept; per-survey routing (v10 head on ZTF,
  v11 on LSST) is the standing deployment option.

## Why a meta-layer

Rubin/LSST generates ~10 million alerts per night; spectroscopic follow-up is
the scarce resource. Brokers (ALeRCE, Fink, Lasair, Babamul, Pitt-Google) and
local experts (SuperNNova, ParSNIP, SALT3-χ², a GRU sequence model) provide
heterogeneous signals with different failure modes, coverage, and time
semantics. metaDEBASS models each expert separately, preserves temporal
exactness (no future information, ever), estimates when each expert can be
trusted, and only then fuses — with a hard guarantee that the fusion cannot
underperform the broker pool it consumes.

## Architecture (fusion_v11)

```
lightcurves (ZTF+LSST)      broker payloads (bronze → silver, exactness-tagged)
        │                                   │
        ▼                                   ▼
  51 base + EXT + negative-flux      28-expert registry → ternary projections
  features, positive-only epochs     (projectors/, per-expert temporal scopes)
        └──────────────┬────────────────────┘
                       ▼
      gold object-epoch snapshots  (n_det = 1..20, no-leakage truncation)
                       ▼
  Stage A — pooled trust LightGBM (expert-ID categorical → per-expert q)
                       ▼
  Stage B — hierarchical follow-up head
      Head-1  P(SN | x)      trained on ALL label tiers (weak labels
                             supervise only the level they constrain)
      Head-2  P(Ia | SN, x)  trained on SPECTROSCOPIC labels ONLY (guard G7)
      compose: p_snia = P1·P2,  p_nonia = P1·(1−P2),  p_other = 1−P1
                       ▼
  per-survey calibration (isotonic, Platt fallback at small n)
                       ▼
  anchored blend:  p = α·model + (1−α)·anchor
      anchor = trust-weighted broker pool with an Ia-capability mask
      α fit per survey × coverage cell, 1-SE-prefer-anchor, survey-guarded
      guarantee (G3): blended ≥ anchor on every calibration slice
                       ▼
  Mondrian conformal sets + utility·p̂ budget/FDR selection → priority lists
```

Key design rules, each the answer to a measured failure:

- **Every label supervises exactly the level it constrains.** A stamp
  classifier's "SN" knows SN-ness, not subtype — it can never touch the Ia
  axis. This is what makes the v10 anti-Ia bug structurally impossible.
- **The meta-layer must never underperform its best input** (the anchored
  blend; v9c's fusion had been *anti-correlated* with its own stamp input on
  LSST).
- **No-leakage epochs**: features at `n_det = k` use only the first k
  *positive* detections; expert events must satisfy `event_time ≤ alert_time`
  (or carry an explicit unsafe/static tag). Negative-flux detections feed only
  the 5 dedicated `NEG_FEATURE` columns (fading tails are evidence, not noise).
- **Locked tests are sacred**: the ZTF v6e2 765-object spec test is preserved
  verbatim across every version (guard G5b asserts value-identity of the
  rebuilt gold), and the LSST-live benchmark below is frozen and quarantined
  from train/cal at build **and** train time (guard G6, association-aware —
  an object's ZTF twin counts as the object).

The GRU sequence expert (`seq_v11`: negative-detection tokens, per-survey
normalization, 5-fold OOF training-row inference) is a registered expert like
any broker and — as of v11 — passes its inclusion gate for the first time.

## Truth & the label engine

Five truth tiers (TNS spectroscopic > TNS-named-untyped > broker consensus >
host/catalog context > weak), built by:

- `scripts/download_tns_bulk.py` — daily TNS public-objects dump (full pull +
  diff upsert); removes per-object API rate limits.
- `scripts/rederive_spec_truth.py` — **run before any training**: the historic
  truth table had 3,149 BTS-unclassified rows force-mapped to "spectroscopic
  nonIa" (61% of that class; dozens provably wrong). This rederives them via
  TNS name-join and demotes the unresolved to a weak `bts_untyped` tier →
  `data/truth/object_truth_v11.parquet`.
- `scripts/build_truth_lsst_live.py` — **epoch-aware** LSST↔TNS crossmatch
  (sep ≤ 2″ AND discovery date consistent with the detection window; stale
  positional matches demoted and excluded — 44/150 of a naive crossmatch's
  "spectroscopic" labels were stale 2017–2024 names).
- `scripts/harvest_ztf_lsst_associations.py` — ZTF↔LSST association harvest
  via TNS internal names (the conesearch direction yields ~0): real LSST
  photometry with ZTF-grade spectroscopic labels (~215 objects, 118 SNe Ia,
  all ≤ 2″) — the main LSST spec-training pool today.
- `jobs/refresh_lsst_live.sh` — weekly: refresh TNS → rebuild live truth →
  hash-route new spectroscopic arrivals into the frozen benchmark → re-score.
  metaDEBASS is a *continuously evaluated* system, not a one-off paper number.

**The LSST-live benchmark** (`data/gold/lsst_live_locked_test.json`): 195
frozen never-trained-on 2026 LSST objects (45 spectroscopic + 150
catalog-confirmed non-SNe), with an append-only manifest; ids whose ZTF twins
sit in locked training are recorded in `excluded_ids` with reasons, never
silently dropped.

## Registered experts (28; `src/debass_meta/projectors/base.py:EXPERT_REGISTRY`)

| Source | Experts | Live on LSST today |
|---|---|---|
| Fink LSST | SNN, CATS, EarlySNIa (per-alert scores) | ✔ (EarlySNIa needs ≥7 epochs) |
| ALeRCE | Rubin stamp; ZTF stamp ×2, LC ×4 | ✔ (stamp) |
| Lasair | Sherlock context | ✔ |
| Babamul | context flags | ✔ |
| Fink ZTF | SNN, RF-Ia, SLSN | ZTF only |
| Pitt-Google | SNN LSST/ZTF, UPSILoN (BigQuery) | ✔ (LSST) |
| AMPEL / ANTARES | SNGuess, Superphot+, ParSNIP-FollowMe | dormant |
| Local (rerunnable) | SuperNNova, ParSNIP, ALeRCE-LC, SALT3-χ², Bazin/Villar, **seq_v9/seq_v11 GRU** | ✔ |

SN-*filter* experts (binary SN-vs-not) are trained with `target=is_sn`, and the
blend's Ia axis pools only genuinely Ia-capable experts (a stamp with no Ia
class contributes SN-ness, never a structural p_snia = 0).

## Quick start

```bash
# Python ≥ 3.10; local dev uses a venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt

# Credentials in ./.env (see table below)

# 1) Truth prerequisites (B0 — required before any v11 training)
python scripts/download_tns_bulk.py --mode full
python scripts/rederive_spec_truth.py            # → data/truth/object_truth_v11.parquet

# 2) Data: lightcurves + broker backfill → silver
python scripts/fetch_lightcurves.py --from-labels data/labels.csv
python scripts/backfill.py --broker all --from-labels data/labels.csv --parallel 8
python scripts/normalize.py

# 3) Gold + train + score (v11)
python scripts/build_snapshots_fusion.py         # gold snapshots + split manifest (G5b/G6)
python scripts/train_fusion_v11.py               # Stage A → heads → blend → conformal → guards
python scripts/score_fusion_v11.py --snapshots data/gold/object_epoch_snapshots_fusion_v11.parquet
```

Legacy entry points (`scripts/score_nightly.py`, `jobs/submit_retrain_v2.sh`,
`scripts/discover_lsst_training.py`) still work for the v7-era per-expert
payloads; new work should target the fusion v11 stack.

### Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `LASAIR_TOKEN` (or `LASAIR_ZTF_TOKEN` / `LASAIR_LSST_TOKEN`) | Lasair ZTF + LSST APIs |
| `TNS_API_KEY`, `TNS_TNS_ID`, `TNS_MARKER_NAME` | TNS bulk dump + crossmatch |
| `RSP_TOKEN` | Rubin RSP TAP (DP1 catalogs) |
| `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT` | Pitt-Google BigQuery |

Gotchas that will save you an afternoon: call
`load_dotenv("<repo>/.env")` with an explicit path;
`normalize_detection(..., survey=...)` takes `"LSST"`/`"auto"` (uppercase —
lowercase silently routes to the ZTF parser); Lasair's LSST endpoint is
`lasair.lsst.ac.uk` (not `lasair-lsst.lsst.ac.uk`, which drops POST auth).

## Training at scale (BU SCC)

The active clone is `/project/pi-brout/rubin_hackathon` (venv: `.venv`).
Full training runs as a three-job SGE chain:

```bash
cd /project/pi-brout/rubin_hackathon
bash jobs/submit_fusion_v11_chain.sh
#  → build   (truth rederive + gold + split + G5b/G6)      16 cores, ~3.5 h
#  → pretrain (seq_v11 SSL + 5-fold OOF classifier)         GPU: -l gpu_c=8.0
#  → expert  (Stage A/B + blend + conformal + guards + benchmark re-score)
```

`FUSION_V11_FORCE=1` forces rebuilds (steps are otherwise idempotent);
`FUSION_V11_ELASTICC2=1` enables the optional ELAsTiCC2 GRU pretraining corpus.
**Never run scoring/training on the login nodes** — interactive processes are
killed at 15 CPU-minutes; use `qsub -P pi-brout` / `qrsh`. Job logs append
across resubmissions (old tracebacks stay in the `.err` above the real one).

## Evaluation contract (guards)

Training refuses to hand you a contaminated number:

| Guard | Assert |
|---|---|
| G1 | ZTF locked-test headline vs prior version (eval-time report) |
| G2 | anti-Ia regression test on LSST spec-Ia rows (loudly `UNEVALUABLE` below n=10 rather than silently green) |
| G3 | blended ≥ anchor on every calibration slice |
| G5b | rebuilt gold value-identical to locked v10 gold on ZTF rows (rtol 1e-9) |
| G6 | frozen benchmark ids (and their association twins) never in train∪cal — at build **and** at train |
| G7 | Head-2 training rows all carry concrete spectroscopic subtype provenance |

439 pytest tests cover the pipeline (`pytest tests/ -q`).

## Repository map

```
src/debass_meta/
  access/        Broker adapters (ALeRCE, Fink ZTF+LSST, Lasair dual, Babamul, TNS, RSP, PGB)
  features/      Detection normalization, 51+EXT+neg LC features, sequence datasets
  ingest/        Bronze → silver → gold builders
  models/        pooled_trust, hierarchical_followup, anchor_blend, seq_classifier, conformal
  projectors/    Expert score → ternary projections (28-expert registry)
  experts/local/ Rerunnable local experts (SNN, ParSNIP, SALT3, GRU seq_v9/v11)
scripts/         Entry points (truth engine, gold builder, train/score fusion_v11, legacy v7)
jobs/            SCC chains (submit_fusion_v11_chain.sh), weekly refresh_lsst_live.sh
docs/            fusion_v11_design.md / _spec.md / _deploy.md, execplans, review memos
reports_from_scc/ Mirrored SCC results (fusion_v8 … fusion_v11 + lsst_live_bench)
data/, models/   (gitignored) medallion data + trained artifacts per version
CLAUDE.md        Working context for AI-assisted development (kept current)
```

## Known limitations

- Ia-vs-nonIa on live LSST is unresolved by all systems at current
  spectroscopic-label depth (~45 clean objects); the benchmark grows weekly.
- v11's hierarchical head costs ~0.04 snia AUC on ZTF vs v10's flat head
  (same-frame); per-survey routing is the standing mitigation.
- Nearly all LSST photometry to date is commissioning/early-ops (official
  survey began 2026-06-30) with broker pipeline churn across that span.
- Fink LSST EarlySNIa needs ≥7 epochs; ALeRCE LSST has stamp only (no LC
  classifiers yet); several registered experts are dormant pending deployment.
- Weak-label calibration overrates the model arm on LSST (spec benchmark says
  anchor wins) — the blend's α absorbs this, but treat weak-label-only
  metrics with suspicion.

## Citation

If you use this code, please cite the underlying experts and brokers:
ALeRCE (Förster et al. 2021, AJ 161 242), Fink (Möller et al. 2021, MNRAS 501
3272), Lasair (Smith et al. 2019, RNAAS 3 26), SuperNNova (Möller & de
Boissiere 2020, MNRAS 491 4277), ParSNIP (Boone 2021, AJ 162 275), LightGBM
(Ke et al. 2017).
