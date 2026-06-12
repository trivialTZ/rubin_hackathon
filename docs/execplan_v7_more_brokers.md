# Exec plan — v7: add more LSST brokers / classifiers to metaDEBASS

**Status:** Drafted 2026-04-25; Tier 2D Phases 1-3 **shipped** 2026-04-26.
Tier 1A (POI) blocked on no public API — emailed POI team. Tier 1B is
Fink-ZTF-only until Fink ships SLSN/TDE on the LSST schema. Tier 2C
AMPEL collapsed to "vendor SNGuess locally" (weights are open-source).

**Tier 2D Babamul progress (2026-04-26):**
- Phase 1 — adapter at `src/debass_meta/access/babamul.py`, projector at
  `src/debass_meta/projectors/babamul.py`, registered in `EXPERT_REGISTRY`.
  Smoke-tested: bronze → silver → gold all green; 13 babamul columns land
  in the snapshot table.
- Phase 2 — pulled 1,920-ZTF cohort, **74.3% Babamul coverage** in 41s.
  Discriminative: snia 0.3% star=1 vs other 68%; snia 0% rock=1 vs other 26%.
- Phase 3 — silver/gold pipeline auto-discovers Babamul through standard
  `expert_key='babamul'` + `temporal_exactness='static_safe'` paths.
- Phase 4-5 — runbook at `docs/runbook_v7_babamul_train.md`, runs on SCC.

**Goal:** Fill the broker-coverage gaps identified after the v6e.2
release. The paper currently lists three honest limitations in
`paper/metaDEBASS_aas/sections/discussion.tex`:

1. Periodic-var operational up-ranking (EclBin+RRLyrae `p_follow` EF 17.88).
2. Brokers don't index DP1 → only 4 local experts contribute on DP1.
3. The 11-head trust registry exists but most heads are inactive on DP1.

This plan attacks (1) and (2) by wiring brokers/classifiers that exist
in the LSST community today but aren't yet integrated. Each addition is
scored by **paper-impact-per-effort**, not alphabetical order.

## Current state baseline (v6e.2, locked)

- 11 trained trust heads + 1 followup head (test AUC 0.970).
- 26 experts in `EXPERT_REGISTRY`; 15 are not in the model (unsafe
  temporal mode, weights unavailable, or under-populated).
- Brokers we already query: **Fink** (ZTF + LSST), **ALeRCE** (ZTF +
  LSST stamp), **Lasair** (Sherlock), **Pitt-Google** (ZTF + LSST),
  partial **ANTARES** via `antares/superphot_plus`.
- Brokers we do **not** query: **AMPEL**, **Babamul**, **POI Broker**,
  **SNAPS** (solar system, not relevant).
- Calendar pin: **Rubin Broker Summit 8–19 June 2026** (IJCLab) — most
  Tier-3 deployment dates likely firm up there.

## Generic add-a-broker recipe

For any new expert, the work touches the same eight slots. Doing them
in this order respects the no-leakage contract and lets you bail early
if API access turns out broken.

1. **Verify access** — small probe script
   `scripts/_exp_<broker>_probe.py` (mirrors the existing `_exp_*`
   pattern) hitting one DP1 + one ZTF object. **Fail fast** if no
   auth/coverage.
2. **`EXPERT_REGISTRY` entry** in
   `src/debass_meta/projectors/base.py` with explicit `temporal_mode`
   (`exact_alert` / `latest_unsafe` / `static_safe` / `rerun_exact`)
   and `survey`.
3. **Adapter** in `src/debass_meta/access/<broker>.py` mirroring
   `access/fink_lsst.py`. Returns
   `list[(diaObjectId, ts, raw_payload_dict)]`.
4. **Bronze write** plumbed into `scripts/backfill.py` (single new
   branch on broker key; reuses `--parallel` machinery).
5. **Normalize** rule in `scripts/normalize.py` (raw →
   `silver/broker_events.parquet` rows with the broker's class
   probabilities).
6. **Projector** in `src/debass_meta/projectors/<expert>.py` mapping
   raw scores → `(p_snia, p_nonIa, p_other)` *or* a clipped SN-filter
   (cap `p_snia ≤ 0.5`).
7. **Trust-target choice** — encoded in
   `feedback_trust_target_capability.md`: SN-filter projections
   (cap `p_snia=0.5`) **MUST** use `target=is_sn`; full Ia-classifier
   projections use `target=is_topclass_correct`. Mis-pick this and
   you'll get inflated 0.99 AUCs that invert on rare-class spec truth
   (the original v5c → v5d incident).
8. **Preflight assertion** — per `feedback_preflight_experiments.md`,
   before any SCC qsub: small-sample script
   `scripts/_exp_preflight_<broker>.py` that runs the **real**
   `build_object_epoch_snapshots` + `train_expert_trust` code paths on
   N=200 labels and asserts (a) coverage > some floor, (b) projection
   output is in `[0,1]` and respects the `p_snia` cap, (c) trust-head
   training rows are non-degenerate.

After preflight passes, the rebuild reuses the v6e.2 recipe:

```bash
$PY scripts/backfill.py --broker <new> --parallel 8
$PY scripts/normalize.py
$PY scripts/build_object_epoch_snapshots.py --shard-id $SGE_TASK_ID --n-shards 8
$PY scripts/build_expert_helpfulness.py
$PY scripts/train_expert_trust.py --experts <new>   # train just the new head; reuse v6e.2 pickles for the rest
$PY scripts/score_dp1_v5d.py --snapshots data/gold/dp1_snapshots_50k.parquet \
    --trust-dir models/trust_safe_v7 --followup-dir models/followup_safe_v7 \
    --out-dir reports/v7_dp1_50k
```

**Critical operational rule** (lesson from v6e.2): never run multiple
`local_infer.py` concurrently against the same silver dir. Either shard
with per-shard silver dirs or run sequentially. The silver-race
incident in `project_v6e2_trust_retrain.md` cost a day.

---

## Tier 1 — Direct paper-limitation fixes (do these first)

### A. POI Broker periodicity → followup-head **feature**, not a trust head

**Why first.** The paper's most-cited operational limitation today is
EclBin+RRLyrae `p_follow` EF 17.88. The discussion section already
states "mitigation needs an explicit period-folding feature". POI
provides exactly that. Wire it as a feature, retrain only the followup
head, write one paragraph in the paper.

**Non-obvious choice.** POI outputs a periodicity score / period
quality, not an SN classification. Forcing it through the ternary
projector wastes signal. Instead:

- Add `periodicity_score` (and possibly `period_days`,
  `periodicity_quality`) to `FEATURE_NAMES` in
  `src/debass_meta/features/lightcurve.py`.
- Add the same names to `DEFAULT_FEATURES` in
  `src/debass_meta/models/early_meta.py`.
- Retrain **only** the followup head. No new trust head. Cheap.

**Files to add:**

- `src/debass_meta/access/poi.py` — query POI by `diaObjectId`.
- `scripts/_exp_poi_probe.py` — confirm POI indexes our DP1 41K pool.

**Open question (gates everything).** POI is downstream; need to
confirm it indexes DP1 sources, not just live alerts. Probe before
committing to the rest.

**Expected paper move.** New row in
`reports/v6e2_dp1_50k/enrichment_per_class.csv`: EclBin+RRLyrae
`p_follow` EF should drop from 17.88 to ideally < 5. If it doesn't, we
report null result and pivot to Tier 1B.

### B. Fink SLSN + Fink TDE projectors (cheapest possible additions)

**Why.** We already query Fink. No new adapter, no new bronze branch
— two new projectors and two registry entries.

**Files:**

- `src/debass_meta/projectors/fink_slsn.py` — SLSN-RF score
  → `(p_snia=0, p_nonIa=score, p_other=1-score)`. Trust target
  = **`is_sn`** (SN-filter; cap p_snia=0).
- `src/debass_meta/projectors/fink_tde.py` — TDE score
  → `(p_snia=0, p_nonIa=0, p_other=score)`. Trust target = **`is_sn`**,
  polarity-inverted in the truth join.
- Two `EXPERT_REGISTRY` entries: `fink/slsn`, `fink/tde`. Both
  `exact_alert` / LSST (also ZTF if Fink ZTF exposes them).
- Update `scripts/normalize.py` to extract the two probability columns
  from the Fink payload we already cache.

**Risk.** ELAsTiCC contamination on Fink LSST SLSN-RF (trained on
ELAsTiCC). Reusing `is_sn` as the trust target is robust to the
inflated-AUC trap (`feedback_trust_target_capability.md`).

**Expected paper move.** Heads 12 + 13 in the trust table; modest
improvement on non-Ia non-periodic transients (TDE family, SLSN
nuclear).

---

## Tier 2 — New full-stream brokers (medium effort, decent ROI)

### C. AMPEL — wire SNGuess + FinalBet, **skip FollowMe**

AMPEL ships three workflows; not all three are worth the slot.

- **SNGuess** — XGBoost binary, "is this a young SN?". Trust target =
  **`is_sn`**, cap `p_snia` at 0.5. Cheap. ELAsTiCC-clean enough
  (trained on real photometry per arXiv:2501.16511).
- **FinalBet** — DL with host-galaxy + redshift priors, > 80 %
  accuracy on extragalactic transients. Trust target =
  **`is_topclass_correct`**. The host-galaxy prior is genuinely
  complementary to our 11 heads: only `lasair/sherlock` brings host
  context today, and Sherlock is `static_safe` not per-epoch.
- **FollowMe** — **skip**. Sits between SNGuess and FinalBet without
  the priors that make FinalBet distinctive. Adding it bloats the
  trust table without clear marginal signal.

**Files:**

- `src/debass_meta/access/ampel.py` — AMPEL exposes results via REST;
  archive token + ZTF-style alert query is the standard pattern (see
  `ampelproject.github.io`).
- `src/debass_meta/projectors/ampel_snguess.py` — SN-filter projection.
- `src/debass_meta/projectors/ampel_finalbet.py` — full ternary; map
  AMPEL leaf classes → (Ia / non-Ia SN / other) using a class-mapping
  dict at the top of the file. Their taxonomy is similar enough to
  ALeRCE's that the mapping is mechanical.
- Two `EXPERT_REGISTRY` entries, both `exact_alert` / any-survey.

**ELAsTiCC contamination.** FinalBet is a DL classifier — same risk
class as `fink_lsst/snn`, `fink_lsst/cats`. Already-known limitation;
report honestly in the paper.

**Open questions (resolve before building):**

- Does AMPEL provide DP1 backfill, or only live alerts? Their archive
  query supports historical lookups but DP1 indexing isn't documented
  — probe needed.
- Token / quota constraints on the REST endpoint.

### D. Babamul / AppleCiDEr — feature additions on followup head, NOT a trust head

**Verified 2026-04-26** (`scripts/_exp_babamul_probe.py`,
`memory/project_babamul_verification.md`). All architecture decisions
below are empirical, not speculative.

**Coverage**:
- **DP1: 0/50 hits** — Babamul went live Feb 2026; DP1 predates it. NOT a
  contributor to the v6e.2 DP1 paper run.
- **ZTF retro: 50/50 bulk-XM hits across 2017–2023** — train on existing
  1,920-object ZTF cohort, no waiting on live-LSST accumulation.

**Wire format**: Avro Object Container Files (`pip install fastavro`).
Two record types: `BabamulLsstAlert` and `BabamulZtfAlert`. 12 Kafka
topics: `babamul.{lsst|ztf}.{ztf|lsst}-match | no-{ztf|lsst}-match.{stellar|hosted|hostless}`
(LSST also has `unknown`).

**AppleCiDEr is gating, not a score.** Recursive walk for
`applecider|score|prob|p_snia|class_prob|pred` returned only
`candidate.sgscore{1,2,3}` (standard ZTF/PS1 star-galaxy, not Babamul).
The classification IS the topic name. **Wire as a 6-state categorical
feature on the followup head**, not a trust head:
- `applecider_class` ∈ `{hosted, hostless, stellar}`  (LSST also `unknown`)
- `cross_survey_match` ∈ `{matched, unmatched}`

**What this means for the trust-head pipeline**: nothing. Skip
`train_expert_trust.py` for Babamul entirely. Skip `EXPERT_REGISTRY`
entirely (no projector, no temporal_mode, no `is_sn` vs.
`is_topclass_correct` decision). Just FEATURE_NAMES additions +
followup-head retrain.

**Verified per-record schema** (one nesting level, per probe run):

```
BabamulLsstAlert / BabamulZtfAlert
├── candid (int)
├── objectId (str)
├── candidate (dict)              ← std ZTF/PS1 alert, incl. sgscore1/2/3
├── prv_candidates / prv_nondetections / fp_hists
├── properties (dict, 6 keys)
│   ├── rock, stationary, star, near_brightstar  ← 4 booleans
│   ├── multisurvey_photstats {u,g,r,i,z,y}      ← per-band stats both surveys
│   └── photstats {u,g,r,i,z,y}                  ← per-band, alert's own survey
└── survey_matches (dict)
    ├── lsst {ra,dec,objectId,prv_candidates,prv_nondetections,fp_hists} | null
    └── ztf  same | null
```

**Concrete files to add:**

- `src/debass_meta/access/babamul.py` — Kafka consumer (SASL_PLAINTEXT +
  SCRAM-SHA-512, port 9093) + REST API (`Authorization: Bearer bbml_...`).
  Mirrors `access/fink_lsst.py` shape. Two write paths: live
  Kafka-to-bronze daemon, and historical bulk REST queries
  (`/surveys/ZTF/objects/cross-matches`, batch ≤100).
- `src/debass_meta/features/lightcurve.py` → 6 new entries in
  `FEATURE_NAMES`:
  ```
  babamul_class_hosted, babamul_class_hostless, babamul_class_stellar,
  babamul_cross_match,           # 1 if matched, 0 unmatched
  babamul_star_flag,             # properties.star
  babamul_near_brightstar_flag,  # properties.near_brightstar
  ```
  (`rock` and `stationary` are skipped — already filtered upstream by our
  asteroid + reliability cuts.) Add the same 6 names to `DEFAULT_FEATURES`
  in `src/debass_meta/models/early_meta.py`.
- `scripts/normalize.py` — branch on Avro record type; emit one row per
  `objectId` into `silver/broker_events.parquet` with the 6 features.
- **No projector. No `EXPERT_REGISTRY` entry. No new trust head.**

**Bonus side-feature**: `survey_matches.{lsst,ztf}` carries the **full
embedded counterpart alert** when matched. This is broker-level
dual-survey lightcurve fusion — replaces the manual 2-arcsec association
table in CLAUDE.md. Wire as a follow-up phase: when present, feed both
photometry sets into `features/lightcurve.py` and let the existing
`survey_is_lsst` flag mark which is which.

**Phase budget** (revised after verification):

| Phase | Old | New | Why |
|---|---|---|---|
| 0 — schema discovery | 0.5 d | done | this probe |
| 1 — adapter + bronze | 2 d | 1.5 d | Avro deser via fastavro |
| 2 — eval cohort | 1 d | 0.5 d | reuse ZTF labels (50/50 verified) |
| 3 — projector | 1.5 d | **0** | none — categorical feature only |
| 4 — training | 1 d | 0.5 d | followup-head retrain; no trust head |
| 5 — paper paragraph | 0.5 d | 0.5 d | unchanged |
| **Total** | **~6.5 d** | **~3 d** | trust-head pipeline drops out |

**Open questions before Phase 1**:

1. `multisurvey_photstats[band]` sub-schema (mag_min/max/mean/std/n_det?).
   Decides whether Babamul photstats replace or augment our 51 LC features.
2. `survey_matches[survey].prv_candidates` schema — match
   `features/detection.py` ingest, or write a Babamul-specific
   normalizer?
3. REST RPS / quota for the 1,920 ZTF bulk pull (not documented).
4. HTTP 404 on `/surveys/ZTF/objects/{id}` for IDs that DO appear in
   bulk-XM — endpoint shape mismatch or real coverage gap? Open one
   issue at `github.com/boom-astro/babamul/issues`.

---

## Tier 3 — Hold

- **ALeRCE LSST-ATAT.** ZTF-side ATAT projector already exists
  (`alerce/LC_classifier_ATAT_forced_phot(beta)`). When ALeRCE flips
  on the LSST-side endpoint, the work is one new registry entry plus a
  one-line `survey="LSST"` switch. Nothing to build now.
- **ORACLE-as-ANTARES-filter.** We parked ORACLE locally in v6e.3
  because of domain mismatch (`project_v6e3_oracle_decision.md`). If
  ANTARES exposes ORACLE on real LSST alerts, the domain mismatch may
  close. Re-evaluate after the **Rubin Broker Summit June 2026**.

---

## Recommended first step (single, self-contained)

Start with **Tier 1A (POI as a feature)**:

1. Today: write `scripts/_exp_poi_probe.py`. One question to answer:
   does POI return non-empty results for our DP1 41K pool?
2. If yes: 2–3 days of work to add `access/poi.py`, two new
   `FEATURE_NAMES` entries, one followup-head retrain (no trust-head
   training, no full gold rebuild — feature-only changes are cheap).
3. New row in `reports/v6e2_dp1_50k/enrichment_per_class.csv` for
   periodic-variable classes; new paragraph in
   `paper/metaDEBASS_aas/sections/results.tex` and
   `discussion.tex`.
4. If POI doesn't index DP1: pivot to **Tier 1B (Fink SLSN/TDE
   projectors)** — smallest fallback, no API risk because we already
   query Fink.

**Do not start AMPEL or Babamul before A + B are merged.** Each
full-stream broker is a multi-week effort; the v6e.2 silver-race
incident shows how easy it is to lose a day to infra.

## Pass criteria per tier

| Tier | Quantitative pass criterion |
|---|---|
| 1A POI | EclBin+RRLyrae `p_follow` EF drops to < 10 (paper-quality move); ideally < 5 |
| 1B Fink SLSN/TDE | Two new heads both reach trust AUC ≥ 0.7 on test split; followup AUC unchanged ± 0.005 |
| 2C AMPEL | At least one of (SNGuess, FinalBet) reaches trust AUC ≥ 0.8; DP1 broker-coverage on DP1 snapshots improves measurably |
| 2D Babamul | At least one of the 6 Babamul features (`babamul_class_*`, `babamul_cross_match`, `babamul_star_flag`, `babamul_near_brightstar_flag`) shows feature-importance ≥ 1 % in followup head; OR EclBin+RRLyrae `p_follow` EF drops below 17.88 (the v6e.2 paper limitation) |

## Files NOT to touch without user say-so

- `models/trust_safe_v6e2/` — frozen baseline.
- `models/followup_safe_v6e2/` — frozen baseline.
- `data/gold/object_epoch_snapshots_safe_v6e2.parquet` — frozen
  baseline. Any Tier 1B / 2 / 3 retrain writes to a `_v7` (or `_v7a`,
  `_v7b`) suffix.

## Sources for the broker survey behind this plan

- [Alerts and brokers — Rubin Observatory](https://rubinobservatory.org/for-scientists/data-products/alerts-and-brokers)
- [Rubin Broker Wiki (errai34)](https://errai34.github.io/rubin-broker-wiki/)
- [Babamul Goes Live](https://www.ztf.caltech.edu/new/babamul-goes-live.html)
- [BOOM and Babamul (arXiv:2511.00164)](https://arxiv.org/html/2511.00164)
- [Transient classifiers for Fink (A&A 2024)](https://www.aanda.org/articles/aa/full_html/2024/12/aa50370-24/aa50370-24.html)
- [ATAT (arXiv:2405.03078)](https://arxiv.org/html/2405.03078v2)
- [AMPEL workflows for LSST (arXiv:2501.16511)](https://www.aanda.org/articles/aa/full_html/2025/06/aa52481-24/aa52481-24.html)
- [ORACLE (arXiv:2501.01496)](https://arxiv.org/html/2501.01496v1)
- [Photometric TDE classifier for Rubin LSST (A&A 2025)](https://www.aanda.org/articles/aa/full_html/2025/11/aa56839-25/aa56839-25.html)
- [Roadmap for Community Alert Filters with ANTARES (RTN-090)](https://rtn-090.lsst.io/)
- [Rubin Broker Summit, 8–19 June 2026](https://indico.ijclab.in2p3.fr/event/12252/)
