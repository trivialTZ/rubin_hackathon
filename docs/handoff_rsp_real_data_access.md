# Handoff: Verify Rubin RSP access + plan real-data ingest for metaDEBASS

**Authored**: 2026-04-24
**Status**: pending user verification on RSP
**Phase**: post-v5d. metaDEBASS's biggest open constraint is 12 LSST spec-typed
objects. User is a DESC member, expects RSP access, asked for a checking
MD that survives context compaction.

This doc is the **complete resumption spec** for the next session. It is
designed to be readable cold with no prior conversation context.

---

## 1. Why this matters (one paragraph)

metaDEBASS v5d (2026-04-24, job 4603140) is our honest baseline — 11 trust heads,
followup cal AUC 0.9694 overall, **0.60 on LSST-spec (CI [0.28, 0.90], n=12
objects)**. The 12-object LSST spec limitation makes architectural variants
(A/B/C/D) statistically indistinguishable and blocks paper-grade claims about
LSST generalization. Earlier we scoped DESC-TOM access (ELAsTiCC2 TEST) as a
fix but reconsidered — it's **simulated** data with possible benchmarking
contamination across brokers.

**The real unlock is real Rubin data via RSP.** User stated they are a DESC
member and should have RSP access. This doc's job: verify that hypothesis
with concrete checks, then scope the ingest path into the existing metaDEBASS
pipeline.

---

## 2. Post-compaction resumption state (read this first after /compact)

- **Current best run**: v5d. Artifacts on SCC at `/project/pi-brout/rubin_hackathon/`:
  - `data/gold/object_epoch_snapshots_trust_safe_v5d.parquet`
  - `models/trust_safe_v5d/` (11 heads)
  - `models/followup_safe_v5d/`
  - `reports/metrics/{expert_trust_metrics,followup_metrics}_safe_v5d.json`
  - `reports/trust_head_slices_v5d.csv`
- **Execplan backing v5d**: `docs/execplan_v5d_sn_filter.md`.
- **Memory files updated** (2026-04-24): `project_final_paper_phase.md`,
  `project_lsst_truth_gap.md`, new `feedback_trust_target_capability.md`.
- **This doc's purpose**: verify RSP access and produce a concrete scope
  document for the real-data ingest. We are NOT writing ingest code yet —
  first step is proving access and inventorying what's available.
- **User is**: DESC member, lead on metaDEBASS, has SCC access (`ssh scc` alias),
  has TNS API key configured locally in `.env`. User is not a Rubin insider
  beyond DESC collaboration level — access through DESC paths, not ops paths.

---

## 3. RSP access verification — concrete checks

### 3.1 Which RSP instance to try (check in order)

| Site | URL | Access basis |
|---|---|---|
| US Data Facility (USDF) | https://data.lsst.cloud | Rubin / DESC / LSSTC — most common for DESC members |
| Interim Data Facility (IDF) | https://data-int.lsst.cloud | LSSTC early users |
| French DAC (IN2P3 CC) | https://data-fr.lsst.eu | LSST:FR collaborators |
| UK DAC (DiRAC) | Available through UK DAC portal | LSST:UK collaborators |

For a US-based DESC member, **start with USDF** (https://data.lsst.cloud).

### 3.2 Login check

- [ ] Navigate to RSP URL above
- [ ] Log in with ORCID, GitHub, or institutional federated identity
  (user's BU / `tztang@bu.edu` federated ID may work via InCommon)
- [ ] After successful login, the portal should show tile launchers:
  **Notebooks**, **Portal**, **API Aspect**
- [ ] If login fails with "no group membership" or "access denied":
  - [ ] Check DESC membership: https://confluence.slac.stanford.edu/display/LSSTDESC/Home
  - [ ] Ask DESC membership chair or the DESC-SN working group for RSP-group
        enrollment. Typical turnaround ≤ 1 week.
  - [ ] Alternative: IDF may accept LSSTC users that USDF doesn't. Try IDF second.

Record in this doc under **§8 Outcome**:
  - Which RSP instance granted access.
  - What username/group you appear as.

### 3.3 Data release inventory

Open **Notebooks → JupyterLab** aspect. In a new notebook, run:

```python
from lsst.daf.butler import Butler
# List all registered Butler repos visible to your account
import lsst.daf.butler as db
print("Butler version:", db.__version__)
# The RSP notebook image preconfigures DP0.2; DP1 requires a different config
repo = "dp02"        # DP0.2 Butler alias
butler = Butler(repo, collections=["2.2i/runs/DP0.2"])
print("collections:", list(butler.registry.queryCollections())[:5])
print("dataset types:", list(butler.registry.queryDatasetTypes(components=False))[:20])
```

Checks to perform in the JupyterLab aspect:

- [ ] `Butler("dp02", ...)` — **DP0.2 (simulated, DESC DC2)** — default for every DESC member
- [ ] `Butler("dp1", ...)` or equivalent alias — **DP1 (real commissioning data)** — verify if released to DESC collaborators; note the actual repo alias
- [ ] Query any available live Butler for **ComCam / LATISS** datasets (real on-sky data from Rubin auxiliary telescope during commissioning)
- [ ] Check `embargoed` or `embargo` repos (proprietary real-data, time-gated release)

If Butler queries fail, try the **TAP service** — lower friction than Butler:

```python
from lsst.rsp import get_tap_service, retrieve_query
service = get_tap_service("tap")
# List available catalog schemas
result = service.search("SELECT TOP 5 * FROM tap_schema.schemas")
print(result.to_table())
# Sanity: object count from DP0.2 Object catalog
result = service.search("SELECT COUNT(*) FROM dp02_dc2_catalogs.Object LIMIT 1")
print(result.to_table())
```

- [ ] TAP schema list — record names (e.g. `dp02_dc2_catalogs`, `dp1`, `tap_schema`, etc.)
- [ ] Object catalog row count per release available

### 3.4 Alert-stream / DIA access check

metaDEBASS runs on alert-level data (per-epoch snapshots from brokers). We need
to know whether alerts from Rubin have actually been streamed to external
brokers (Fink / ALeRCE / Pitt-Google).

Check inside the RSP (if available) or externally:

- [ ] Does `dp02` / `dp1` include `diaObject` / `diaSource` tables? TAP query:
  ```sql
  SELECT TOP 5 diaObjectId, ra, decl, nDiaSources
  FROM <release>.DiaObject
  WHERE nDiaSources > 5
  ```
- [ ] Do **Fink LSST**, **ALeRCE Rubin**, **Pitt-Google LSST** have DP1 alerts indexed?
  Quick external checks (no login needed):
    - Fink: https://api.lsst.fink-portal.org/api/v1/objects (try `objectId=<diaObjectId from TAP>`)
    - ALeRCE LSST: https://api.alerce.online (stamp + probability endpoints)
    - Pitt-Google LSST: BigQuery `ardent-cycling-243415.pittgoogle_broker_lsst.*`
  Pick 3–5 `diaObjectId`s from the TAP query above and try backfilling each
  broker. Record how many return a real score vs "not found".

If brokers are silent on DP1 alerts, that's a known Rubin-ops issue. Fallback
plan: **run Fink/ALeRCE/Pitt-Google classifiers locally on our own ingest**,
or extract broker probabilities from the RSP-hosted `diaObject` rows if they
store `classifierScore` (some releases do).

### 3.5 Spec-truth overlap check

For paper claims, we need **spec-typed** real Rubin objects. Source options:

- [ ] TNS crossmatch against Rubin `diaObject` positions (2 arcsec). TNS has
      our API key already; write a one-shot script that joins DP1 `diaObject`
      positions against TNS typing.
- [ ] Fink's `xm_tns_type` / `xm_tns_fullname` already provided on LSST alerts
      (if the broker indexed DP1 — check §3.4).
- [ ] DESC SN working group may have curated DP1 → TNS crossmatch tables
      shared internally. Ask `#desc-sn` Slack.
- [ ] YSE / BTS / DES-SN legacy spec surveys — crossmatch their catalogs
      against DP1 positions for archival spec truth.

---

## 4. What to collect during the check (minimum deliverable)

Produce a short **access report** (plain markdown, under 1 page) with:

1. **Which RSP instance** you got into (USDF / IDF / other).
2. **Which Butler repos / TAP schemas** are visible.
3. **Approximate object counts** per release:
   - DP0.2 Object catalog: expected ~100M (simulated)
   - DP1 Object catalog: **record exact N**
   - DP1 diaObject count: record N
   - DP1 diaSource count: record N
4. **Time coverage** — DP1 obs-nights from / to (mjdMin, mjdMax).
5. **Broker coverage on DP1** — for 10 random diaObjectIds, how many have
   scores in Fink LSST / ALeRCE LSST / Pitt-Google LSST?
6. **Spec-truth overlap** — for the same 10 objects, how many have TNS types?
7. **Data export policy** — can raw flux be pulled off RSP and stored in our
   repo? Check DESC / Rubin data rights doc. If no, scope an **RSP-side pipeline**
   where we run metaDEBASS training inside RSP notebooks and only export models +
   aggregate metrics.

---

## 5. Scoping the real-data ingest into metaDEBASS (only after access confirmed)

These are the metaDEBASS code paths that will change once we know we have access.
Do not start these until §3 is green.

### 5.1 File-map impact

| File | Change |
|---|---|
| `data/labels.csv` | Append real DP1 `diaObjectId` rows with TNS types / discovery_broker='rubin' |
| `src/debass_meta/access/` | New `rubin_rsp.py` adapter — Butler / TAP queries for lightcurves |
| `scripts/fetch_lightcurves.py` | Extend to call RSP adapter when label has `survey=LSST` + `source=rubin` |
| `scripts/backfill.py` | Same — include DP1-aware broker queries |
| `scripts/build_truth_multisource.py` | Add Rubin-native truth tier (RSP `classificationLabels` if present) |
| `src/debass_meta/features/detection.py` | Verify schema parity with Rubin DIA `diaSource` columns |
| `CLAUDE.md` | Document new access path + credentials |
| `.env` | Add RSP token / credential file pointer |

### 5.2 Credential + data-rights plumbing

- RSP access is **interactive browser** — no long-lived machine token in most
  configurations. Options:
    1. Run metaDEBASS ingest **inside RSP JupyterLab** (no local pull). Suits
       Rubin policy, but forks the pipeline from SCC.
    2. Use `lsst.rsp.get_tap_service()` **token export** — short-lived token
       from the RSP portal, pasted into SCC. Check expiry / rotation rules.
    3. Use the **Butler direct HTTPS API** with OIDC token from the portal.
- Don't commit any token to git. Add to `.env` + `.gitignore` (already there).
- Respect the DESC/Rubin data-rights doc. Some releases (DP1 in particular)
  may be **DESC-only** until a public date — don't auto-publish the resulting
  gold parquet.

### 5.3 Contamination audit

metaDEBASS's ML-trained experts (`supernnova`, `fink_lsst/*`, `alerce_lc`,
`parsnip`, `oracle`) were all trained on ELAsTiCC / ELAsTiCC2 / earlier Rubin
simulations. Real Rubin DP1 data is **genuinely held out** from these training
sets by construction (DP1 was not available when those models were trained).
So contamination is NOT a blocker for DP1-based evaluation.

Exception: any ML model that was **trained on DP1** (e.g. Fink may have
retrained on DP1 for their live service). Audit each broker's training-data
disclosure before citing AUC numbers in the paper.

### 5.4 Expected effort (rough, assuming access is granted)

| Step | Effort |
|---|---|
| RSP access verification (§3) | 30 min – 2 h |
| Data inventory + broker coverage check (§4) | half-day |
| `rubin_rsp.py` adapter | 1–2 days |
| Label + truth pipeline updates | 1 day |
| First real-data retrain (v6a) | half-day SCC + pipeline reruns |
| First real-data slice analysis | half-day |

**Total**: ~1–2 weeks if nothing is blocked.

---

## 6. Risks / what could go wrong

- **No DP1 alerts in external brokers** → either run brokers locally or
  extract broker scores from Rubin AP outputs stored on RSP.
- **Only DP0.2 is accessible, not DP1** → DP0.2 is DESC DC2 simulation, not
  real Rubin data. Useful for pipeline plumbing but does NOT resolve the
  "real LSST data" goal. Write the ingest to be ready, but don't confuse
  DP0.2 metrics for DP1 / real-data metrics.
- **Access requires additional DESC paperwork** → can take weeks. In the
  meantime, work on orthogonal improvements (follow-up simulation harness,
  bootstrap CIs, figures).
- **Data-rights restrictions prevent pulling raw flux off RSP** → pipeline
  must be forked to run inside RSP, models exported out.
- **Broker models retrained on DP1** → new contamination vector; treat
  per-broker training-data disclosure as a first-class audit.

---

## 7. What NOT to do

- **Don't commit** RSP tokens, raw DP1 flux, or other restricted data to git.
- **Don't publish AUC numbers on real Rubin data** before confirming each
  broker model's training data is disclosed (contamination audit).
- **Don't conflate DP0.2 metrics with DP1 / real-data metrics.** They are
  different data generating processes.
- **Don't delete v5d artifacts** on SCC. They remain our honest baseline.
- **Don't start building `rubin_rsp.py`** before §3 is green — scope is
  different depending on which access path works.

---

## 8. Outcome section — fill this in during verification

*(User fills this in live, then posts the filled doc back in the next session
so work can resume.)*

- [ ] RSP instance accessed: `__________` (e.g. USDF / IDF / other)
- [ ] JupyterLab launch successful: yes / no / partial
- [ ] TAP service reachable: yes / no
- [ ] Butler reachable: yes / no
- [ ] Releases visible:
    - DP0.2: yes / no
    - DP1: yes / no
    - LATISS: yes / no
    - ComCam: yes / no
    - Other: `________`
- [ ] DP1 diaObject count: `________`
- [ ] DP1 time coverage (MJD min/max): `________`
- [ ] Sampled 10 diaObjectIds — broker coverage (x/10):
    - Fink LSST: `____`
    - ALeRCE LSST: `____`
    - Pitt-Google LSST: `____`
- [ ] Sampled 10 diaObjectIds — TNS type overlap: `____`
- [ ] Data-rights allows local export: yes / no / partial
- [ ] Blockers / gotchas noticed: `________`

---

## 9. Next-step plan (ordered, resume cold from here)

1. User runs §3 checks on RSP, fills §8.
2. User posts filled §8 back in next session (or at least the high-level
   answer: "DP1 works / doesn't work, data export OK / restricted").
3. Claude scopes concrete v6a plan based on §8 outcome:
   - If DP1 + broker coverage + export OK → build `rubin_rsp.py` adapter,
     append ~1K real DP1 diaObjects to `labels.csv`, run backfill+truth+trust
     on SCC, produce v6a metrics.
   - If DP1 + limited broker coverage → add a metaDEBASS-side classification pass
     inside RSP (run Fink/ALeRCE/Pitt-Google API calls from RSP notebooks).
   - If only DP0.2 → build the adapter anyway (reusable) and iterate on
     pipeline correctness; don't inflate claims.
   - If no access at all → drop this path, pursue follow-up simulation
     harness (previously-identified "Candidate 1").
4. After v6a: re-do trust-head slicing, A/B/C/D ablation (now statistically
   meaningful on 10× larger LSST-spec), and paper update.

---

## 10. Memory breadcrumb (append to MEMORY.md after RSP check)

```
- [Rubin RSP access](reference_rsp_access.md) — DESC member at <URL>; DP1/DP0.2 visible; broker coverage X%; data export <policy>; next step v6a
```

Create `memory/reference_rsp_access.md` once §8 is filled. Include:
- RSP URL used, group membership.
- Token expiry / rotation rules.
- DP1 data volume + time coverage.
- Broker coverage baseline.
- Any data-rights constraints that affect repo commits.
