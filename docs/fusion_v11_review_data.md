# fusion_v11 — Data-Feasibility Review (LIVE-verified 2026-07-05 UTC)

Reviewer: data-feasibility agent — **two independent live-verification passes merged**
(both ran the checks end-to-end with the credentials in `.env`; TNS api key,
LASAIR_LSST_TOKEN; Fink LSST needs no auth). Where the passes overlap the numbers agree;
pass-2 additions are the epoch-window yield measurements in §3, the Fink per-alert
missingness measurement in §5, and the **§6 label-inventory finding (BLOCKING for Head-2
on the current truth table)**. Verdict per arm at the end of each section.

---

## 1. TNS bulk daily dump — CONFIRMED (design §1/§3.6 P1: GO)

**Endpoint (master, full catalog, regenerated daily at 00:00 UTC):**

```
POST https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip
form data:  api_key=<TNS_API_KEY>
header:     User-Agent: tns_marker{"tns_id":"<TNS_TNS_ID>","type":"user","name":"<TNS_MARKER_NAME>"}
```

- Works with our existing `type=user` marker (no bot registration needed) — exactly the
  `TNSCredentials.user_agent` already built by `src/debass_meta/access/tns.py`.
- Measured: HTTP 200, `Content-Type: application/zip`, **13.6 MB**, 8.3 s download.
  **No Content-Length header** (chunked) → HTTP Range resume is NOT reliable on the master.
- Zip member `tns_public_objects.csv`. **Line 1 is a metadata timestamp**
  (`2026-07-05 00:00:00`) — parse with `skiprows=1`; line 2 is the header.
- **200,145 rows, 23 columns** (verified):
  `objid, name_prefix, name, ra, declination, redshift, typeid, type, reporting_groupid,
  reporting_group, source_groupid, source_group, discoverydate, discoverymag,
  discmagfilter, filter, reporters, time_received, internal_names,
  Discovery_ADS_bibcode, Class_ADS_bibcodes, creationdate, lastmodified`.
  `ra/declination` are decimal degrees; `discoverydate` full timestamp
  (range 1976-05-31 → 2026-07-04); 25,914 typed rows.
- **Daily diff files exist**: `tns_public_objects_YYYYMMDD.csv.zip` (same auth, same
  columns, meta line = day range) contains ONLY rows modified that UTC day
  (2026-07-04 file: 481 rows, 40 KB). Verified available for the previous ≥2 days;
  the current (unfinished) day 404s.
- **Hourly files (`..._YYYYMMDD_HH.csv.zip`) 404 with our credentials** — do not design
  around them.
- **Rate limit**: the bulk endpoint returns `x-rate-limit-limit: 20` per 60 s window —
  irrelevant at one pull/day, but retry loops must back off (reuse `TNSClient`'s 429
  handling).

**Amendments to §3.6 `fetch_tns_bulk.py`:**
1. "Cached, resumable" → implement as: full re-download of the master (13.6 MB, seconds)
   when cache is stale/corrupt, plus incremental **daily-diff upsert keyed on `objid`**
   using `lastmodified`. Do not implement byte-range resume (no Content-Length).
2. Free bonus for P1: `internal_names` contains ZTF ids (e.g. "ZTF25acdwuzj, ATLAS25odj,
   …") — TNS↔ZTF mapping comes free with the dump; BTS is not needed for name mapping.

---

## 2. ELAsTiCC2 public corpus — CONFIRMED (design §3.4 P5 gated arm: **GO**)

**Location (public, no auth, HTTP, `Accept-Ranges: bytes` → `wget -c` resumable):**

```
https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/
  ELASTICC2_TRAIN_02.tar.bz2            7,906,500,755 bytes (7.91 GB)   [single tarball]
  ELASTICC2_TRAIN_02_<MODEL>/           36 per-class dirs, each ~40 HEAD + ~40 PHOT
                                        SNANA FITS (.FITS.gz; sample PHOT file 15.3 MB)
  A_FORMAT.TXT                          format doc (SNANA HEAD/PHOT; SNID→diaObjectId map)
  A_MODEL_SUMMARY.TXT                   per-model LC counts
  AVRO/                                 19 alert-format tar.bz2 (alternative)
```

- Per-model counts (from `A_MODEL_SUMMARY.TXT`, NLC_WRITE): SNIa-SALT3 **172,180**;
  SNII-Templates 78,306; SNII+HostXT 80,560; SNIax 40k-class; … total ≈ **1.5M
  lightcurves** with truth across 36 classes.
- Format: SNANA FITS pairs (HEAD = per-object metadata incl. SNID/ra/decl/redshift/truth
  model; PHOT = photometry rows indexed by PTROBS_MIN/MAX). Readable with plain
  `astropy.io.fits`; no SNANA install required.
- ELAsTiCC1 fallback: `TRAINING_SAMPLES/FULL_ELASTICC_TRAIN.tar` (7.91 GB) same portal.

**SCC feasibility: GO.** One resumable 7.9 GB HTTP download (or, better, **selective
per-class fetch** of the SN-like dirs only — order 3–4 GB — since the GRU pretraining
target is the Ia axis). Decompression of the full tar.bz2 needs ~30–40 GB scratch;
per-class fetch avoids that. `scripts/fetch_elasticc2.py` should take a `--models` list
and default to the SN classes.

---

## 3. BTS/ZTF-spec ∩ LSST association yield — MEASURED; **the conesearch direction in
§1/§3.6 is wrong by ~an order of magnitude; invert the harvest** (amend P1; see BLOCKING)

**Pool counts (real data, 2026-07-05):**
- `data/truth/ztf_bts.parquet`: spec objects with peak ≥ 2025-10-01 AND dec ≤ +12:
  **233** (of 10,318). **Freshness correction**: the parquet is STALE — file mtime
  2026-04-20, max content peak 2026-03-28, only 90 ZTF26\* ids. The 233 therefore covers
  ~6 of the 9 months since 2025-10; today's true pool is ~1.4–1.5× larger. Also only
  100/233 are genuinely BTS-typed (133 have `bts_type='-'` — see §6).
- TNS bulk (fresh): typed, disc ≥ 2025-10-01, dec ≤ +12: **905** (655 with ZTF internal
  names; 837 SN-typed). `object_truth.parquet` is NOT usable as the pool
  (`tns_discovery_date` populated for only 508/10,684 ZTF spec rows → just 15 pass).

**Measured hit rates against LSST (two independent brokers, two independent samples):**
- Random 25 of the 233 → Lasair LSST `/api/cone/` @2″: **0/25**.
- Same 25 → Fink LSST `/api/v1/conesearch` @2″: **0/25** (positive control at a known
  LSST object position returns the object, so the endpoints work).
- Independent pass-2 sample (seed 42, 25 of a 248-object BTS+object_truth union pool):
  **1/25 = 4%** — the one hit being ZTF26aabktja → diaObjectId 170446346054009110 at
  sep 0.14″, which is **AT 2026aok, TNS-untyped, nDiaSources=1**. Cross-validation: the
  cone hit and the name-join hit below are the same object, found by both methods.
- All 233 names → Lasair `crossmatch_tns.tns_name IN (...)`: **2/233 = 0.86%**
  (2025adas nDiaSources=2, 2026aok nDiaSources=1 — both unusable lightcurves).
- Extrapolated conesearch-direction yield over the whole recent southern spec pool:
  **~8 objects (95% CI ≈ 1–28), essentially all with <3 detections.** The design's
  "Expected yield O(50–300)" FAILS for the ZTF→LSST conesearch direction. Root cause:
  first-months LSST coverage/templates barely overlap the ZTF spec stream yet.
  Consistency check: 42 epoch-valid typed objects (below) / 879–905 recent southern
  typed TNS = ~5% capture — the two directions agree.

**The harvest that DOES work — invert the direction.** Lasair LSST already ships the
TNS↔LSST positional match as the `crossmatch_tns` table (2,824 rows total). Measured:

| slice (Lasair LSST, live counts) | n |
|---|---|
| crossmatch_tns total | 2,824 |
| typed (`type != ''`) | **247** rows = 247 diaObjectIds = **170 unique TNS names** (SN 217, SN Ia 134 rows) |
| typed AND disc_date ≥ 2025-10-01 | 92 |
| typed AND nDiaSources ≥ 3 (join objects) | 124 |
| typed AND disc ≥ 2025-10 AND nDia ≥ 3 | **54** |
| typed AND ZTF internal name (TNS-dump join: 179 rows = 72.5%) | 177 via disc_int_name (81 with nDia ≥ 3; 49 with disc ≥ 2025-10) |
| untyped AND ZTF in disc_int_name AND nDia ≥ 3 | 168 |
| **typed AND epoch-valid (the design §3.6 cut: disc ∈ [first−90d, last+30d])** | **47 rows → 42 unique TNS names** |
| epoch-valid AND SN-like | 41 names (**23 SN Ia**) |
| epoch-valid AND ZTF-associated | **34 names (20 Ia)** |

**Epoch-window measurements (pass 2 — this is the actual usable yield):** applying the
design's own staleness cut to the 247 typed rows keeps only 47 (**81% of Lasair's typed
TNS crossmatches are epoch-stale** — 2018–2025 transients positionally matched to
unrelated 2026 flux; live-data proof that design bug #4 and `build_truth_lsst_live.py`'s
epoch filter are necessary, not hypothetical). The 42 epoch-valid names were discovered
2025-11 → 2026-06 (3×2025, 39×2026); their median `nDiaSources` is 3 (p90 = 608 —
a few AGN-dense). Growth outlook: TNS typed inflow at dec ≤ +13 is ~66/30d (dump
measurement) → at ~5% LSST capture expect **~3–10 new epoch-valid spec objects/month**
now, rising as coverage/templates complete.

**Amendment to `harvest_ztf_lsst_associations.py` (P1):** do ONE paged
`tables="objects,crossmatch_tns"` query and parse ZTF ids out of
`crossmatch_tns.disc_int_name` (e.g. "ZTF25acdwuzj, ATLAS25odj, …") to emit association
CSV rows — no per-object conesearch (0.9% hit rate, rate-limit exposure). Keep the cone
endpoint only as a fallback for TNS objects lacking `disc_int_name`.

**BLOCKING-level consequence for design §3.1/§3.5/§6.** The typed pool that is
epoch-valid and has usable lightcurves is ~54–124 objects — and the frozen 2026-07-04
benchmark (~150 spec) was drawn from this same pool, while design-§6 policy routes ALL
new spec arrivals to TEST until n_test ≥ 300. Net: the **LSST spectroscopic TRAIN/CAL pool is single-digit
to low-tens for months**, not the "LSST associations + live spec-train" set §3.1 head-2
and §3.5 Stage-A trust assume. The hierarchy still stands (head-2 trains on ZTF ~4k spec;
graceful degradation is designed in), but v11 must (a) state that head-2 and LSST trust
start effectively ZTF-spec-only, (b) expect the per-survey LSST head-2 calibration to hit
its tiny-n fallback immediately, and (c) reconsider §6's test-first policy (e.g. hash-based
alternation from the start, or a lower test target) if any LSST spec-train support is
wanted this year.

---

## 4. Lasair LSST API for `build_truth_lsst_live.py` — CONFIRMED with 3 amendments

- **Conesearch**: `POST https://lasair.lsst.ac.uk/api/cone/` form
  `{ra, dec [deg], radius [arcsec], requestType: "nearest"|"all"}`, header
  `Authorization: Token $LASAIR_LSST_TOKEN` → `{"objects":[{"object":<diaObjectId>,
  "separation":<arcsec>}],"count":N,"nearest":{...}}`. Verified.
- **Epoch fields**: `objects.firstDiaSourceMjdTai`, `objects.lastDiaSourceMjdTai`,
  `objects.nDiaSources`, `objects.ra/decl` all live-verified via `/api/query/`.
  `crossmatch_tns` columns (live-probed, full list): `ra, decl, tns_name (no prefix),
  tns_prefix, disc_mag, disc_mag_filter, type, z, hostz, host_name, ext_catalogs,
  disc_int_name, disc_date, lastmodified_date, sender, reporters, source_group, htm16,
  lasairmodified_date, id`. → **TNS discovery date IS available server-side**
  (`disc_date`), so the epoch-aware cut (tns_discovery ∈ [first_det−90d, last_det+30d])
  can be pushed into the SQL `conditions` string; MjdTai↔date conversion done client-side.
- **AMENDMENT (design §1 line "crossmatch_tns via watchlist_hits wl_id=46")**: the
  `watchlist_hits` table **does not exist on Lasair LSST** (live error: `Unknown table
  'ztf.watchlist_hits'`). The working mechanism — already used by the 2026-07-04 eval —
  is the implicit positional join `tables="objects,crossmatch_tns"`. Strike wl_id=46
  from the design.
- **AMENDMENT (join radius)**: the implicit join is WIDER than 2″ — measured over all
  247 typed pairs: median 0.28″, 87% ≤ 2″, **max 2.97″**. `build_truth_lsst_live.py`
  must compute separation client-side from `objects.ra/decl` vs `crossmatch_tns.ra/decl`
  and apply the ≤2″ cut itself (keep `sep_arcsec` in the output as the design intends).
- **AMENDMENT (primary vs backup)**: `crossmatch_tns` is Lasair-refreshed
  (`lasairmodified_date` was current-day) but is footprint/match-filtered (2,824 rows) —
  use it as the *match seed*; use the TNS bulk dump (§1) as the authoritative
  type/discoverydate source so staleness demotion has one source of truth.
- **AMENDMENT (window semantics — tail matches)**: same-transient LATE matches fail the
  [−90d, +30d] window *by construction*: e.g. AT 2026aok (disc 2026-01-15) has LSST
  first det at MJD 61185 (+130 d, nDiaSources=1) — a real fading tail of the same
  transient, not a mismatch. `build_truth_lsst_live.py` should label these
  `tail_xmatch` (excluded from train AND eval, like `stale_xmatch`) instead of silently
  dropping, so the manifest records why each candidate was rejected.
- Scale context: the Lasair LSST `objects` table holds **3,247,336 diaObjects** (live
  `COUNT(*)`; the design's 12.9M figure is alerts, not objects). Full-table client-side
  crossmatch via the 1000-row-paged query API is infeasible — the `crossmatch_tns` seed
  (2,824 rows) + per-position cone fallback is the only workable pattern, and it is
  enough. ~45 authenticated API calls in one session drew no throttle; no rate-limit
  headers are exposed — chunk any bulk cone work with sleeps.

---

## 5. Fink LSST xm fields as truth-side backup — CONFIRMED

`POST https://api.lsst.fink-portal.org/api/v1/sources` json `{"diaObjectId": "..."}` (no
auth) → per-alert rows (125 rows for test object 313695087469527067 / AT 2025adlz)
carrying, per row: `f:xm_tns_fullname` ("AT 2025adlz"), `f:xm_tns_type`,
`f:xm_tns_redshift`, `f:xm_simbad_otype`, `f:xm_legacydr8_{zphot,e_zphot,pstar,fqual}`,
`f:xm_gaiadr3_{Plx,e_Plx,VarFlag,DR3Name}`, `f:xm_mangrove_{HyperLEDA_name,2MASS_name,
lum_dist,ang_dist}`, `f:xm_vsx_Type`, plus (not yet in `access/fink.py`'s harvest list)
`f:xm_gcvs_type`, `f:xm_spicy_class`, `f:xm_x3hsp_type`, `f:xm_x4lac_type`.
`/api/v1/conesearch` (json `{ra,dec,radius}`) also works.

**Caveats for truth use:** (a) sentinel string **"Fail"** appears as a value
(`xm_simbad_otype`, `xm_gaiadr3_DR3Name`) — must be mapped to missing;
(b) xm values are **as-of-alert AND intermittently missing even after classification** —
measured on SN 2026ctw (diaObjectId 170019717277810735, 649 alerts, classified
2026-02-25/MJD 61096): `xm_tns_type = "SN Ia"` on 447 alerts but the literal string
`"nan"` on **178 of the 470 post-classification alerts, including the latest alert**;
`xm_tns_fullname` also flips `AT 2026ctw` → `SN 2026ctw` over time. **Truth extraction
MUST aggregate across ALL alerts of an object** (take the max-MJD non-nan value) — a
latest-alert read silently loses types. Cleanest backup pattern: use aggregated
`xm_tns_fullname` as a *name pointer* and take type/discoverydate from the TNS bulk dump;
keep xm_simbad/gaia/legacydr8 for the host-context tier. Fine as backup/cross-check,
never primary over the TNS bulk.

---

## 6. BLOCKING label-inventory defect (HAVE side, design §1 + §3.1) — found while measuring §3

`object_truth.parquet` spectroscopic rows: 10,684 (5,512 snia / 5,172 nonIa_snlike).
**3,149 of the 5,172 nonIa_snlike spec rows (61%) are `label_source='ztf_bts'` with
`bts_type='-'`** — BTS objects with NO classification, force-mapped by the BTS ingest to
`ternary='nonIa_snlike'`, `label_quality='spectroscopic'` (verified in both
`ztf_bts.parquet` — 3,150/10,318 rows — and `object_truth.parquet` — 3,149 rows).

Live TNS-dump resolution of those 3,149 names (fresh 2026-07-05 dump, name join):
3,082 match; **101 are typed on TNS today — 39 are SN Ia** (i.e. currently mislabeled as
spectroscopic nonIa), 15 generic "SN", and **25 are not SNe at all** (19 CV, 4 Galaxy,
1 Varstar, 1 AGN). The remaining ~3,048 are genuinely untyped SN candidates.

**Why this is blocking**: design §3.1 trains Head-2 "ONLY on label_quality='spectroscopic'
SN rows". On the current truth table, **61% of Head-2's nonIa class would be unlabeled
filler** — a population that skews Ia if anything (BTS bright unclassified). This
reproduces bug-class #1 ("a weak label forced into a subtype") inside the flagship ZTF
spec corpus and poisons exactly the Ia-vs-nonIa axis v11 exists to repair. The §1
inventory line "~10.7k spectroscopic truth rows" is really **~7.5k typed + ~3.1k
SN-candidate (weak) rows**.

**Required P1 amendments (all machinery live-verified in this review):**
1. Truth building: `bts_type='-'` (or any unmapped BTS type) must never emit
   `label_quality='spectroscopic'` + a subtype ternary. Re-resolve via the TNS bulk dump
   name join (101 resolve today; more weekly via `lastmodified` upserts).
2. Unresolved rows → SN-level weak tier only (`is_sn=1`, Head-1; excluded from Head-2),
   e.g. `label_quality='bts_untyped'` (mirrors `tns_untyped`).
3. Head-2 row selection asserts a concrete subtype provenance (`tns_type` or non-'-'
   `bts_type`); add a guard: Head-2 training rows with untyped provenance == 0.
4. G1 (ZTF locked-test AUC within v10 CI): the locked 765-object test labels must be
   re-derived under the fix and v10 re-scored on cleaned labels for apples-to-apples —
   do NOT silently re-label the locked set without recording the delta.

---

## 7. Verdict summary

| Design arm | Live result | Go/No-go |
|---|---|---|
| P1 TNS bulk (`fetch_tns_bulk.py`) | Endpoint+auth+schema confirmed; 13.6 MB/8 s; daily diffs exist; hourly do not | **GO** (amend resume semantics) |
| P1 epoch-aware truth (`build_truth_lsst_live.py`) | All fields live-verified; wl_id=46 wrong → objects,crossmatch_tns join; ≤2″ cut client-side | **GO** (3 amendments) |
| P1 ZTF↔LSST association harvest | Conesearch direction ~0.9% hit → ~8 objects; inverted crossmatch_tns harvest → 177 typed / 168 untyped ZTF-named | **GO only in inverted form**; spec-TRAIN pool near-empty after benchmark freeze (see §3 blocking note) |
| P5 ELAsTiCC2 pretraining | 7.91 GB tar.bz2 + per-class FITS on open NERSC portal, range-resumable | **GO** (gated, per design) |
| Truth backup: Fink LSST xm | Confirmed incl. 4 extra xm fields; "Fail" sentinel; per-alert type intermittently missing (178/470 post-classification alerts on test object) | **GO with cross-alert aggregation rule** |
| §3.1 Head-2 on the CURRENT `object_truth.parquet` | 61% of spec nonIa class is BTS-untyped filler (39 provably Ia, 25 non-SN) | **NO-GO until §6 fix lands in P1** (fix is cheap; TNS dump resolves it) |
| ZTF truth freshness | `ztf_bts.parquet` mtime 2026-04-20, content ≤2026-03-28; `object_truth` TNS fields ≤2026-03-15 | refresh via P1 TNS dump; add BTS re-pull to `refresh_lsst_live.sh` if BTS-only fields stay in use |
