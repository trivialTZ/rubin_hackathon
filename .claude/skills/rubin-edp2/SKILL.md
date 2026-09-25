---
name: rubin-edp2
description: How to authenticate to the Rubin Science Platform and query EDP2/DP2 (TAP catalogs, ADQL, image cutouts). Use whenever a task touches data.lsst.cloud, RSP_TOKEN, dp2/dp1 tables, DiaObject/DiaSource lightcurves, ivoa.ObsCore, or LSST cutouts.
---

# Rubin Science Platform — EDP2 access

Everything here was verified live against `https://data.lsst.cloud` on **2026-08-29**.
Numbers marked *(live)* came back from a real query that day.

## 1. The key

`RSP_TOKEN` lives in **`/Users/tz/Documents/GitHub/rubin_hackathon/.env`** (line
`RSP_TOKEN=gt-...`). That file is gitignored (`.gitignore:33`) and holds every other
credential too — TNS, Lasair, GCP, Babamul.

- **Never print, echo, log, or paste the token value.** Not into a report, not into a
  commit, not into a comment, not into a message. Read it through the loader below and
  pass it straight to the request.
- Format: Gafaelfawr bearer token, `gt-` prefix, 48 chars.
- Identity as of 2026-08-29 *(live)*: user `tztang`, token name `debass_cross`, created
  2026-07-28, **no expiry set**. Scopes: `read:tap`, `read:image`, `exec:notebook`,
  `exec:portal`, `write:files`, `user:token`.

### Load it

```python
import sys; sys.path.insert(0, "src")
from debass_meta.access.rubin_rsp import load_rsp_token, RSPClient
token = load_rsp_token()          # $RSP_TOKEN first, then .env at repo root
client = RSPClient()              # same lookup, builds an authed pyvo TAPService
```

`load_rsp_token()` resolves the repo root from `$DEBASS_ROOT`, else from the module
path. Source: `src/debass_meta/access/rubin_rsp.py`.

### Check it is alive before blaming anything else

```bash
python3 .claude/skills/rubin-edp2/check_rsp_access.py
```

or inline:

```python
import urllib.request, json
r = urllib.request.Request("https://data.lsst.cloud/auth/api/v1/token-info",
                           headers={"Authorization": f"Bearer {token}"})
print(json.load(urllib.request.urlopen(r, timeout=30)))   # do not print `token`
```

**A 401 on any RSP call means the token was revoked or replaced, not that the service is
down.** Fix: the user mints a new one at <https://data.lsst.cloud/auth/tokens> (New
Token → scopes `read:tap` + `read:image`) and replaces the `RSP_TOKEN=` line in `.env`.
Only the user can do this — it requires their federated login. Do not attempt to
authenticate as them.

## 2. TAP — the catalog API

Endpoint `https://data.lsst.cloud/api/tap`, ADQL over HTTP, bearer auth.

```python
df = client.query("SELECT TOP 5 diaObjectId, ra, dec, nDiaSources FROM dp2.DiaObject")
```

`RSPClient.query()` returns a pandas DataFrame and **preserves int64** — the corruption
trap in §5 is downstream of it, not in it. Handy wrappers already written:
`count()`, `pool_diaobjects()`, `fetch_diasources()` (batched `IN (...)`, 500 ids/batch),
`cone_search_diaobject()`. They default to `dp1.*` — pass `dp2.*` explicitly or use
`query()` directly.

### What's actually there

Discover, don't assume: `SELECT schema_name FROM tap_schema.schemas`, then
`SELECT schema_name, table_name FROM tap_schema.tables`. As of 2026-08-29 the visible
schemas include `dp2`, `dp1`, `dp02_dc2_catalogs` (simulated DC2), `ivoa`, `tap_schema`.

**"EDP2" is the release name; the TAP schema is spelled `dp2`.** There is no `edp2`
schema. Scripts that probe should try `dp2` first and fall back
(`scripts/_exp_rsp_debass38_pull.py:SCHEMA_PREFERENCE`).

| Table | Rows *(live)* |
|---|---|
| `dp2.DiaObject` | 232,004,216 |
| `dp2.DiaSource` | 1,000,825,975 |
| `dp2.ForcedSourceOnDiaObject` | 24,259,599,059 |

Also present: `dp2.Object`, `Source`, `ForcedSource`, `CoaddPatches`, `Visit`,
`VisitDetector`, `SSObject`, `SSSource`, `mpc_orbits`, `ShearObject`,
`IsolatedStarStellarMotions`.

**Time coverage** *(live)*: `dp2.DiaSource.midpointMjdTai` spans **60790.117 → 61047.155**
(2025-05-05 → 2026-01-17). Anything outside that window has no DP2 DIA photometry — check
this before concluding an object is missing.

### Columns that matter (`dp2.DiaSource`, 90 total)

| Column | Type | Unit |
|---|---|---|
| `diaSourceId`, `diaObjectId`, `visit` | long | |
| `detector` | short | |
| `midpointMjdTai` | double | d |
| `ra`, `dec` | double | deg |
| `band` | char | one of `u g r i z y` |
| `psfFlux`, `psfFluxErr` | float | nJy |
| `scienceFlux`, `scienceFluxErr` | float | nJy |
| `snr`, `reliability` | float | |

- It is **`dec`, not `decl`**.
- `psfFlux` is a *difference* flux in nanojansky (AB zeropoint 31.4) and **can be
  negative**. Never blindly `log10` it — see §5.
- ADQL dialect limits: **no `CAST`, no arithmetic inside `GROUP BY`.**
- Cone search: `CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', <ra>, <dec>, <r_deg>)) = 1`.
- Single queries above roughly 60K rows time out. Batch instead.

## 3. Images — SODA cutouts

DP2 publishes **only deep coadds**. Confirmed *(live)*:
`ivoa.ObsCore WHERE obs_collection='LSST.DP2'` → one subtype, `lsst.deep_coadd`,
925,460 rows. **There are no per-epoch visit images or difference images in DP2** (DP1
has raw/visit_image/difference_image/template_coadd). So an LSST image panel next to a
lightcurve is a static reference, not a time series.

Resolve the datalink ID per (position, band), then cut:

```python
o = client.query(
    "SELECT obs_id, lsst_band, access_url FROM ivoa.ObsCore "
    "WHERE obs_collection='LSST.DP2' AND dataproduct_subtype='lsst.deep_coadd' "
    f"AND CONTAINS(POINT('ICRS',{ra},{dec}), s_region)=1")
# pull the ID= param out of access_url, then:
# https://data.lsst.cloud/api/cutout/sync?ID=<urlencoded ivo uri>&POS=CIRCLE <ra> <dec> <r_deg>
```

Returns FITS. Working end-to-end implementation, including zscale→PNG rendering:
`scripts/_exp_fetch_lsst_cutouts.py`.

**Rate limit: 35 requests/minute.** Use ≤2 workers and honour `429`. `Retry-After` comes
back as *either* delta-seconds *or* an HTTP-date — `float(hdr)` raises `ValueError` on
`'Tue, 28 Jul 2026 21:03:45 GMT'`. Use `retry_after_seconds()` from that script.
Overlapping coadd patches yield duplicate (object, band) rows; dedupe or they overwrite
each other's output files.

## 4. Working code to copy from

| Path | What it does |
|---|---|
| `src/debass_meta/access/rubin_rsp.py` | `RSPClient`, `load_rsp_token`, batched pulls |
| `scripts/_exp_rsp_debass38_pull.py` | schema autodiscovery, cone match, DiaSource pull, integrity check |
| `scripts/_exp_fetch_lsst_cutouts.py` | ObsCore → SODA cutout → PNG, with backoff |
| `scripts/build_debass_lsst_viewer.py` | joins DP2 to DECam SNANA photometry |
| `docs/handoff_rsp_real_data_access.md` | original access-verification plan (2026-04-24) |

## 5. Traps that have actually bitten

**`diaObjectId` is ~1e17 and does not survive float64.** 2^53 ≈ 9e15, so any float
round-trip silently mangles the low digits — you get a *different, real* object and no
error. `hits.iloc[0]["diaObjectId"]` collapses the row to a float64 Series; so does
`pd.to_numeric`, `df.iterrows()`, `pd.DataFrame(list_of_dicts)` on an int-or-None column,
and writing to CSV (it lands as `7.579403004286077e+17`). **Always extract column-first
and keep `Int64`:**

```python
oid = int(hits["diaObjectId"].iloc[0])                       # correct
ids = pd.array([int(s) for s in raw], dtype="Int64")         # correct
```

Verify by comparing the number of rows pulled against `DiaObject.nDiaSources`; a mismatch
means you queried the wrong id. Full write-up in memory `diaobjectid-float64-trap`.

**Alert-stream `diaObjectId` ≠ catalog `diaObjectId`.** IDs harvested from the live alert
stream are absent from both `dp1` and `dp2`. Match **positionally** (cone search, ~2″) and
treat any declared id as a cross-check only.

**`reliability` is the dominant quality term and it is low almost everywhere.** In one
534-row DP2 sample, 431 rows scored < 0.5 (median 3.8e-6). Cutting at `reliability > 0.9`
moved DECam↔LSST agreement from ~2.6 mag of disagreement to 0.38 mag. **Cut on
`reliability` before comparing DP2 fluxes to anything.**

**Sign convention when comparing to another survey**: Δm = m_LSST − m_DECam =
**−**2.5·log₁₀(F_LSST / F_DECam). Positive Δm means LSST measured it *fainter*. Getting
this backwards inverts the conclusion. Also note that requiring positive flux on both
sides to form a magnitude is a **one-sided cut biased toward agreement** — report the flux
ratio over all pairs alongside it, and state both sample sizes.

## 6. Data handling

- DP2 is **DESC/collaboration-restricted until its public date**. Do not commit pixels,
  fluxes, or derived gold tables to git, and do not publish them to any external service
  (Artifacts included) without the user explicitly saying so.
- `data/` and `reports/` are gitignored — write pulled data there.
- Never commit `.env` or any token.
