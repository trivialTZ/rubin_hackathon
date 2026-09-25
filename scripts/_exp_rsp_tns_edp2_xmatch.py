#!/usr/bin/env python3
"""Cross-match TNS objects discovered during EDP2 against dp2.DiaObject.

Generalises _exp_rsp_debass38_pull.py from the 38 DEBaSS targets to every TNS
object whose discovery date falls in (or shortly before) the DP2 DIA window.

Steps
  1. TNS objects with discovery MJD in [--mjd-start - --pre-days, --mjd-end].
  2. Footprint prefilter: keep objects within --fov-deg of any dp2.Visit
     pointing (cached to visits.csv). Records how many visits covered each
     object and how many fell in [disc-30d, disc+100d].
  3. One single-cone DiaObject query per candidate (OR'd cones defeat Qserv
     chunk pruning), parallel client-side, checkpointed to cones.jsonl so an
     interrupted run resumes.
  4. DiaSource aggregates (first/last MJD, count, max S/N) for matched IDs,
     for a first-detection-vs-TNS-discovery comparison.

Output: reports/rsp_tns_edp2/  (gitignored; DP2 is data-rights restricted)
  visits.csv          dp2.Visit pointings
  candidates.csv      TNS objects in window + footprint coverage columns
  cones.jsonl         raw per-candidate cone results (checkpoint)
  tns_edp2_match.csv  one row per candidate, IDs as decimal strings
  diasources.parquet  DiaSource rows for matched objects (real + control)
  control_60as.*      random-offset chance-match control
  dia_density_sample.csv  DiaObject counts within 30" (DIA-coverage test)
  summary.json        aggregate counts

Usage:
    python scripts/_exp_rsp_tns_edp2_xmatch.py [--radius 3.0] [--workers 8]
                                               [--limit N] [--no-sources]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

REPO = Path(__file__).resolve().parents[1]
SKILL = (Path.home() / ".claude" / "skills" / "rubin-edp2").resolve()
sys.path.insert(0, str(SKILL))

from rsp_token import load_rsp_token  # noqa: E402
from tap import tap_sync  # noqa: E402

TNS_CSV = REPO / "data" / "tns_public_objects.csv"
OUT_DIR = REPO / "reports" / "rsp_tns_edp2"

# dp2.DiaSource.midpointMjdTai range (live 2026-08-30 / 2026-09-23).
DP2_MJD_START = 60790.117
DP2_MJD_END = 61047.155

BANDS = "ugrizy"
CONE_COLS = ["diaObjectId", "ra", "dec", "nDiaSources"] + [
    f"{b}_psfFluxNdata" for b in BANDS
]
TNS_COLS = ["objid", "name_prefix", "name", "ra", "declination", "redshift",
            "type", "reporting_group", "discoverydate", "discoverymag",
            "discmagfilter", "internal_names"]


def query(adql: str, token: str, tries: int = 5) -> list[dict]:
    """Sync TAP with retry. Async /401s are transient here; sync is reliable."""
    for k in range(tries):
        try:
            return tap_sync(adql, token, timeout=120).as_dicts()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 ** k + np.random.rand())
    return []


def load_visits(token: str) -> pd.DataFrame:
    path = OUT_DIR / "visits.csv"
    if not path.exists():
        rows = query("SELECT visit, band, ra, dec, skyRotation, expMidptMJD, "
                     "expTime FROM dp2.Visit WHERE expTime > 0", token)
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path, dtype={"visit": "Int64"})


def load_tns(mjd_lo: float, mjd_hi: float) -> pd.DataFrame:
    t = pd.read_csv(TNS_CSV, skiprows=1, low_memory=False, usecols=TNS_COLS)
    dd = pd.to_datetime(t["discoverydate"], errors="coerce")
    t["disc_mjd"] = np.nan
    ok = dd.notna()
    t.loc[ok, "disc_mjd"] = Time(dd[ok].values.astype("datetime64[ns]")).mjd
    t = t[(t["disc_mjd"] >= mjd_lo) & (t["disc_mjd"] <= mjd_hi)]
    t = t.rename(columns={"ra": "tns_ra", "declination": "tns_dec",
                          "type": "tns_type", "redshift": "tns_z"})
    t["tns_name"] = (t["name_prefix"].fillna("") + " " + t["name"]).str.strip()
    return t.reset_index(drop=True)


def footprint(tns: pd.DataFrame, vis: pd.DataFrame, fov_deg: float) -> pd.DataFrame:
    """Coverage by visit-centre distance (LSSTCam radius ~1.75 deg)."""
    ct = SkyCoord(tns["tns_ra"].values * u.deg, tns["tns_dec"].values * u.deg)
    cv = SkyCoord(vis["ra"].values * u.deg, vis["dec"].values * u.deg)
    # (idx into ct, idx into cv): the first return indexes the search-around set.
    it, iv, sep, _ = cv.search_around_sky(ct, fov_deg * u.deg)
    df = pd.DataFrame({"i": it, "mjd": vis["expMidptMJD"].values[iv],
                       "band": vis["band"].values[iv], "sep": sep.deg})
    df["dt"] = df["mjd"] - tns["disc_mjd"].values[df["i"]]
    inner = df[df["sep"] <= 1.75]
    act = inner[(inner["dt"] > -30) & (inner["dt"] < 100)]
    g = df.groupby("i")
    out = tns.copy()
    out["min_visit_sep_deg"] = g["sep"].min().reindex(out.index)
    out["n_visits"] = inner.groupby("i").size().reindex(out.index).fillna(0).astype(int)
    out["n_visits_active"] = act.groupby("i").size().reindex(out.index).fillna(0).astype(int)
    out["active_bands"] = act.groupby("i")["band"].agg(
        lambda s: "".join(b for b in BANDS if b in set(s))).reindex(out.index)
    out["first_visit_dt"] = inner.groupby("i")["dt"].min().reindex(out.index)
    out["last_visit_dt"] = inner.groupby("i")["dt"].max().reindex(out.index)
    return out[out["min_visit_sep_deg"].notna()].reset_index(drop=True)


def cone_adql(ra: float, dec: float, r_deg: float) -> str:
    return (
        f"SELECT {', '.join(CONE_COLS)}, "
        f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) AS sep_deg "
        f"FROM dp2.DiaObject WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {r_deg})) = 1"
    )


def run_cones(names: list[str], ras: list[float], decs: list[float], token: str,
              radius: float, workers: int, ck: Path) -> dict:
    """name -> list of hit dicts. Resumes from the jsonl checkpoint `ck`."""
    done: dict[str, list] = {}
    if ck.exists():
        for line in ck.open():
            rec = json.loads(line)
            if "error" not in rec:
                done[rec["name"]] = rec["hits"]
    todo = [i for i in range(len(names)) if names[i] not in done]
    print(f"cones [{ck.name}]: {len(done)} cached, {len(todo)} to query "
          f"({workers} workers, r={radius}\")")
    lock = threading.Lock()
    r_deg = radius / 3600.0

    def one(i: int) -> tuple[str, dict]:
        try:
            hits = query(cone_adql(ras[i], decs[i], r_deg), token)
            # IDs stay Python int from as_dicts(); JSON keeps them exact.
            return names[i], {"name": names[i], "hits": hits}
        except Exception as e:  # noqa: BLE001
            return names[i], {"name": names[i], "error": f"{type(e).__name__}: {str(e)[:160]}"}

    t0 = time.time()
    n_err = 0
    with ck.open("a") as fh, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, i) for i in todo]
        for k, f in enumerate(as_completed(futs), 1):
            name, rec = f.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            if "error" in rec:
                n_err += 1
            else:
                done[name] = rec["hits"]
            if k % 200 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"  {k}/{len(futs)}  {el:.0f}s  "
                      f"eta {el / k * (len(futs) - k):.0f}s  errors={n_err}", flush=True)
    return done


def summarise_cones(cand: pd.DataFrame, done: dict, radius: float) -> pd.DataFrame:
    recs = []
    for name in cand["name"]:
        hits = done.get(name)
        if hits is None:
            recs.append({"name": name, "queried": False, "n_cone_hits": None})
            continue
        hits = sorted(hits, key=lambda h: h["sep_deg"])
        rec = {"name": name, "queried": True, "n_cone_hits": len(hits)}
        if hits:
            h = hits[0]
            rec.update(
                diaObjectId=str(int(h["diaObjectId"])),  # decimal string, never float
                sep_arcsec=h["sep_deg"] * 3600.0,
                dia_ra=h["ra"], dia_dec=h["dec"],
                nDiaSources=h["nDiaSources"],
                **{f"n_{b}": h.get(f"{b}_psfFluxNdata") for b in BANDS},
            )
            if len(hits) > 1:
                rec["sep2_arcsec"] = hits[1]["sep_deg"] * 3600.0
        recs.append(rec)
    cols = ["diaObjectId", "sep_arcsec", "dia_ra", "dia_dec", "nDiaSources",
            *[f"n_{b}" for b in BANDS], "sep2_arcsec"]
    r = pd.DataFrame(recs)
    r = r.reindex(columns=[*r.columns, *[c for c in cols if c not in r.columns]])
    m = cand.merge(r, on="name", how="left")
    m["matched_2as"] = m["sep_arcsec"].le(2.0)
    return m


SRC_COLS = ("diaObjectId", "diaSourceId", "visit", "band", "midpointMjdTai",
            "psfFlux", "psfFluxErr", "snr", "reliability")


def pull_sources(ids: list[str], token: str, workers: int, path: Path) -> pd.DataFrame:
    """Full DiaSource rows for the given diaObjectIds, cached to `path`."""
    have = pd.DataFrame(columns=list(SRC_COLS))
    if path.exists():
        have = pd.read_parquet(path)
    got = set(have["diaObjectId"].astype(str))
    todo = [i for i in ids if i not in got]
    chunks = [todo[i:i + 100] for i in range(0, len(todo), 100)]
    print(f"DiaSource rows: {len(got)} objects cached, {len(todo)} to pull "
          f"in {len(chunks)} chunks", flush=True)

    def one(ch: list[str]) -> list[dict]:
        return query(f"SELECT {', '.join(SRC_COLS)} FROM dp2.DiaSource "
                     f"WHERE diaObjectId IN ({','.join(ch)})", token)

    rows: list[dict] = []
    with ThreadPoolExecutor(min(workers, 4)) as ex:
        for k, f in enumerate(as_completed([ex.submit(one, c) for c in chunks]), 1):
            try:
                rows.extend(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"  DiaSource chunk failed: {type(e).__name__}: {str(e)[:160]}")
            if k % 5 == 0 or k == len(chunks):
                print(f"  DiaSource {k}/{len(chunks)}", flush=True)
    if rows:
        new = pd.DataFrame({c: [r.get(c) for r in rows] for c in SRC_COLS})
        for c in ("diaObjectId", "diaSourceId", "visit"):
            new[c] = pd.array([int(x) for x in new[c]], dtype="Int64")
        have = pd.concat([have.astype(new.dtypes.to_dict()), new], ignore_index=True)
        have.to_parquet(path, index=False)
    return have


def source_stats(m: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """Per-match DiaSource timing relative to the TNS discovery date.

    A real counterpart should be detected around discovery; a chance DiaObject
    (host nucleus, subtraction residual, unrelated variable) need not be.
    """
    s = src.assign(diaObjectId=src["diaObjectId"].astype(str))
    s = s.merge(m[["diaObjectId", "disc_mjd"]].dropna().drop_duplicates("diaObjectId"),
                on="diaObjectId")
    s["dt"] = s["midpointMjdTai"] - s["disc_mjd"]
    s["win"] = s["dt"].between(-30, 100)
    s["det5"] = s["snr"] >= 5
    g = s.groupby("diaObjectId")
    st = pd.DataFrame({
        "n_src": g.size(),
        "dia_mjd_first": g["midpointMjdTai"].min(),
        "dia_mjd_last": g["midpointMjdTai"].max(),
        "snr_max": g["snr"].max(),
        "n_src_win": g["win"].sum(),
        "n_det5_win": (s["win"] & s["det5"]).groupby(s["diaObjectId"]).sum(),
        "n_pos_win": (s["win"] & s["det5"] & (s["psfFlux"] > 0)).groupby(s["diaObjectId"]).sum(),
    })
    pdet = s[s["det5"] & (s["psfFlux"] > 0)]
    st["dia_mjd_first_pos"] = pdet.groupby("diaObjectId")["midpointMjdTai"].min()
    # epoch of brightest positive difference flux
    pos = s[s["psfFlux"] > 0]
    ipk = pos.groupby("diaObjectId")["psfFlux"].idxmax()
    st["peak_dt"] = pd.Series(pos.loc[ipk, "dt"].values, index=ipk.index)
    st["frac_src_win"] = st["n_src_win"] / st["n_src"]
    # temporally consistent: >=1 S/N>=5 positive detection within [-30,+100] d
    # and the brightest positive epoch in that window too.
    st["time_consistent"] = (st["n_pos_win"] >= 1) & st["peak_dt"].between(-30, 100)
    return st.reset_index()


def offset_positions(cand: pd.DataFrame, n: int, offset_arcsec: float,
                     seed: int = 0) -> pd.DataFrame:
    """Random-PA offsets of a random subset of candidates (chance-match control)."""
    rng = np.random.default_rng(seed)
    sub = cand.sample(n=min(n, len(cand)), random_state=seed).reset_index(drop=True)
    c = SkyCoord(sub["tns_ra"].values * u.deg, sub["tns_dec"].values * u.deg)
    o = c.directional_offset_by(rng.uniform(0, 360, len(sub)) * u.deg,
                                offset_arcsec * u.arcsec)
    sub["name"] = sub["name"] + "_off"
    sub["tns_ra"], sub["tns_dec"] = o.ra.deg, o.dec.deg
    return sub


def dia_density(m: pd.DataFrame, n_unmatched: int, n_matched: int, token: str,
                workers: int, path: Path, radius_arcsec: float = 30.0) -> pd.DataFrame:
    """Count DiaObjects within 30" of actively covered candidates.

    Visit-centre coverage overstates DIA coverage (chip gaps, no template).
    Zero DiaObjects within 30" means DIA effectively did not run there, so the
    unmatched are split into "not processed" vs "processed but missed".
    n_unmatched < 0 tests every unmatched candidate. Each count is appended to
    <path>.jsonl as it arrives; failed queries are retried on the next run.
    """
    act = m["n_visits_active"] > 0
    parts = []
    for group, sel, n in (("unmatched", act & ~m["matched_2as"], n_unmatched),
                          ("matched", act & m["matched_2as"], n_matched)):
        df = m[sel]
        k = len(df) if n < 0 else min(n, len(df))
        parts.append(df.sample(n=k, random_state=1).assign(group=group))
    samp = pd.concat(parts, ignore_index=True)

    have: dict[str, int] = {}
    if path.exists():
        c = pd.read_csv(path).dropna(subset=["n_dia_30as"])
        have.update(zip(c["name"], c["n_dia_30as"].astype(int)))
    ck = path.with_suffix(".jsonl")
    if ck.exists():
        for line in ck.open():
            r = json.loads(line)
            if r["n"] is not None:
                have[r["name"]] = r["n"]
    todo = [i for i in range(len(samp)) if samp["name"].iloc[i] not in have]
    r_deg = radius_arcsec / 3600.0

    def one(i: int) -> tuple[str, int | None]:
        ra, dec = samp["tns_ra"].iloc[i], samp["tns_dec"].iloc[i]
        q = (f"SELECT COUNT(*) AS n FROM dp2.DiaObject WHERE CONTAINS("
             f"POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r_deg})) = 1")
        try:
            return samp["name"].iloc[i], query(q, token)[0]["n"]
        except Exception:  # noqa: BLE001
            return samp["name"].iloc[i], None

    print(f"DIA density: {len(samp) - len(todo)} cached, {len(todo)} to query", flush=True)
    t0 = time.time()
    n_err = 0
    with ck.open("a") as fh, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, i) for i in todo]
        for k, f in enumerate(as_completed(futs), 1):
            name, n = f.result()
            fh.write(json.dumps({"name": name, "n": n}) + "\n")
            fh.flush()
            if n is None:
                n_err += 1
            else:
                have[name] = n
            if k % 500 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"  density {k}/{len(futs)}  {el:.0f}s  "
                      f"eta {el / k * (len(futs) - k):.0f}s  errors={n_err}", flush=True)
    samp["n_dia_30as"] = samp["name"].map(have)
    cols = ["name", "group", "reporting_group", "tns_dec", "discoverymag",
            "n_visits_active", "n_dia_30as"]
    samp[cols].to_csv(path, index=False)
    return samp


def summarise(m: pd.DataFrame) -> dict:
    """Counts for a matched table (real or control)."""
    mm = m[m["matched_2as"]]
    act = m["n_visits_active"] > 0
    out = {
        "n": int(len(m)),
        "active_coverage": int(act.sum()),
        "matched_2as": int(len(mm)),
        "matched_2as_given_active": int((act & m["matched_2as"]).sum()),
        "sep_arcsec_median": float(mm["sep_arcsec"].median()) if len(mm) else None,
    }
    if "time_consistent" in m:
        tc = m["matched_2as"] & m["time_consistent"].eq(True)
        out.update(time_consistent_2as=int(tc.sum()),
                   time_consistent_2as_given_active=int((tc & act).sum()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjd-start", type=float, default=DP2_MJD_START)
    ap.add_argument("--mjd-end", type=float, default=DP2_MJD_END)
    ap.add_argument("--pre-days", type=float, default=60.0,
                    help="also keep TNS objects discovered this long before DP2 start")
    ap.add_argument("--fov-deg", type=float, default=2.1,
                    help="visit-centre radius for the footprint prefilter")
    ap.add_argument("--radius", type=float, default=3.0,
                    help="cone radius in arcsec; matched_2as flags <= 2\"")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N candidates only")
    ap.add_argument("--no-sources", action="store_true")
    ap.add_argument("--control", type=int, default=1000,
                    help="random-offset control cones (0 disables)")
    ap.add_argument("--control-offset", type=float, default=60.0,
                    help="control offset in arcsec")
    ap.add_argument("--density-sample", type=int, default=-1,
                    help="unmatched active candidates to test for DIA coverage "
                         "(-1 all, 0 disables)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = load_rsp_token()
    vis = load_visits(token)
    tns = load_tns(args.mjd_start - args.pre_days, args.mjd_end)
    cand = footprint(tns, vis, args.fov_deg)
    cand = cand.sort_values("disc_mjd").reset_index(drop=True)
    cand.to_csv(OUT_DIR / "candidates.csv", index=False)
    print(f"visits {len(vis):,}; TNS in window {len(tns):,}; "
          f"within {args.fov_deg} deg of a visit {len(cand):,}; "
          f"visit in [-30,+100]d {int((cand['n_visits_active'] > 0).sum()):,}")
    if args.limit:
        cand = cand.head(args.limit)

    done = run_cones(cand["name"].tolist(), cand["tns_ra"].tolist(),
                     cand["tns_dec"].tolist(), token, args.radius, args.workers,
                     OUT_DIR / "cones.jsonl")
    m = summarise_cones(cand, done, args.radius)

    ctl = None
    if args.control:
        # Controls drawn from actively covered candidates, same footprint/time mix.
        off = offset_positions(cand[cand["n_visits_active"] > 0], args.control,
                               args.control_offset)
        cdone = run_cones(off["name"].tolist(), off["tns_ra"].tolist(),
                          off["tns_dec"].tolist(), token, args.radius, args.workers,
                          OUT_DIR / f"control_{args.control_offset:g}as.jsonl")
        ctl = summarise_cones(off, cdone, args.radius)

    if not args.no_sources:
        ids = set(m.loc[m["matched_2as"], "diaObjectId"].dropna())
        if ctl is not None:
            ids |= set(ctl.loc[ctl["matched_2as"], "diaObjectId"].dropna())
        src = pull_sources(sorted(ids), token, args.workers, OUT_DIR / "diasources.parquet")
        for name, df in (("real", m), ("control", ctl)):
            if df is None:
                continue
            st = source_stats(df, src)
            df = df.merge(st, on="diaObjectId", how="left")
            # First positive S/N>=5 detection; the first DiaSource of any sign is
            # often a negative subtraction residual and fakes an early "lead".
            df["lead_days"] = df["disc_mjd"] - df["dia_mjd_first_pos"]  # >0: EDP2 earlier
            bad = df["n_src"].notna() & (df["n_src"] != df["nDiaSources"])
            if bad.any():
                print(f"  WARNING [{name}]: {int(bad.sum())} objects n_src != nDiaSources")
            if name == "real":
                m = df
            else:
                ctl = df

    m.to_csv(OUT_DIR / "tns_edp2_match.csv", index=False)
    spec = m["tns_type"].notna()
    mm = m[m["matched_2as"]]
    summ = {
        "tns_in_window": len(tns),
        "real": summarise(m),
        "typed": int(spec.sum()),
        "typed_matched_2as": int((spec & m["matched_2as"]).sum()),
        "typed_matched_by_type": mm["tns_type"].value_counts().head(15).to_dict(),
        "multi_hit_2as": int((mm["sep2_arcsec"] <= 2.0).sum()),
    }
    if "lead_days" in m:
        ld = mm.loc[mm["sep_arcsec"] <= 0.5, "lead_days"].dropna()
        summ.update(lead_days_median_sep0p5=float(ld.median()) if len(ld) else None,
                    edp2_first_sep0p5=int((ld > 0).sum()),
                    edp2_first_by_3d_sep0p5=int((ld > 3).sum()),
                    n_with_pos_det_sep0p5=int(len(ld)))
    if ctl is not None:
        ctl.to_csv(OUT_DIR / f"control_{args.control_offset:g}as.csv", index=False)
        summ["control"] = summarise(ctl)
        summ["control_offset_arcsec"] = args.control_offset
    if args.density_sample:
        d = dia_density(m, args.density_sample, 150, token, args.workers,
                        OUT_DIR / "dia_density_sample.csv")
        u_ = d.loc[d["group"] == "unmatched", "n_dia_30as"].dropna()
        f0 = float((u_ == 0).mean())
        act = m["n_visits_active"] > 0
        n_hit = int((act & m["matched_2as"]).sum())
        n_miss = int((act & ~m["matched_2as"]).sum())
        covered = n_hit + (1 - f0) * n_miss
        # finite-population correction: zero when every unmatched one was tested
        fpc = (n_miss - len(u_)) / max(n_miss - 1, 1)
        se = np.sqrt(f0 * (1 - f0) / len(u_) * fpc) * n_miss
        # The 30" count at a matched position includes the match itself, so
        # "others == 0" is how often a DIA-processed position looks empty
        # (sparse field). Use it to correct the covered-unmatched count.
        others = d.loc[d["group"] == "matched", "n_dia_30as"].dropna() - 1
        f_sparse = float((others == 0).mean())
        n_cov_miss = int((u_ > 0).sum())
        cov_miss_corr = n_cov_miss / (1 - f_sparse)
        summ["dia_coverage"] = {
            "unmatched_total": n_miss,
            "unmatched_sampled": int(len(u_)),
            "unmatched_query_failed": int(d.loc[d["group"] == "unmatched", "n_dia_30as"].isna().sum()),
            "unmatched_zero_dia_30as_frac": f0,
            "unmatched_with_dia_30as": n_cov_miss,
            "est_active_in_dia_coverage": round(covered),
            "est_active_in_dia_coverage_se": round(float(se)),
            "est_recovery_2as_in_dia_coverage": round(n_hit / covered, 3),
            "matched_sampled": int(len(others)),
            "matched_zero_other_dia_30as_frac": round(f_sparse, 3),
            "est_recovery_2as_sparse_corrected": round(n_hit / (n_hit + cov_miss_corr), 3),
            "est_recovery_0p5as_sparse_corrected": round(
                int((act & m["sep_arcsec"].le(0.5)).sum()) / (n_hit + cov_miss_corr), 3),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
