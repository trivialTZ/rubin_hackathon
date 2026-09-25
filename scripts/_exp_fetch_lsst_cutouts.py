#!/usr/bin/env python3
"""Fetch LSST DP2 deep-coadd cutouts for the DEBaSS x LSST overlap sample.

For each object with both DEBaSS and DP2 photometry, pull a 20" radius cutout
in every available band, render to PNG (zscale, greyscale) and cache both the
PNG bytes and the pixel stats.

DP2 exposes ONLY lsst.deep_coadd -- there are no per-epoch visit or difference
images until full DP2 (Oct-Dec 2026), so these are static reference images,
not a time series.

Output: data/lsst_cutouts/<SNID>_<band>.png  +  cutout_index.json
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
import urllib.parse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from debass_meta.access.rubin_rsp import RSPClient, load_rsp_token  # noqa: E402

OUT = REPO / "data" / "lsst_cutouts"
RADIUS_ARCSEC = 20.0
CUTOUT_URL = "https://data.lsst.cloud/api/cutout/sync"
PNG_PX = 256


def retry_after_seconds(hdr: str | None, default: float = 20.0) -> float:
    """Retry-After is either delta-seconds or an HTTP-date (RFC 9110)."""
    if not hdr:
        return default
    try:
        return float(hdr)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone
        when = parsedate_to_datetime(hdr)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except Exception:  # noqa: BLE001
        return default


def render_png(fits_bytes: bytes) -> tuple[bytes | None, dict]:
    """FITS -> zscaled greyscale PNG. Returns (png_bytes, stats)."""
    from astropy.io import fits
    from astropy.visualization import ZScaleInterval
    from PIL import Image

    with fits.open(BytesIO(fits_bytes)) as hdul:
        data = None
        for h in hdul:
            if getattr(h, "data", None) is not None and h.data.ndim == 2:
                data = h.data.astype(float)
                break
    if data is None or data.size == 0:
        return None, {}

    finite = np.isfinite(data)
    if not finite.any():
        return None, {}
    stats = {
        "shape": list(data.shape),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "median": float(np.nanmedian(data)),
    }
    lo, hi = ZScaleInterval().get_limits(data[finite])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        lo, hi = np.nanpercentile(data[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((data - lo) / (hi - lo), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0)
    img = Image.fromarray((scaled * 255).astype(np.uint8), mode="L")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # FITS origin is bottom-left
    img = img.resize((PNG_PX, PNG_PX), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), stats


def fetch_one(args) -> dict:
    """Fetch one cutout. The cutout API rate-limits at 35/min, so back off and
    retry on 429 rather than burning the job."""
    snid, band, ivoid, ra, dec, token = args
    rec = {"SNID": snid, "band": band, "ok": False}
    try:
        pos = f"CIRCLE {ra} {dec} {RADIUS_ARCSEC / 3600.0}"
        url = (f"{CUTOUT_URL}?ID={urllib.parse.quote(ivoid, safe='')}"
               f"&POS={urllib.parse.quote(pos)}")
        r = None
        for attempt in range(6):
            r = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                             timeout=300)
            if r.status_code != 429:
                break
            time.sleep(retry_after_seconds(r.headers.get("Retry-After")) + 5 * attempt)
        if r is None or r.status_code != 200:
            rec["error"] = f"HTTP {getattr(r, 'status_code', '?')}: {r.text[:120] if r else ''}"
            return rec
        png, stats = render_png(r.content)
        if png is None:
            rec["error"] = "no 2D image plane in cutout"
            return rec
        path = OUT / f"{snid}_{band}.png"
        path.write_bytes(png)
        rec.update(ok=True, png=path.name, fits_bytes=len(r.content),
                   png_bytes=len(png), **stats)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return rec


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    token = load_rsp_token()
    client = RSPClient()

    cov = pd.read_csv(REPO / "reports" / "rsp_debass38" / "coverage_debass_x_lsst.csv")
    both = cov[cov.debass_file.notna() & cov.matched_diaObjectId.notna()]
    print(f"objects with DEBaSS + DP2: {len(both)}")

    # Resolve the deep-coadd datalink ID per (object, band).
    jobs = []
    for i in range(len(both)):
        snid = both["SNID"].iloc[i]
        ra = float(both["tns_ra"].iloc[i])
        dec = float(both["tns_dec"].iloc[i])
        try:
            o = client.query(
                "SELECT obs_id, lsst_band, access_url FROM ivoa.ObsCore "
                "WHERE obs_collection='LSST.DP2' AND dataproduct_subtype='lsst.deep_coadd' "
                f"AND CONTAINS(POINT('ICRS',{ra},{dec}), s_region)=1"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {snid}: ObsCore query failed: {type(e).__name__}")
            continue
        # One cutout per (object, band): overlapping patches would otherwise
        # queue duplicate fetches that all write the same filename.
        seen = set()
        for j in range(len(o)):
            band = str(o["lsst_band"].iloc[j])
            if (snid, band) in seen:
                continue
            dl = str(o["access_url"].iloc[j])
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(dl).query)
            if "ID" not in qs:
                continue
            seen.add((snid, band))
            jobs.append((snid, band, qs["ID"][0], ra, dec, token))
        print(f"  {snid}: {len(seen)} bands")

    # Skip anything already on disk so a re-run resumes instead of restarting.
    jobs = [j for j in jobs if not (OUT / f"{j[0]}_{j[1]}.png").exists()]
    print(f"\nfetching {len(jobs)} cutouts at {RADIUS_ARCSEC}\" "
          f"(2 workers, API limit 35/min) ...")
    results = []
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        for n, rec in enumerate(ex.map(fetch_one, jobs), 1):
            results.append(rec)
            if not rec["ok"]:
                print(f"  [{n}/{len(jobs)}] {rec['SNID']} {rec['band']} FAIL "
                      f"{rec.get('error')}")
            elif n % 10 == 0:
                print(f"  [{n}/{len(jobs)}] ok", flush=True)

    ok = [r for r in results if r["ok"]]
    total = sum(r["png_bytes"] for r in ok)
    (OUT / "cutout_index.json").write_text(json.dumps(results, indent=1))
    print(f"\n{len(ok)}/{len(results)} cutouts OK, {total/1e6:.1f} MB PNG")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
