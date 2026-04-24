"""Probe LSST lightcurve coverage head-to-head: Lasair LSST vs Fink LSST vs ALeRCE LSST.

For a random sample of LSST diaObjectIds from labels.csv, call each endpoint and
report: HTTP status, #detections returned, useful photometry fields present,
and latency. We need to know which source to prefer before patching fetch.
"""
from __future__ import annotations
import os, sys, time, json, random
from pathlib import Path
import requests
import pandas as pd

sys.path.insert(0, "src")
from debass_meta.access.identifiers import infer_identifier_kind

N_PROBE = 8
SEED = 42


def probe_lasair(oid: str, endpoint: str, token: str, timeout: int = 20) -> dict:
    url = f"{endpoint.rstrip('/')}/api/object/{oid}/"
    headers = {"Authorization": f"Token {token}"} if token else {}
    t0 = time.time()
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        dt = time.time() - t0
        try:
            data = r.json()
        except Exception:
            data = {}
        sources = data.get("diaSourcesList") or []
        if not sources and isinstance(data.get("lasairData"), dict):
            sources = data["lasairData"].get("diaSourcesList") or []
        sample_keys = sorted(list(sources[0].keys()))[:15] if sources else []
        return {
            "status": r.status_code,
            "n_sources": len(sources),
            "sample_keys": sample_keys,
            "latency_s": round(dt, 2),
        }
    except Exception as e:
        return {"status": "ERR", "error": type(e).__name__, "msg": str(e)[:120],
                "latency_s": round(time.time()-t0, 2)}


def probe_fink_lsst(oid: str, timeout: int = 20) -> dict:
    url = "https://api.lsst.fink-portal.org/api/v1/sources"
    payload = {"diaObjectId": oid, "output-format": "json"}
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        dt = time.time() - t0
        try:
            data = r.json()
        except Exception:
            data = []
        n = len(data) if isinstance(data, list) else 0
        sample_keys = sorted(list(data[0].keys()))[:20] if (n and isinstance(data[0], dict)) else []
        # Look for photometry-carrying fields
        photo_hints = [k for k in sample_keys if any(t in k.lower() for t in
                       ["psfflux", "mag", "band", "midpointtai", "mjd", "ccd"])]
        return {
            "status": r.status_code,
            "n_sources": n,
            "sample_keys": sample_keys[:12],
            "photometry_fields": photo_hints,
            "latency_s": round(dt, 2),
        }
    except Exception as e:
        return {"status": "ERR", "error": type(e).__name__, "msg": str(e)[:120],
                "latency_s": round(time.time()-t0, 2)}


def probe_alerce_lsst(oid: str, timeout: int = 20) -> dict:
    # ALeRCE LSST: try their v2 API if exposed. The object endpoint is
    # https://api.alerce.online/lsst/v1/objects/{oid} — confirmed by docs.
    for url in [
        f"https://api.alerce.online/lsst/v1/objects/{oid}/lightcurve",
        f"https://api.alerce.online/lsst/v1/objects/{oid}",
        f"https://api.alerce.online/ztf/v1/objects/{oid}",  # sanity: should 404 for LSST
    ]:
        t0 = time.time()
        try:
            r = requests.get(url, timeout=timeout)
            dt = time.time() - t0
            ok = r.ok
            try:
                data = r.json()
            except Exception:
                data = None
            n_detections = 0
            if isinstance(data, dict):
                n_detections = (
                    len(data.get("detections") or [])
                    or len(data.get("diaSources") or [])
                )
            elif isinstance(data, list):
                n_detections = len(data)
            yield_one = {
                "url": url,
                "status": r.status_code,
                "ok": ok,
                "n_detections": n_detections,
                "latency_s": round(dt, 2),
            }
            if r.ok and n_detections > 0:
                return yield_one
            last = yield_one
        except Exception as e:
            last = {"url": url, "status": "ERR", "error": type(e).__name__,
                    "msg": str(e)[:80], "latency_s": round(time.time()-t0, 2)}
    return last


def main() -> None:
    labels = pd.read_csv("data/labels.csv")
    labels["id_kind"] = labels["object_id"].apply(infer_identifier_kind)
    lsst = labels[labels["id_kind"] == "lsst_dia_object_id"]
    random.seed(SEED)
    sample = lsst["object_id"].sample(N_PROBE, random_state=SEED).tolist()
    print(f"Probing {len(sample)} LSST ids:\n  " + "\n  ".join(sample))
    print()

    las_endpoint = os.environ.get("LASAIR_LSST_ENDPOINT", "")
    las_token = os.environ.get("LASAIR_LSST_TOKEN", "")
    print(f"[env] LASAIR_LSST_ENDPOINT set: {bool(las_endpoint)} | TOKEN set: {bool(las_token)}")
    print()

    summary = {"lasair": {"hits": 0, "total_src": 0, "err": 0},
               "fink":   {"hits": 0, "total_src": 0, "err": 0},
               "alerce": {"hits": 0, "total_src": 0, "err": 0}}

    for oid in sample:
        print(f"=== {oid} ===")
        las = probe_lasair(oid, las_endpoint, las_token)
        print("  lasair:", json.dumps(las, default=str))
        if isinstance(las.get("n_sources"), int) and las["n_sources"] > 0:
            summary["lasair"]["hits"] += 1
            summary["lasair"]["total_src"] += las["n_sources"]
        if las.get("status") == "ERR":
            summary["lasair"]["err"] += 1

        fnk = probe_fink_lsst(oid)
        print("  fink:  ", json.dumps(fnk, default=str))
        if isinstance(fnk.get("n_sources"), int) and fnk["n_sources"] > 0:
            summary["fink"]["hits"] += 1
            summary["fink"]["total_src"] += fnk["n_sources"]
        if fnk.get("status") == "ERR":
            summary["fink"]["err"] += 1

        alr = probe_alerce_lsst(oid)
        print("  alerce:", json.dumps(alr, default=str))
        if isinstance(alr.get("n_detections"), int) and alr["n_detections"] > 0:
            summary["alerce"]["hits"] += 1
            summary["alerce"]["total_src"] += alr["n_detections"]
        if alr.get("status") == "ERR":
            summary["alerce"]["err"] += 1
        print()

    print("=== SUMMARY ===")
    for src, s in summary.items():
        print(f"  {src}: coverage {s['hits']}/{len(sample)} | avg srcs/obj "
              f"{(s['total_src']/s['hits']) if s['hits'] else 0:.1f} | errors {s['err']}")


if __name__ == "__main__":
    main()
