#!/usr/bin/env python3
"""scripts/fink_local_rubin.py - run Fink's Rubin/LSST SNN and CATS classifiers locally (no Spark).

Re-implements the bodies of the two pandas UDFs of fink-science (pinned clone on SCC) as plain functions, with the
same preprocessing and the same model files:

  fink_rubin_snn   fink_science/rubin/snn/processor.py::snn_ia_elasticc, model name passed by fink-broker
                   (fink_broker/rubin/science.py) = "elasticc_binary_broad/SN_vs_other"  ->  f:clf_snnSnVsOthers_score
  fink_rubin_cats  fink_science/rubin/cats/processor.py::predict_nn, TFSMLayer on
                   cats_small_nometa_serial_219_savedmodel, class mapping {0:11,1:12,2:13,3:21,4:22}
                                                                           ->  f:clf_cats_class / f:clf_cats_score

Input history Fink used for an alert = prvDiaSources (earlier diaSources of the object) followed by the alert's own
diaSource, ordered in time; the psfFlux columns (nJy) are scaled by 10**(-(31.4-27.5)/2.5) for SNN (FLUXCAL) and used
raw (min-max normalised per light curve) for CATS. Only diaSources count (no forced photometry), negative fluxes included.

Sub-commands (run on SCC with the venv in /projectnb/pi-brout/tztang/fink_local; `source env.sh` first):
  fetch-fink  --ids ids.txt --out DIR        cache Fink /api/v1/sources rows (public API, <= 2 req/s)
  validate    --fink-dir DIR --out DIR       rebuild what Fink saw per alert, compare with the stored scores
  score       --cohort NAME --lc-dir DIR ... per-epoch scores for cached lightcurves, local-expert parquet schema

Object / source ids are strings everywhere (Rubin ids are ~1e17; never pass them through float64).
DP2-derived outputs stay on SCC; never copy them into the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FL_ROOT = Path(os.environ.get("FL", "/projectnb/pi-brout/tztang/fink_local"))
FINK_SCIENCE = Path(os.environ.get("FINK_SCIENCE_DIR", FL_ROOT / "fink-science"))
SNN_MODEL_NAME = "elasticc_binary_broad/SN_vs_other"  # fink-broker rubin/science.py (unchanged since 2025-10)
CATS_MODEL_DIR = "cats_small_nometa_serial_219_savedmodel"
FAC_FLUXCAL = 10 ** (-(31.4 - 27.5) / 2.5)
CATS_FILTERS = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5, "y": 6}
CATS_CLASS_NAMES = ["SN-like", "Fast", "Long", "Periodic", "non-Periodic (AGN)"]
CATS_CODE = {0: 11, 1: 12, 2: 13, 3: 21, 4: 22}  # mapping_cats_general in fink-broker
FINK_API = "https://api.lsst.fink-portal.org/api/v1/sources"
EXPERT_SNN = "fink_rubin_snn"
EXPERT_CATS = "fink_rubin_cats"

_FETCH_COLS = [
    "r:diaSourceId", "r:diaObjectId", "r:midpointMjdTai", "r:band", "r:psfFlux", "r:psfFluxErr", "r:isNegative", "r:snr",
    "f:clf_snnSnVsOthers_score", "f:clf_cats_class", "f:clf_cats_score", "f:fink_science_version",
    "f:fink_broker_version",
]


def _git_rev(path: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"], text=True).strip()


def model_version() -> str:
    return f"fink-science@{_git_rev(FINK_SCIENCE)} snn={SNN_MODEL_NAME} cats={CATS_MODEL_DIR}"


# --------------------------------------------------------------------------------------------------------------------
# Models (plain-function equivalents of the Fink pandas UDFs)
# --------------------------------------------------------------------------------------------------------------------
class SNN:
    """snn_ia_elasticc: P(SN) = prob_class0 of the 'SN_vs_other' binary model."""

    def __init__(self) -> None:
        from supernnova.validation.validate_onthefly import classify_lcs  # noqa: PLC0415
        self._classify = classify_lcs
        self.model = str(FINK_SCIENCE / "fink_science/data/models/snn_models" / SNN_MODEL_NAME / "model.pt")

    def predict(self, alerts: list[dict], chunk: int = 2000) -> dict[str, float]:
        """alerts: [{key, mjd[], band[], flux[], err[]}] (psfFlux in nJy). Returns key -> P(SN) (NaN when dropped)."""
        out: dict[str, float] = {}
        for s in range(0, len(alerts), chunk):
            part = alerts[s:s + chunk]
            n = [len(a["mjd"]) for a in part]
            pdf = pd.DataFrame({
                "SNID": np.repeat([str(a["key"]) for a in part], n),
                "MJD": np.concatenate([np.asarray(a["mjd"], dtype=float) for a in part]),
                "FLUXCAL": np.concatenate([np.asarray(a["flux"], dtype=float) for a in part]) * FAC_FLUXCAL,
                "FLUXCALERR": np.concatenate([np.asarray(a["err"], dtype=float) for a in part]) * FAC_FLUXCAL,
                "FLT": np.concatenate([np.asarray(a["band"], dtype=object) for a in part]),
            })
            ids, probs = self._classify(pdf, self.model, "cpu")
            for i, p in zip(ids, probs):
                out[str(i)] = float(np.asarray(p).reshape(-1)[0])
        for a in alerts:
            out.setdefault(str(a["key"]), float("nan"))
        return out


def _norm_column(col) -> np.ndarray:  # fink_science.rubin.cats.utilities.norm_column
    col = np.array(col)
    if len(col) == 1:
        return np.array([1.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return (col - col.min()) / np.ptp(col)


class CATS:
    """predict_nn: 5-way softmax (SN-like, Fast, Long, Periodic, non-Periodic) of the CBPF 'cats small nometa' model."""

    def __init__(self) -> None:
        import tensorflow as tf  # noqa: PLC0415
        from tensorflow import keras  # noqa: PLC0415
        self._tf, self._keras = tf, keras
        layer = tf.keras.layers.TFSMLayer(
            str(FINK_SCIENCE / "fink_science/data/models/cats_models" / CATS_MODEL_DIR), call_endpoint="serving_default")
        inp = tf.keras.layers.Input(shape=(395, 4), dtype=tf.float32)
        self.net = tf.keras.Model(inp, layer(inp))

    def predict(self, alerts: list[dict], chunk: int = 512) -> dict[str, list[float]]:
        """Returns key -> 5 probabilities; all zeros for single-point light curves (Fink: class -1, score 0)."""
        out: dict[str, list[float]] = {}
        keep = []
        for a in alerts:
            if len(a["mjd"]) > 1:
                keep.append(a)
            else:
                out[str(a["key"])] = [0.0] * 5
        pad = self._keras.utils.pad_sequences
        for s in range(0, len(keep), chunk):
            part = keep[s:s + chunk]
            filt = [np.array([CATS_FILTERS[f] for f in a["band"]]).astype(np.int16) for a in part]
            mjd = [np.asarray(a["mjd"], dtype=float) - a["mjd"][0] for a in part]
            flux = [_norm_column(a["flux"]) for a in part]
            err = [_norm_column(a["err"]) for a in part]
            flux = pad(flux, maxlen=395, value=-999.0, padding="post", dtype=np.float32)
            mjd = pad(mjd, maxlen=395, value=-999.0, padding="post", dtype=np.float32)
            err = pad(err, maxlen=395, value=-999.0, padding="post", dtype=np.float32)
            band = pad(filt, maxlen=395, value=0.0, padding="post", dtype=np.uint8)
            lc = np.concatenate([mjd[..., None], flux[..., None], err[..., None], band[..., None]], axis=-1)
            pred = self.net.predict([lc], verbose=0)["dense_24"]
            for a, p in zip(part, pred):
                out[str(a["key"])] = [float(x) for x in p]
        return out


def cats_class_score(p5: list[float]) -> tuple[int, float]:
    """cats_class / cats_score exactly as fink-broker derives them (class -1 / score 0 when the array is all 0)."""
    arr = np.asarray(p5, dtype=float)
    if not np.nanmax(arr) > 0:
        return -1, 0.0
    return CATS_CODE[int(np.argmax(arr))], float(np.max(arr))


# --------------------------------------------------------------------------------------------------------------------
# Epoch construction
# --------------------------------------------------------------------------------------------------------------------
def make_alerts(oid: str, rows: list[dict]) -> list[dict]:
    """rows: diaSources sorted in time. Epoch k (1-based) = what Fink saw at the k-th alert: rows[:k]."""
    mjd = np.array([r["mjd"] for r in rows], dtype=float)
    flux = np.array([r["flux"] for r in rows], dtype=float)
    err = np.array([r["err"] for r in rows], dtype=float)
    band = [r["band"] for r in rows]
    return [{"key": f"{oid}:{k}", "mjd": mjd[:k], "flux": flux[:k], "err": err[:k], "band": band[:k]}
            for k in range(1, len(rows) + 1)]


def load_cached_lc(path: Path, with_raw: bool = False):
    """Cached lightcurve JSON -> diaSource rows only (no forced photometry), time-ordered, deduplicated.

    Handles the ALeRCE-style cache (`measurement_id`, `_source` = alerce_lsst | fink_lsst_fp) and the DP2 TAP cache
    (`diaSourceId`, `_source` = dp2_diasource). Negative fluxes are kept (Fink does not cut on sign).
    """
    with open(path) as fh:
        dets = json.load(fh)
    rows, seen = [], set()
    for d in dets:
        if not isinstance(d, dict):
            continue
        src = str(d.get("_source") or "")
        if "fp" in src or "forced" in src:
            continue
        sid = d.get("diaSourceId") if d.get("diaSourceId") is not None else d.get("measurement_id")
        flux, err, band = d.get("psfFlux"), d.get("psfFluxErr"), d.get("band")
        mjd = d.get("midpointMjdTai") if d.get("midpointMjdTai") is not None else d.get("mjd")
        if sid is None or flux is None or err is None or band not in CATS_FILTERS or mjd is None:
            continue
        sid = str(int(sid)) if not isinstance(sid, str) else sid
        if sid in seen:
            continue
        seen.add(sid)
        rows.append({"sid": sid, "mjd": float(mjd), "flux": float(flux), "err": float(err), "band": band})
    rows.sort(key=lambda r: (r["mjd"], r["sid"]))
    return (rows, len(dets)) if with_raw else rows


def fink_rows(fink_dir: Path, oid: str) -> list[dict]:
    """diaSource rows (same layout as load_cached_lc) from a cached Fink /sources dump, [] when absent."""
    p = fink_dir / f"{oid}.json"
    if not p.exists():
        return []
    return [{"sid": str(r["r:diaSourceId"]), "mjd": float(r["r:midpointMjdTai"]), "flux": float(r["r:psfFlux"]),
             "err": float(r["r:psfFluxErr"]), "band": r["r:band"]}
            for r in json.loads(p.read_text())
            if r.get("r:psfFlux") is not None and r.get("r:band") in CATS_FILTERS]


# --------------------------------------------------------------------------------------------------------------------
# fetch-fink
# --------------------------------------------------------------------------------------------------------------------
def cmd_fetch(args) -> None:
    import requests  # noqa: PLC0415
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = [x.strip() for x in Path(args.ids).read_text().split() if x.strip()]
    for i, oid in enumerate(ids):
        dst = out / f"{oid}.json"
        if dst.exists():
            continue
        for attempt in range(6):
            try:
                r = requests.post(FINK_API, json={"diaObjectId": oid, "output-format": "json"}, timeout=120)
                if r.status_code >= 500 or r.status_code == 429:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                txt = r.text  # parse ints exactly: json.loads keeps big ints as Python ints
                data = json.loads(txt)
                rows = [{k: (str(v) if k in ("r:diaSourceId", "r:diaObjectId") and v is not None else v)
                         for k, v in a.items() if k in _FETCH_COLS} for a in data]
                dst.write_text(json.dumps(rows))
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  {oid}: attempt {attempt} failed: {exc}", file=sys.stderr)
                time.sleep(2 ** attempt)
        time.sleep(0.6)  # <= 2 req/s
        if (i + 1) % 25 == 0:
            print(f"fetched {i + 1}/{len(ids)}", flush=True)


# --------------------------------------------------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------------------------------------------------
def _summ(d: np.ndarray, label: str) -> dict:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    return {"set": label, "n": int(len(d)), "median_abs": float(np.median(np.abs(d))) if len(d) else None,
            "p90_abs": float(np.quantile(np.abs(d), 0.9)) if len(d) else None,
            "max_abs": float(np.max(np.abs(d))) if len(d) else None,
            "frac_lt_0.01": float(np.mean(np.abs(d) < 0.01)) if len(d) else None,
            "frac_lt_0.05": float(np.mean(np.abs(d) < 0.05)) if len(d) else None}


def _cached_complete(oid: str, lc_dirs: list[Path], cap: int) -> list[dict] | None:
    """diaSources of the cached lightcurve when the cache is complete (fewer raw rows than the row cap), else None."""
    for d in lc_dirs:
        p = d / f"{oid}.json"
        if p.exists():
            with open(p) as fh:
                n_raw = len(json.load(fh))
            return load_cached_lc(p) if n_raw < cap else None
    return None


def cmd_validate(args) -> None:
    """Rebuild what Fink saw per alert and compare with the stored scores.

    Variant `api`: history = the object's earlier alerts as returned by the Fink API. Fink's own prvDiaSources can
    contain diaSources that never became a stored alert, so a second variant `union` adds the diaSources of the cached
    (ALeRCE) lightcurve when that cache is complete (< cap rows incl. forced photometry).
    """
    fink_dir, out = Path(args.fink_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lc_dirs = [Path(d) for d in (args.lc_dir or [])]
    alerts, meta = [], []
    for f in sorted(fink_dir.glob("*.json")):
        rows = json.loads(f.read_text())
        rows = [r for r in rows if r.get("r:midpointMjdTai") is not None and r.get("r:psfFlux") is not None]
        rows.sort(key=lambda r: (r["r:midpointMjdTai"], r["r:diaSourceId"]))
        oid = f.stem
        lcrows = [{"sid": r["r:diaSourceId"], "mjd": r["r:midpointMjdTai"], "flux": r["r:psfFlux"],
                   "err": r["r:psfFluxErr"], "band": r["r:band"]} for r in rows]
        cache = _cached_complete(oid, lc_dirs, args.cap) if lc_dirs else None
        union = None
        if cache is not None:
            have = {r["sid"] for r in lcrows}
            union = sorted(lcrows + [r for r in cache if r["sid"] not in have], key=lambda r: (r["mjd"], r["sid"]))
        for k, a in enumerate(make_alerts(oid, lcrows), start=1):
            alerts.append(a)
            r = rows[k - 1]
            m = {"object_id": oid, "k": k, "diaSourceId": r["r:diaSourceId"], "mjd": r["r:midpointMjdTai"],
                 "fink_snn": r.get("f:clf_snnSnVsOthers_score"), "fink_cats_class": r.get("f:clf_cats_class"),
                 "fink_cats_score": r.get("f:clf_cats_score"),
                 "fink_science_version": r.get("f:fink_science_version"),
                 "fink_broker_version": r.get("f:fink_broker_version"), "cache_complete": union is not None,
                 "n_hist_api": k, "n_hist_union": None}
            if union is not None:
                pos = next(i for i, u in enumerate(union) if u["sid"] == r["r:diaSourceId"])
                u_alert = make_alerts(oid, union[:pos + 1])[-1]
                u_alert["key"] = f"U:{oid}:{k}"
                alerts.append(u_alert)
                m["n_hist_union"] = pos + 1
            meta.append(m)
    n_ob = len({m["object_id"] for m in meta})
    print(f"{len(meta)} alerts from {n_ob} objects", flush=True)
    t0 = time.time()
    snn = SNN().predict(alerts)
    print(f"SNN done {time.time() - t0:.0f}s", flush=True)
    cats = CATS().predict(alerts)
    print(f"CATS done {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(meta)
    for var, pref in (("api", ""), ("union", "U:")):
        df[f"snn_{var}"] = [snn.get(f"{pref}{m['object_id']}:{m['k']}", np.nan) for m in meta]
        p5 = [cats.get(f"{pref}{m['object_id']}:{m['k']}", [np.nan] * 5) for m in meta]
        cc = [cats_class_score(p) if not np.isnan(p[0]) else (-9, np.nan) for p in p5]
        df[f"cats_class_{var}"] = [c[0] for c in cc]
        df[f"cats_score_{var}"] = [c[1] for c in cc]
        df[f"cats_psn_{var}"] = [p[0] for p in p5]
    for c in ("fink_snn", "fink_cats_class", "fink_cats_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(out / "validation_alerts.parquet", index=False)

    def block(d: pd.DataFrame, var: str) -> dict:
        r: dict = {"n_alerts": int(len(d)), "n_objects": int(d.object_id.nunique())}
        dn = d[f"snn_{var}"] - d.fink_snn
        ok = d.fink_snn.notna() & d[f"snn_{var}"].notna()
        r["snn"] = _summ(dn[ok].to_numpy(), "all")
        r["snn"]["pearson_r"] = (float(np.corrcoef(d[f"snn_{var}"][ok], d.fink_snn[ok])[0, 1]) if ok.sum() > 2 else None)
        r["snn_by_k"] = [_summ(dn[ok & m].to_numpy(), lab) for lab, m in
                         [("k=1", d.k == 1), ("k=2-4", d.k.between(2, 4)), ("k=5-19", d.k.between(5, 19)),
                          ("k>=20", d.k >= 20)]]
        dc = d[f"cats_score_{var}"] - d.fink_cats_score
        okc = d.fink_cats_score.notna() & d[f"cats_score_{var}"].notna()
        r["cats_score"] = _summ(dc[okc].to_numpy(), "all")
        r["cats_score"]["pearson_r"] = (float(np.corrcoef(d[f"cats_score_{var}"][okc], d.fink_cats_score[okc])[0, 1])
                                        if okc.sum() > 2 else None)
        agree = d[f"cats_class_{var}"] == d.fink_cats_class
        r["cats_class_agreement"] = float(agree[okc].mean())
        r["cats_class_agreement_k>1"] = float(agree[okc & (d.k > 1)].mean())
        r["cats_by_k"] = [_summ(dc[okc & m].to_numpy(), lab) for lab, m in
                          [("k=1", d.k == 1), ("k=2-4", d.k.between(2, 4)), ("k=5-19", d.k.between(5, 19)),
                           ("k>=20", d.k >= 20)]]
        r["by_fink_science_version"] = {
            str(v): {"n": int(len(g)), "snn_median_abs": float((g[f"snn_{var}"] - g.fink_snn).abs().median()),
                     "snn_frac_lt_0.01": float(((g[f"snn_{var}"] - g.fink_snn).abs() < 0.01).mean()),
                     "cats_median_abs": float((g[f"cats_score_{var}"] - g.fink_cats_score).abs().median()),
                     "cats_class_agree": float((g[f"cats_class_{var}"] == g.fink_cats_class).mean())}
            for v, g in d.groupby(d.fink_science_version.fillna("NA"))}
        return r

    rep: dict = {"model_version": model_version(), "all_alerts_api_history": block(df, "api")}
    sub = df[df.cache_complete]
    if len(sub):
        rep["complete_cache_subset_api_history"] = block(sub, "api")
        rep["complete_cache_subset_union_history"] = block(sub, "union")
    (out / "validation_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


# --------------------------------------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------------------------------------
def cmd_score(args) -> None:
    lc_dir = Path(args.lc_dir)
    if args.ids:
        ids = [x.strip() for x in Path(args.ids).read_text().split() if x.strip()]
        files = [lc_dir / f"{i}.json" for i in ids]
    else:
        files = sorted(lc_dir.glob("*.json"))
    # Rubin objects only: all-digit stems of diaObjectId length (ZTF ids start with 'ZTF')
    files = [f for f in files if f.stem.isdigit() and len(f.stem) >= 15 and f.exists()]
    files = sorted(files, key=lambda f: f.stem)
    shard = int(args.shard_id) if args.shard_id is not None else int(os.environ.get("SGE_TASK_ID", "1")) - 1
    nsh = int(args.n_shards) if args.n_shards is not None else int(os.environ.get("SGE_TASK_LAST", "1"))
    files = [f for i, f in enumerate(files) if i % nsh == shard]
    print(f"[{args.cohort}] shard {shard}/{nsh}: {len(files)} objects", flush=True)

    alerts, meta = [], []
    for f in files:
        rows, n_raw = load_cached_lc(f, with_raw=True)
        n_cache = len(rows)
        if args.fink_dir:  # union with Fink's own alert rows (same diaSources, plus ones the cache lacks)
            have = {r["sid"] for r in rows}
            rows = sorted(rows + [r for r in fink_rows(Path(args.fink_dir), f.stem) if r["sid"] not in have],
                          key=lambda r: (r["mjd"], r["sid"]))
        if not rows:
            continue
        capped = n_raw >= args.cap
        al = make_alerts(f.stem, rows)
        n_pos = np.cumsum([r["flux"] > 0 for r in rows])
        for k, (a, r) in enumerate(zip(al, rows), start=1):
            a["key"] = f"{f.stem}:{k}"
            meta.append({"object_id": f.stem, "diaSourceId": r["sid"], "n_det": k, "n_pos_det": int(n_pos[k - 1]),
                         "alert_mjd": r["mjd"], "alert_jd": r["mjd"] + 2400000.5, "n_raw_rows": n_raw,
                         "lc_capped": bool(capped), "n_cache_diasources": n_cache, "key": a["key"]})
        alerts.extend(al)
    print(f"  {len(meta)} epochs", flush=True)
    base = Path(args.out) / args.cohort / "local_expert_outputs"
    if not meta:
        return
    ver = model_version()
    common = {"survey": "LSST", "model_version": ver, "available": True, "dp2": bool(args.dp2), "cohort": args.cohort}
    tag = f"part-{shard:04d}.parquet"
    t0 = time.time()
    if not args.skip_snn:
        snn = SNN().predict(alerts)
        df = pd.DataFrame(meta)
        df["expert"] = EXPERT_SNN
        df["p_sn"] = [snn[k] for k in df.key]
        df["class_probabilities"] = [json.dumps({"SN": p, "other": 1.0 - p}, sort_keys=True) for p in df.p_sn]
        for k, v in common.items():
            df[k] = v
        d = base / EXPERT_SNN
        d.mkdir(parents=True, exist_ok=True)
        df.drop(columns="key").to_parquet(d / tag, index=False)
        print(f"  SNN {len(df)} rows {time.time() - t0:.0f}s", flush=True)
    if not args.skip_cats:
        cats = CATS().predict(alerts)
        df = pd.DataFrame(meta)
        df["expert"] = EXPERT_CATS
        p5 = [cats[k] for k in df.key]
        cc = [cats_class_score(p) for p in p5]
        df["cats_class"] = [c[0] for c in cc]
        df["cats_score"] = [c[1] for c in cc]
        df["p_sn_like"] = [p[0] for p in p5]
        df["class_probabilities"] = [json.dumps(dict(zip(CATS_CLASS_NAMES, p)), sort_keys=True) for p in p5]
        for k, v in common.items():
            df[k] = v
        d = base / EXPERT_CATS
        d.mkdir(parents=True, exist_ok=True)
        df.drop(columns="key").to_parquet(d / tag, index=False)
        print(f"  CATS {len(df)} rows {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch-fink")
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("validate")
    p.add_argument("--fink-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lc-dir", action="append", help="cached lightcurve dir(s) for the union-history variant")
    p.add_argument("--cap", type=int, default=50, help="row cap of the cached lightcurves (incl. forced photometry)")
    p = sub.add_parser("score")
    p.add_argument("--cohort", required=True)
    p.add_argument("--lc-dir", required=True)
    p.add_argument("--ids", default=None, help="optional id list (one per line); default every Rubin lightcurve")
    p.add_argument("--out", default=str(FL_ROOT / "outputs"))
    p.add_argument("--dp2", action="store_true", help="mark outputs as DP2-derived (reprocessed photometry)")
    p.add_argument("--shard-id", default=None)
    p.add_argument("--n-shards", default=None)
    p.add_argument("--cap", type=int, default=50,
                   help="row cap of the cached lightcurves (incl. forced photometry); lc_capped = raw rows >= cap")
    p.add_argument("--fink-dir", default=None, help="optional dir of fetch-fink dumps merged into the history")
    p.add_argument("--skip-snn", action="store_true")
    p.add_argument("--skip-cats", action="store_true")
    args = ap.parse_args()
    {"fetch-fink": cmd_fetch, "validate": cmd_validate, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
