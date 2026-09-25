#!/usr/bin/env python
"""Fetch ELAsTiCC2 training lightcurves (DESC public, NERSC portal) for the
optional GRU pretraining corpus (fusion_v11 sequence arm, all gated).

Data layer (spec §1, live-verified 2026-07-05):
  base  https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/
  no auth; ``Accept-Ranges: bytes`` (⇒ byte-range RESUME).  Per-class dirs
  ``ELASTICC2_TRAIN_02_<MODEL>/`` each hold ~40 HEAD + ~40 PHOT SNANA
  ``.FITS.gz`` shards (~15 MB each); SNIa-SALT3 alone is 172,180 LCs.  ~1.5M
  LCs / 36 classes total (full sample ≈ 7.91 GB).  Plain ``astropy.io.fits``
  reads the shards — this script only fetches bytes.

Default (spec §2.4, deviation 22): SELECTIVE per-class fetch of the SN-like
classes (~3-4 GB) — avoids the 30-40 GB decompress scratch a full-tarball pull
would need.  ``--full`` fetches every class dir instead.  Every download is
RESUMABLE: a partial shard is completed with a Range request, a complete shard
is skipped, so the job is safe to re-run / requeue on SCC.

Usage
-----
  # discover the exact model-class dir names on the portal
  python scripts/fetch_elasticc2.py --list

  # default: SN-like classes only, resumable, into data/elasticc2/
  python scripts/fetch_elasticc2.py --out data/elasticc2

  # explicit subset
  python scripts/fetch_elasticc2.py --models SNIa-SALT3,SNII-Templates

  # the whole 7.91 GB sample (every class dir)
  python scripts/fetch_elasticc2.py --full --out data/elasticc2

Downstream: scripts/train_seq_encoder.py SSL-pretrains on these (SNANA→sequence
adaptor lives in the P6 pretrain job); if the fetch is infeasible in-session the
sequence arm proceeds without ELAsTiCC2 (design §3.4) — this script + the SCC
pretrain job are the deliverable either way.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/"
    "ELASTICC2_TRAINING_SAMPLE_2/"
)
MODEL_DIR_PREFIX = "ELASTICC2_TRAIN_02_"
_USER_AGENT = "metaDEBASS-fetch-elasticc2/1.0 (+fusion_v11 seq arm)"
_HREF_RE = re.compile(r'href="([^"?][^"]*)"', re.IGNORECASE)

# SN-like model classes (default selective fetch).  Matched case-insensitively
# as substrings of the ``<MODEL>`` suffix so the exact portal spelling
# (SNIa-SALT3, SNII-Templates, SNIax, SNIa-91bg, SNIb/c-Templates, SLSN-I, …)
# is picked up without hardcoding every string — the Ia axis is the point of
# the arm, so all thermonuclear + core-collapse + superluminous SNe are in.
_SN_PATTERNS = ("snia", "snii", "snib", "snic", "sniax", "sn91bg", "91bg",
                "slsn", "sn-i", "tde", "pisn", "cart", "ilot")


def is_sn_model(model: str) -> bool:
    """True when the class name is an SN-like class (default-fetch member)."""
    key = model.lower()
    return any(pat in key for pat in _SN_PATTERNS)


def _parse_index_links(html: str) -> list[str]:
    """Hrefs from an Apache-style directory index (relative, deduped, ordered)."""
    seen: set[str] = set()
    out: list[str] = []
    for href in _HREF_RE.findall(html):
        if href in ("../", "./") or href.startswith(("/", "http://", "https://", "?")):
            continue
        if href not in seen:
            seen.add(href)
            out.append(href)
    return out


def _fetch_text(url: str, *, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def list_model_dirs(base_url: str = BASE_URL) -> list[str]:
    """Model-class names (``<MODEL>``) available under the base index."""
    links = _parse_index_links(_fetch_text(base_url))
    models = []
    for href in links:
        name = href.rstrip("/")
        if name.startswith(MODEL_DIR_PREFIX):
            models.append(name[len(MODEL_DIR_PREFIX):])
    return sorted(set(models))


def list_shards(base_url: str, model: str) -> list[str]:
    """``.FITS.gz`` shard filenames (HEAD + PHOT) inside one class dir."""
    dir_url = f"{base_url}{MODEL_DIR_PREFIX}{model}/"
    links = _parse_index_links(_fetch_text(dir_url))
    return [h for h in links if h.lower().endswith(".fits.gz")]


def select_models(models_arg: str | None, available: list[str], *, full: bool) -> list[str]:
    """Resolve the requested class subset against what the portal exposes.

    ``--full`` → everything; an explicit ``--models`` list is validated
    (unknown names are a hard error listing valid ones); otherwise the SN-like
    default subset.
    """
    if full:
        return list(available)
    if models_arg:
        wanted = [m.strip() for m in models_arg.split(",") if m.strip()]
        avail = set(available)
        unknown = [m for m in wanted if m not in avail]
        if unknown:
            raise SystemExit(
                f"unknown model class(es) {unknown}; available: {sorted(available)}")
        return wanted
    return [m for m in available if is_sn_model(m)]


def _remote_size(url: str, *, timeout: float = 60.0) -> tuple[int | None, bool]:
    """(content_length, accepts_ranges) via a HEAD request; (None, False) on error."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            accepts = (resp.headers.get("Accept-Ranges", "").lower() == "bytes")
            return (int(length) if length is not None else None), accepts
    except (urllib.error.URLError, ValueError):
        return None, False


def download_file(url: str, dest: Path, *, timeout: float = 300.0,
                  chunk: int = 1 << 20) -> str:
    """Resumable single-file download.  Returns "skip" | "resume" | "full"."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total, accepts = _remote_size(url, timeout=timeout)
    have = dest.stat().st_size if dest.exists() else 0
    if total is not None and have == total:
        return "skip"
    headers = {"User-Agent": _USER_AGENT}
    mode = "wb"
    outcome = "full"
    if have and accepts and (total is None or have < total):
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        outcome = "resume"
    elif have:  # cannot resume — restart cleanly
        have = 0
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, mode) as fh:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
    return outcome


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/elasticc2", help="Destination root dir")
    ap.add_argument("--base-url", default=BASE_URL, help="ELAsTiCC2 sample index URL")
    ap.add_argument("--models", default=None,
                    help="Comma-separated class names (default: SN-like subset)")
    ap.add_argument("--full", action="store_true",
                    help="Fetch EVERY class dir (~7.91 GB) instead of the SN subset")
    ap.add_argument("--list", action="store_true",
                    help="List available class-dir names and exit (no download)")
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="Cap shards per class (smoke / partial pull)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve + list shards that WOULD be fetched, no bytes")
    args = ap.parse_args()

    base_url = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    t0 = time.time()

    try:
        available = list_model_dirs(base_url)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"could not reach ELAsTiCC2 portal ({base_url}): {exc}")
    if not available:
        raise SystemExit(f"no {MODEL_DIR_PREFIX}* class dirs found at {base_url}")

    if args.list:
        print(f"{len(available)} ELAsTiCC2 class dirs at {base_url}:", flush=True)
        for m in available:
            print(f"  {'[SN]' if is_sn_model(m) else '    '} {m}", flush=True)
        return

    models = select_models(args.models, available, full=args.full)
    print(f"selected {len(models)}/{len(available)} class dirs "
          f"({'full' if args.full else 'SN subset' if not args.models else 'explicit'})",
          flush=True)

    out_root = Path(args.out)
    n_files = n_skip = n_get = n_bytes_files = 0
    for model in models:
        try:
            shards = list_shards(base_url, model)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  [{model}] index unreachable ({exc}) — skipping", flush=True)
            continue
        if args.limit_shards is not None:
            shards = shards[: args.limit_shards]
        model_dir = out_root / f"{MODEL_DIR_PREFIX}{model}"
        print(f"  [{model}] {len(shards)} shards → {model_dir}", flush=True)
        for shard in shards:
            n_files += 1
            url = f"{base_url}{MODEL_DIR_PREFIX}{model}/{shard}"
            dest = model_dir / shard
            if args.dry_run:
                print(f"    would fetch {url}", flush=True)
                continue
            try:
                outcome = download_file(url, dest, timeout=args.timeout)
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"    FAILED {shard} ({exc}) — re-run to resume", flush=True)
                continue
            if outcome == "skip":
                n_skip += 1
            else:
                n_get += 1
                n_bytes_files += dest.stat().st_size if dest.exists() else 0

    print(f"done: {n_files} shards ({n_skip} already complete, {n_get} fetched, "
          f"{n_bytes_files / 1e9:.2f} GB new) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
