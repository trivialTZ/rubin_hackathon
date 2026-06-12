#!/usr/bin/env python3
"""Validate refreshed NeurIPS-style figures and status deck facts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from debass_meta.features.lightcurve import FEATURE_NAMES
from debass_meta.projectors.base import ALL_EXPERT_KEYS


EXPECTED_ASSETS = [
    "presentations/figures/top12_auc.png",
    "presentations/figures/top12_auc.pdf",
    "presentations/figures/reliability_v7.png",
    "presentations/figures/reliability_v7.pdf",
    "presentations/figures/architecture_v7.png",
    "presentations/figures/architecture_v7.pdf",
    "docs/slides_figures/fig5_architecture.png",
    "docs/slides_figures/fig5_architecture.pdf",
]


def _png_nonblank(path: Path) -> None:
    with Image.open(path) as img:
        if img.width < 200 or img.height < 150:
            raise AssertionError(f"{path} is unexpectedly small: {img.size}")
        gray = img.convert("L")
        lo, hi = gray.getextrema()
        if hi - lo < 8:
            raise AssertionError(f"{path} appears blank or near-blank")


def _asset_checks() -> None:
    for rel in EXPECTED_ASSETS:
        path = ROOT / rel
        if not path.exists():
            raise AssertionError(f"Missing asset: {path}")
        if path.stat().st_size < 1024:
            raise AssertionError(f"Asset is too small: {path}")
        if path.suffix == ".png":
            _png_nonblank(path)


def _repo_fact_checks() -> None:
    if len(FEATURE_NAMES) != 51:
        raise AssertionError(f"FEATURE_NAMES count changed: {len(FEATURE_NAMES)}")
    if len(ALL_EXPERT_KEYS) != 28:
        raise AssertionError(f"ALL_EXPERT_KEYS count changed: {len(ALL_EXPERT_KEYS)}")


def _pptx_text(path: Path) -> str:
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    chunks: list[str] = []
    with ZipFile(path) as zf:
        slide_names = sorted(
            (n for n in zf.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")),
            key=lambda n: int(n.rsplit("slide", 1)[1].split(".xml", 1)[0]),
        )
        for name in slide_names:
            root = ET.fromstring(zf.read(name))
            chunks.extend(t.text or "" for t in root.findall(".//a:t", ns))
    return "\n".join(chunks)


def _deck_checks(pptx: Path) -> None:
    if not pptx.exists():
        raise AssertionError(f"Missing deck: {pptx}")
    text = _pptx_text(pptx)
    required = [
        "expert_confidence",
        "p_follow_proxy",
        "latest_object_unsafe",
        "51",
        "28",
        "12",
    ]
    for token in required:
        if token not in text:
            raise AssertionError(f"Deck is missing required token: {token}")
    forbidden = ["0.9949", "Slide numbers", "one calibrated follow-up score"]
    for token in forbidden:
        if token in text:
            raise AssertionError(f"Deck still contains stale token: {token}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pptx", default="presentations/metaDEBASS_status_2026-04-26.pptx")
    args = parser.parse_args()
    _repo_fact_checks()
    _asset_checks()
    _deck_checks(ROOT / args.pptx)
    print("NeurIPS figure/deck checks passed")


if __name__ == "__main__":
    main()
