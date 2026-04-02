"""Helpers for broker/object identifier compatibility."""
from __future__ import annotations

from typing import Literal

IdentifierKind = Literal["ztf_object_id", "lsst_dia_object_id", "unknown"]


def infer_identifier_kind(object_id: str | int | None) -> IdentifierKind:
    text = str(object_id or "").strip()
    if not text:
        return "unknown"
    if text.upper().startswith("ZTF"):
        return "ztf_object_id"
    if text.isdigit():
        return "lsst_dia_object_id"
    return "unknown"


def is_lsst_lasair_endpoint(endpoint: str) -> bool:
    base = str(endpoint or "").rstrip("/").lower()
    return "lasair.lsst.ac.uk" in base and "lasair-ztf.lsst.ac.uk" not in base


def build_lasair_api_endpoint(endpoint: str) -> str:
    base = str(endpoint or "").rstrip("/")
    if base.endswith("/api"):
        return base
    return f"{base}/api"
