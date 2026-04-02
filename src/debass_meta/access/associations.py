"""Helpers for resolving broker requests through object-association tables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import BrokerOutput
from .identifiers import IdentifierKind, infer_identifier_kind

_ZTF_ASSOCIATION_BROKERS = {"alerce", "fink", "alerce_history"}


@dataclass(frozen=True)
class ResolvedObjectReference:
    primary_object_id: str
    requested_object_id: str | None
    primary_identifier_kind: IdentifierKind
    requested_identifier_kind: IdentifierKind | None
    survey: str
    resolution_status: str
    resolution_reason: str | None = None
    associated_object_id: str | None = None
    association_kind: str | None = None
    association_source: str | None = None
    association_sep_arcsec: float | None = None

    @property
    def can_query(self) -> bool:
        return bool(self.requested_object_id)


def load_lsst_ztf_associations(
    path: Path | None,
    *,
    max_sep_arcsec: float | None = None,
) -> dict[str, dict[str, object]]:
    if path is None or not Path(path).exists():
        return {}

    import pandas as pd

    df = pd.read_csv(path, dtype={"lsst_object_id": "string", "ztf_object_id": "string"})
    if len(df) == 0:
        return {}

    lookup: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        object_id = _to_text(row.get("lsst_object_id"))
        if not object_id:
            continue
        match_status = _to_text(row.get("match_status")) or "unknown"
        ztf_object_id = _to_text(row.get("ztf_object_id"))
        sep_arcsec = _to_float(row.get("sep_arcsec"))
        if match_status != "matched" or not ztf_object_id:
            continue
        if max_sep_arcsec is not None and sep_arcsec is not None and sep_arcsec > max_sep_arcsec:
            continue
        lookup[object_id] = {
            "ztf_object_id": ztf_object_id,
            "sep_arcsec": sep_arcsec,
            "association_kind": _to_text(row.get("association_kind")) or "position_crossmatch",
            "association_source": _to_text(row.get("association_source")) or "alerce_conesearch",
            "match_count": _to_int(row.get("match_count")),
        }
    return lookup


def resolve_object_reference(
    primary_object_id: str,
    *,
    broker: str,
    associations: dict[str, dict[str, object]] | None = None,
) -> ResolvedObjectReference:
    primary_object_id = str(primary_object_id).strip()
    primary_identifier_kind = infer_identifier_kind(primary_object_id)
    survey = "LSST" if primary_identifier_kind == "lsst_dia_object_id" else "ZTF"

    if broker not in _ZTF_ASSOCIATION_BROKERS or primary_identifier_kind != "lsst_dia_object_id":
        return ResolvedObjectReference(
            primary_object_id=primary_object_id,
            requested_object_id=primary_object_id,
            primary_identifier_kind=primary_identifier_kind,
            requested_identifier_kind=primary_identifier_kind,
            survey=survey,
            resolution_status="direct",
        )

    assoc = (associations or {}).get(primary_object_id)
    if assoc is None:
        return ResolvedObjectReference(
            primary_object_id=primary_object_id,
            requested_object_id=None,
            primary_identifier_kind=primary_identifier_kind,
            requested_identifier_kind=None,
            survey=survey,
            resolution_status="unresolved",
            resolution_reason="missing_crossmatch_association",
        )

    requested_object_id = str(assoc["ztf_object_id"]).strip()
    return ResolvedObjectReference(
        primary_object_id=primary_object_id,
        requested_object_id=requested_object_id,
        primary_identifier_kind=primary_identifier_kind,
        requested_identifier_kind=infer_identifier_kind(requested_object_id),
        survey=survey,
        resolution_status="crossmatched",
        resolution_reason="position_crossmatch",
        associated_object_id=requested_object_id,
        association_kind=str(assoc.get("association_kind") or "position_crossmatch"),
        association_source=str(assoc.get("association_source") or "alerce_conesearch"),
        association_sep_arcsec=_to_float(assoc.get("sep_arcsec")),
    )


def apply_reference_metadata(
    output: BrokerOutput,
    reference: ResolvedObjectReference,
) -> BrokerOutput:
    output.object_id = reference.primary_object_id
    output.survey = reference.survey
    output.primary_object_id = reference.primary_object_id
    output.requested_object_id = reference.requested_object_id
    output.primary_identifier_kind = reference.primary_identifier_kind
    output.requested_identifier_kind = reference.requested_identifier_kind
    output.associated_object_id = reference.associated_object_id
    output.association_kind = reference.association_kind
    output.association_source = reference.association_source
    output.association_sep_arcsec = reference.association_sep_arcsec
    return output


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text or text.lower() == "nan" or text == "<NA>":
        return None
    return text


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None
