"""Abstract base class for all broker/classifier adapters."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .identifiers import IdentifierKind

SemanticType = Literal[
    "probability",
    "binary_score",
    "context_class",
    "crossmatch_tag",
    "heuristic_class",
    "mixed",
]


@dataclass
class BrokerOutput:
    """Canonical container for a single broker response.

    raw_label_or_score  — verbatim value returned by the broker
    semantic_type       — how to interpret the value
    canonical_projection — normalised float in [0,1] where meaningful,
                           else None (never overwrite raw with this)
    """
    broker: str
    object_id: str
    query_time: float          # unix timestamp
    raw_payload: dict[str, Any]
    semantic_type: SemanticType
    survey: str = "ZTF"
    source_endpoint: str | None = None
    request_params: dict[str, Any] = field(default_factory=dict)
    status_code: int | None = None
    primary_object_id: str | None = None
    requested_object_id: str | None = None
    primary_identifier_kind: IdentifierKind | None = None
    requested_identifier_kind: IdentifierKind | None = None
    associated_object_id: str | None = None
    association_kind: str | None = None
    association_source: str | None = None
    association_sep_arcsec: float | None = None
    # Per-field breakdowns (list of dicts with keys: field, raw_label_or_score,
    # semantic_type, canonical_projection)
    fields: list[dict[str, Any]] = field(default_factory=list)
    # Optional event-level rows retained alongside the raw payload.
    events: list[dict[str, Any]] = field(default_factory=list)
    availability: bool = True  # False when fixture fallback was used
    fixture_used: bool = False


class BrokerAdapter(ABC):
    """One adapter per broker/local-expert source."""

    name: str                    # subclass must set
    phase: int = 1
    semantic_type: SemanticType  # subclass must set

    # ------------------------------------------------------------------ #
    # Required interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def probe(self) -> dict[str, Any]:
        """Lightweight liveness check. Returns status dict."""

    @abstractmethod
    def fetch_object(self, object_id: str) -> BrokerOutput:
        """Fetch classifier output for a single object."""

    @abstractmethod
    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        """Fetch raw photometric light curve for an object."""

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def save_fixture(self, data: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    def load_fixture(self, path: Path) -> Any:
        with open(path) as fh:
            return json.load(fh)

    def unsupported_identifier_output(
        self,
        object_id: str,
        *,
        source_endpoint: str | None = None,
        request_params: dict[str, Any] | None = None,
        survey: str = "ZTF",
        identifier_kind: IdentifierKind = "unknown",
        expected_identifier_kind: IdentifierKind | None = None,
        reason: str = "unsupported_identifier",
    ) -> BrokerOutput:
        raw_payload: dict[str, Any] = {
            "reason": reason,
            "identifier_kind": identifier_kind,
        }
        if expected_identifier_kind is not None:
            raw_payload["expected_identifier_kind"] = expected_identifier_kind
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw_payload,
            semantic_type=self.semantic_type,
            survey=survey,
            source_endpoint=source_endpoint,
            request_params=request_params or {},
            primary_object_id=object_id,
            requested_object_id=object_id,
            primary_identifier_kind=identifier_kind,
            requested_identifier_kind=identifier_kind,
            fields=[],
            events=[],
            availability=False,
            fixture_used=False,
        )

    def unavailable_output(
        self,
        object_id: str,
        *,
        source_endpoint: str | None = None,
        request_params: dict[str, Any] | None = None,
        survey: str = "ZTF",
        identifier_kind: IdentifierKind = "unknown",
        reason: str = "unavailable",
        raw_payload_extra: dict[str, Any] | None = None,
    ) -> BrokerOutput:
        raw_payload: dict[str, Any] = {
            "reason": reason,
            "identifier_kind": identifier_kind,
        }
        if raw_payload_extra:
            raw_payload.update(raw_payload_extra)
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload=raw_payload,
            semantic_type=self.semantic_type,
            survey=survey,
            source_endpoint=source_endpoint,
            request_params=request_params or {},
            primary_object_id=object_id,
            requested_object_id=None,
            primary_identifier_kind=identifier_kind,
            requested_identifier_kind=None,
            fields=[],
            events=[],
            availability=False,
            fixture_used=False,
        )

    @staticmethod
    def now() -> float:
        return time.time()
