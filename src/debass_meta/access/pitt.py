"""Pitt-Google broker adapter — Phase 2 stub.

Requires Google Cloud service-account configuration.
Deferred until phase-1 stack is proven.
"""
from __future__ import annotations

from typing import Any

from .base import BrokerAdapter, BrokerOutput, SemanticType


class PittAdapter(BrokerAdapter):
    name = "pitt_google"
    phase = 2
    semantic_type: SemanticType = "probability"

    def probe(self) -> dict[str, Any]:
        return {"broker": self.name, "status": "stubbed", "reason": "phase 2 — GCP credentials required"}

    def fetch_object(self, object_id: str) -> BrokerOutput:
        return BrokerOutput(
            broker=self.name,
            object_id=object_id,
            query_time=self.now(),
            raw_payload={},
            semantic_type=self.semantic_type,
            fields=[],
            availability=False,
            fixture_used=False,
        )

    def fetch_lightcurve(self, object_id: str) -> dict[str, Any]:
        return {}
