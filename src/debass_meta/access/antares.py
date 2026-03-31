"""ANTARES broker adapter — Phase 2 stub.

Open client surface exists but stream access requires credentials.
Deferred until phase-1 stack is proven.
"""
from __future__ import annotations

from typing import Any

from .base import BrokerAdapter, BrokerOutput, SemanticType


class AntaresAdapter(BrokerAdapter):
    name = "antares"
    phase = 2
    semantic_type: SemanticType = "probability"

    def probe(self) -> dict[str, Any]:
        return {"broker": self.name, "status": "stubbed", "reason": "phase 2 — credentials required"}

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
