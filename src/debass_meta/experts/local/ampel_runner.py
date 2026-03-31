"""AMPEL local expert runner.

AMPEL is a framework, not a single classifier. This wrapper runs
specific T2 units (e.g. T2SNcosmo, T2TabulatedRiseDeclineStat)
and emits broker_outputs-compatible ExpertOutput records.

Requires: ampel-hu-astro (https://github.com/AmpelAstro/Ampel-HU-astro)
"""
from __future__ import annotations

from typing import Any

from .base import LocalExpert, ExpertOutput

# T2 units to run by default
_DEFAULT_UNITS = ["T2SNcosmo", "T2TabulatedRiseDeclineStat"]


class AmpelRunner(LocalExpert):
    name = "ampel"
    semantic_type = "probability"  # varies by unit; most return probability-like outputs

    def __init__(self, units: list[str] | None = None) -> None:
        self.units = units or _DEFAULT_UNITS
        self._ampel_available = False
        try:
            import ampel  # noqa: F401
            self._ampel_available = True
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, lightcurves: Any, labels: Any) -> None:
        # AMPEL units are pre-trained; no fitting needed at the framework level
        pass

    def predict_epoch(self, object_id: str, lightcurve: Any, epoch_jd: float) -> ExpertOutput:
        truncated = self._truncate(lightcurve, epoch_jd)

        if self._ampel_available:
            results = self._run_units(object_id, truncated)
        else:
            results = {unit: None for unit in self.units}

        # Aggregate: if any unit returns a probability, use it; else stub
        probs: dict[str, float] = {}
        for unit, res in results.items():
            if isinstance(res, dict) and "prob_ia" in res:
                probs[f"{unit}_prob_ia"] = float(res["prob_ia"])

        if not probs:
            probs = {"stub_prob_ia": 0.5}  # uniform prior placeholder

        return ExpertOutput(
            expert=self.name,
            object_id=object_id,
            epoch_jd=epoch_jd,
            class_probabilities=probs,
            raw_output=results,
            model_version="stub" if not self._ampel_available else "ampel",
            available=self._ampel_available,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "units": self.units,
            "available": self._ampel_available,
            "requires": "ampel-hu-astro",
            "epoch_aware": True,
            "note": "Framework wrapper — output semantics depend on the T2 unit used.",
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _truncate(self, lightcurve: Any, epoch_jd: float) -> Any:
        if isinstance(lightcurve, list):
            return [det for det in lightcurve if det.get("jd", 0) <= epoch_jd]
        try:
            import pandas as pd
            if isinstance(lightcurve, pd.DataFrame) and "jd" in lightcurve.columns:
                return lightcurve[lightcurve["jd"] <= epoch_jd]
        except ImportError:
            pass
        return lightcurve

    def _run_units(self, object_id: str, lightcurve: Any) -> dict[str, Any]:
        """Run each configured AMPEL T2 unit. Stub until full AMPEL setup."""
        # Full implementation requires AMPEL context, DB connection, and unit configs.
        # See: https://github.com/AmpelAstro/Ampel-HU-astro
        return {unit: None for unit in self.units}
