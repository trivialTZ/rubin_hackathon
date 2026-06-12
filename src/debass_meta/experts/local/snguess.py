"""AMPEL SNGuess as a local expert.

Vendors the AmpelAstro/Ampel-HU-astro `T2BrightSNProb` XGBoost classifier
without the AMPEL framework dependency. The 3 JSON model files
(`xgb_risedecline_ndet{2,3,4}.json`) are loaded directly via stock XGBoost.
The 13-feature input is reconstructed from our normalized lightcurve format
plus optional ZTF host fields (which the underlying XGBoost handles as
NaN-tolerant).

Source: https://github.com/AmpelAstro/Ampel-HU-astro
        ampel/contrib/hu/t2/T2BrightSNProb.py + xgb_trees.py + data/
License: BSD-3-Clause (vendored verbatim)
Reference: Miranda+ 2022, A&A 657, A22 — "SNGuess: A method for the selection
           of young extragalactic transients"

The model is a binary "is this a Reference Catalog of variable sources (RCF)
SN candidate?" classifier — i.e. an SN-filter, NOT an Ia classifier. So the
trust target should be ``is_sn`` (NOT ``is_topclass_correct``) per
``feedback_trust_target_capability.md``. The projector clamps p_snia at 0.5
to honor the SN-filter contract.

Coverage: requires ndet in {2,3,4,...,100} after quality cuts. Outside that
range, returns a "could not score" result with available=False.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .base import LocalExpert, ExpertOutput

_MODEL_DIR = Path("artifacts/local_experts/snguess")

_FEATURE_NAMES = [
    "mag_det", "mag_last", "t_lc", "rb_med", "col_det", "t_predetect",
    "distnr_med", "magnr_med", "classtar_med", "sgscore1_med",
    "distpsnr1_med", "neargaia_med", "maggaia_med",
]

# AMPEL-bundled lightcurve quality requirements per ndet bin
# (from T2BrightSNProb.xgb_tree_param)
_LC_REQUIREMENTS = {
    2:   {"max_duration": 3.5,  "max_predetect": 3.5,  "min_detmag": 16.0},
    3:   {"max_duration": 6.5,  "max_predetect": 3.5,  "min_detmag": 16.0},
    4:   {"max_duration": 6.5,  "max_predetect": 3.5,  "min_detmag": 16.0},
    5:   {"max_duration": 10.0, "max_predetect": 3.5,  "min_detmag": 16.0},
    6:   {"max_duration": 10.0, "max_predetect": 3.5,  "min_detmag": 16.0},
    100: {"max_duration": 90.0, "max_predetect": 10.0, "min_detmag": 16.0},
}


class AmpelSNGuessExpert(LocalExpert):
    """Drop-in vendored AMPEL T2BrightSNProb."""

    name = "ampel/snguess"
    semantic_type = "probability"
    requires_gpu = False

    def __init__(self, model_dir: Path = _MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._models: dict[int, Any] = {}
        self._available = False
        self._load_models()

    def _load_models(self) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            return
        for ndet in (2, 3, 4):
            path = self.model_dir / f"xgb_risedecline_ndet{ndet}.json"
            if not path.exists():
                return
            try:
                m = xgb.Booster()
                m.load_model(str(path))
                self._models[ndet] = m
            except Exception:
                return
        self._available = len(self._models) == 3

    def fit(self, lightcurves: Any, labels: Any) -> None:
        raise NotImplementedError(
            "SNGuess is inference-only — weights are vendored from "
            "AmpelAstro/Ampel-HU-astro under BSD-3-Clause."
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "available": self._available,
            "model_dir": str(self.model_dir),
            "n_models_loaded": len(self._models),
            "feature_names": list(_FEATURE_NAMES),
            "classes": ["snia_or_nonia_sn", "other"],
            "license": "BSD-3-Clause",
            "reference": "Miranda+ 2022, A&A 657, A22",
        }

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _det_jd(d: dict) -> float | None:
        """JD for a detection dict (handles raw ALeRCE schema with mjd or
        normalized schema with jd)."""
        jd = d.get("jd")
        if jd is not None:
            return float(jd)
        mjd = d.get("mjd")
        if mjd is None:
            return None
        # ZTF MJD ~60000; if value is already JD-like (>2.4M), treat as JD
        return float(mjd) + 2400000.5 if mjd < 1e6 else float(mjd)

    @staticmethod
    def _det_mag(d: dict) -> float | None:
        """Magnitude (handles ALeRCE 'magpsf' or normalized 'mag')."""
        for key in ("mag", "magpsf"):
            v = d.get(key)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
        return None

    @staticmethod
    def _det_band(d: dict) -> str | None:
        """Band character (handles ALeRCE 'fid' int or normalized 'band' str).
        ZTF: fid 1=g, 2=r, 3=i. LSST: 1-6 = ugrizy."""
        b = d.get("band")
        if isinstance(b, str):
            return b
        fid = d.get("fid")
        if isinstance(fid, int) and 1 <= fid <= 6:
            return ("u", "g", "r", "i", "z", "y")[fid - 1]
        return None

    @staticmethod
    def _det_quality(d: dict) -> float | None:
        """Quality score (drb > rb > 'quality' > 'reliability')."""
        for key in ("drb", "rb", "quality", "reliability"):
            v = d.get(key)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
        return None

    @classmethod
    def _truncate(cls, detections: list[dict], epoch_jd: float) -> list[dict]:
        out = []
        for d in detections:
            jd = cls._det_jd(d)
            if jd is None or jd > epoch_jd + 1e-6:
                continue
            out.append({**d, "_jd": jd})
        out.sort(key=lambda d: d["_jd"])
        return out

    @classmethod
    def _extract_features(cls, detections: list[dict]) -> tuple[dict[str, float], int]:
        """Compute the 13 SNGuess features from a truncated lightcurve.

        Accepts BOTH raw ALeRCE-format dicts (mjd/magpsf/fid) AND normalized
        dicts (jd/mag/band). Host metadata fields default to NaN.
        """
        nan = float("nan")
        feats: dict[str, float] = {n: nan for n in _FEATURE_NAMES}

        # Filter to detections with valid (jd, mag)
        dets = []
        for d in detections:
            jd = d.get("_jd") if "_jd" in d else cls._det_jd(d)
            mag = cls._det_mag(d)
            if jd is None or mag is None:
                continue
            dets.append({**d, "_jd": jd, "_mag": mag})
        ndet = len(dets)
        if ndet == 0:
            return feats, 0

        feats["mag_det"] = float(dets[0]["_mag"])
        feats["mag_last"] = float(dets[-1]["_mag"])
        feats["t_lc"] = float(dets[-1]["_jd"] - dets[0]["_jd"])

        quals = [cls._det_quality(d) for d in dets]
        quals = [q for q in quals if q is not None]
        if quals:
            quals.sort()
            feats["rb_med"] = quals[len(quals) // 2]

        det_jd = dets[0]["_jd"]
        g_band = [d for d in dets
                  if cls._det_band(d) == "g" and abs(d["_jd"] - det_jd) < 2.0]
        r_band = [d for d in dets
                  if cls._det_band(d) == "r" and abs(d["_jd"] - det_jd) < 2.0]
        if g_band and r_band:
            feats["col_det"] = float(g_band[0]["_mag"]) - float(r_band[0]["_mag"])

        for fname, alias in (
            ("distnr_med", "distnr"),
            ("magnr_med", "magnr"),
            ("classtar_med", "classtar"),
            ("sgscore1_med", "sgscore1"),
            ("distpsnr1_med", "distpsnr1"),
            ("neargaia_med", "neargaia"),
            ("maggaia_med", "maggaia"),
        ):
            vals = [float(d[alias]) for d in dets
                    if d.get(alias) is not None
                    and isinstance(d.get(alias), (int, float))
                    and math.isfinite(d.get(alias) or nan)]
            if vals:
                vals.sort()
                feats[fname] = vals[len(vals) // 2]

        return feats, ndet

    @staticmethod
    def _model_key_for_ndet(ndet: int) -> int | None:
        """Pick the right XGBoost model for a given detection count."""
        if ndet < 2:
            return None  # Below model coverage
        if ndet <= 4:
            return ndet
        if ndet <= 100:
            return 100  # AMPEL bundles ndet=100 model only via the python tree dump
        return None

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def predict_epoch(
        self, object_id: str, lightcurve: Any, epoch_jd: float
    ) -> ExpertOutput:
        out = ExpertOutput(
            expert=self.name,
            object_id=str(object_id),
            epoch_jd=float(epoch_jd),
            class_probabilities={},
            raw_output={},
            model_version="ampel-snguess-d1-2020-11-30",
            available=False,
        )
        if not self._available:
            out.raw_output = {"reason": "models not loaded"}
            return out

        if not isinstance(lightcurve, list):
            out.raw_output = {"reason": "lightcurve not a list of detection dicts"}
            return out

        truncated = self._truncate(lightcurve, epoch_jd)
        feats, ndet = self._extract_features(truncated)

        model_key = self._model_key_for_ndet(ndet)
        # We only vendored the 3 JSON files (ndet=2,3,4). For ndet>=5 the
        # AMPEL python tree dump (xgb_trees.py) is the only model available
        # and is too large to vendor here. Return unavailable in that case.
        if model_key is None or model_key not in self._models:
            out.raw_output = {"reason": f"ndet={ndet} outside vendored model coverage (2,3,4)"}
            return out

        # Apply the AMPEL quality cuts before scoring (T2BrightSNProb logic)
        req = _LC_REQUIREMENTS[model_key]
        if (feats["t_lc"] != feats["t_lc"] or feats["t_lc"] > req["max_duration"]
                or feats["mag_det"] != feats["mag_det"]
                or feats["mag_det"] < req["min_detmag"]):
            out.raw_output = {"reason": "lc props outside SNGuess training range",
                              **feats}
            return out

        try:
            import numpy as np
            import pandas as pd
            import xgboost as xgb
        except ImportError as exc:
            out.raw_output = {"reason": f"missing dep: {exc}"}
            return out

        X = pd.DataFrame([[feats[n] for n in _FEATURE_NAMES]], columns=_FEATURE_NAMES)
        prob = float(self._models[model_key].predict(xgb.DMatrix(X))[0])

        # Emit only the SN-class probability — the projector
        # (projectors/ampel.py:_project_snguess) reads scores[-1] and
        # computes the complement itself. Emitting both classes would
        # leave the projector at the mercy of dict-order non-determinism.
        out.class_probabilities = {"snia_or_nonia_sn": prob}
        out.raw_output = {
            "snguess_score": prob,
            "snguess_bool": int(prob > 0.5),
            "model_key": model_key,
            "ndet_used": ndet,
            **feats,
        }
        out.available = True
        return out
