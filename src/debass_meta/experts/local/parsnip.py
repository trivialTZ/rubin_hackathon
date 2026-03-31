"""ParSNIP local expert wrapper.

ParSNIP inference requires two artifacts:
  - ``model.pt``: the trained generative ParSNIP model
  - ``classifier.pkl``: a trained ``parsnip.Classifier`` that maps ParSNIP
    latent predictions to calibrated class probabilities

The DEBASS lightcurve cache stores ZTF detections as MJD + magnitudes. This
wrapper converts those detections into the lcdata/ParSNIP table format expected
by ``astro-parsnip``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from .base import ExpertOutput, LocalExpert

_DEFAULT_MODEL_DIR = Path("artifacts/local_experts/parsnip")
_DEFAULT_CLASSIFIER_NAMES = ("classifier.pkl", "classifier.pickle")
_FID_TO_BAND = {
    1: "ztfg",
    2: "ztfr",
    3: "ztfi",
    "1": "ztfg",
    "2": "ztfr",
    "3": "ztfi",
    "g": "ztfg",
    "r": "ztfr",
    "i": "ztfi",
    "ztfg": "ztfg",
    "ztfr": "ztfr",
    "ztfi": "ztfi",
}


def _to_builtin(value: Any) -> Any:
    """Convert numpy/scalar-ish values into plain Python types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class ParSNIPExpert(LocalExpert):
    name = "parsnip"
    semantic_type = "probability"
    CLASSES = ["SN Ia", "SN II", "SN IIn", "SN Ibc", "SN IIb", "SLSN-I"]

    def __init__(
        self,
        model_dir: Path = _DEFAULT_MODEL_DIR,
        device: str | None = None,
        threads: int = 8,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or os.environ.get("DEBASS_PARSNIP_DEVICE") or self._default_device()
        self.threads = int(os.environ.get("DEBASS_PARSNIP_THREADS", str(threads)))
        self._parsnip_available = False
        self._model = None
        self._classifier = None
        self._model_path = self._resolve_model_path()
        self._classifier_path = self._resolve_classifier_path()
        self._load_model()

    def _default_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _resolve_model_path(self) -> Path:
        env_path = os.environ.get("DEBASS_PARSNIP_MODEL")
        if env_path:
            return Path(env_path)
        return self.model_dir / "model.pt"

    def _resolve_classifier_path(self) -> Path:
        env_path = os.environ.get("DEBASS_PARSNIP_CLASSIFIER")
        if env_path:
            return Path(env_path)

        for name in _DEFAULT_CLASSIFIER_NAMES:
            path = self.model_dir / name
            if path.exists():
                return path

        return self.model_dir / _DEFAULT_CLASSIFIER_NAMES[0]

    def _load_model(self) -> None:
        try:
            import parsnip
            self._parsnip_available = True
        except ImportError:
            self._parsnip_available = False
            return

        if self._model_path.exists():
            try:
                self._model = parsnip.load_model(
                    str(self._model_path),
                    device=self.device,
                    threads=self.threads,
                )
            except Exception:
                self._model = None

        if self._model is not None and self._classifier_path.exists():
            try:
                self._classifier = parsnip.Classifier.load(str(self._classifier_path))
            except Exception:
                self._classifier = None

        class_names = getattr(self._classifier, "class_names", None)
        if class_names is not None:
            self.CLASSES = [str(_to_builtin(name)) for name in class_names]

    # ------------------------------------------------------------------ #
    # Interface                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, lightcurves: Any, labels: Any) -> None:
        if not self._parsnip_available:
            raise RuntimeError("parsnip is not installed")
        raise NotImplementedError(
            "This repository can consume trained ParSNIP artifacts for inference, "
            "but it does not yet include the ParSNIP training/export pipeline that "
            "produces model.pt and classifier.pkl."
        )

    def predict_epoch(self, object_id: str, lightcurve: Any, epoch_jd: float) -> ExpertOutput:
        truncated = self._truncate(lightcurve, epoch_jd)
        n_det = len(truncated) if isinstance(truncated, list) else 0

        if self._is_live_ready():
            probs, raw_output = self._run_parsnip(object_id, truncated)
            model_version = "loaded"
        else:
            probs = {cls: 1.0 / len(self.CLASSES) for cls in self.CLASSES}
            raw_output = {"n_det": n_det}
            model_version = "stub"

        return ExpertOutput(
            expert=self.name,
            object_id=object_id,
            epoch_jd=epoch_jd,
            class_probabilities=probs,
            raw_output=raw_output,
            semantic_type=self.semantic_type,
            model_version=model_version,
            available=self._parsnip_available,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "classes": self.CLASSES,
            "model_dir": str(self.model_dir),
            "model_path": str(self._model_path),
            "classifier_path": str(self._classifier_path),
            "available": self._parsnip_available,
            "model_loaded": self._is_live_ready(),
            "model_component_loaded": self._model is not None,
            "classifier_loaded": self._classifier is not None,
            "inference_implemented": True,
            "requires": "astro-parsnip>=1.4.0, model.pt, classifier.pkl, multi-band photometry",
            "epoch_aware": True,
            "device": self.device,
            "threads": self.threads,
            "bands": list(self._supported_bands()),
            "calibration": "classifier probabilities are only calibrated if classifier.pkl was trained and calibrated",
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _is_live_ready(self) -> bool:
        return self._parsnip_available and self._model is not None and self._classifier is not None

    def _supported_bands(self) -> list[str]:
        if self._model is None:
            return []
        try:
            return [str(band) for band in self._model.settings.get("bands", [])]
        except Exception:
            return []

    def _truncate(self, lightcurve: Any, epoch_jd: float) -> Any:
        """Return only detections at or before epoch_jd.

        Cached lightcurves use MJD; epoch_jd is passed as JD from local_infer.py.
        Handles both mjd (ALeRCE cache) and jd (ZTF alert stream) keys.
        """
        if isinstance(lightcurve, list):
            def _det_jd(d: dict) -> float:
                if "jd" in d:
                    return float(d["jd"])
                if "mjd" in d:
                    return float(d["mjd"]) + 2400000.5
                return 0.0

            return [det for det in lightcurve if _det_jd(det) <= epoch_jd]
        try:
            import pandas as pd
            if isinstance(lightcurve, pd.DataFrame):
                if "jd" in lightcurve.columns:
                    return lightcurve[lightcurve["jd"] <= epoch_jd]
                if "mjd" in lightcurve.columns:
                    return lightcurve[lightcurve["mjd"] + 2400000.5 <= epoch_jd]
        except ImportError:
            pass
        return lightcurve

    def _canonical_band(self, detection: dict[str, Any]) -> str | None:
        for key in ("band", "filter", "flt"):
            value = detection.get(key)
            if value is None:
                continue
            canonical = _FID_TO_BAND.get(str(value).strip().lower())
            if canonical is not None:
                return canonical

        fid = detection.get("fid")
        if fid is None:
            return None
        return _FID_TO_BAND.get(fid)

    def _observation_time(self, detection: dict[str, Any]) -> float | None:
        if detection.get("jd") is not None:
            try:
                return float(detection["jd"])
            except Exception:
                return None
        if detection.get("mjd") is not None:
            try:
                return float(detection["mjd"]) + 2400000.5
            except Exception:
                return None
        return None

    def _mag_to_flux(self, magnitude: float, magnitude_error: float, zeropoint: float = 25.0) -> tuple[float, float]:
        flux = 10 ** (-0.4 * (magnitude - zeropoint))
        fluxerr = flux * (math.log(10.0) / 2.5) * magnitude_error
        return flux, fluxerr

    def _iter_detections(self, lightcurve: Any) -> list[dict[str, Any]]:
        if isinstance(lightcurve, list):
            return [det for det in lightcurve if isinstance(det, dict)]
        try:
            import pandas as pd
            if isinstance(lightcurve, pd.DataFrame):
                return lightcurve.to_dict(orient="records")
        except ImportError:
            pass
        raise TypeError(f"Unsupported lightcurve type for ParSNIP: {type(lightcurve)!r}")

    def _to_parsnip_light_curve(self, object_id: str, lightcurve: Any):
        from astropy.table import Table

        rows = []
        meta: dict[str, Any] = {"object_id": object_id, "redshift": math.nan, "mwebv": math.nan}

        for detection in self._iter_detections(lightcurve):
            band = self._canonical_band(detection)
            obs_time = self._observation_time(detection)
            if band is None or obs_time is None or not math.isfinite(obs_time):
                continue

            flux = detection.get("flux")
            fluxerr = detection.get("fluxerr")
            if flux is not None and fluxerr is not None:
                try:
                    flux = float(flux)
                    fluxerr = float(fluxerr)
                except Exception:
                    flux = None
                    fluxerr = None

            if flux is None or fluxerr is None:
                magnitude = detection.get("magpsf")
                magnitude_error = detection.get("sigmapsf")
                if magnitude is None or magnitude_error is None:
                    continue
                try:
                    magnitude = float(magnitude)
                    magnitude_error = float(magnitude_error)
                except Exception:
                    continue
                if not math.isfinite(magnitude) or not math.isfinite(magnitude_error) or magnitude_error <= 0:
                    continue
                flux, fluxerr = self._mag_to_flux(magnitude, magnitude_error)

            if not math.isfinite(flux) or not math.isfinite(fluxerr) or fluxerr <= 0:
                continue

            rows.append((obs_time, flux, fluxerr, band))

            for key in ("ra", "dec", "redshift", "z", "mwebv"):
                if key in {"ra", "dec"} and key in meta:
                    try:
                        if math.isfinite(float(meta[key])):
                            continue
                    except Exception:
                        pass
                if key == "redshift":
                    try:
                        if math.isfinite(float(meta.get("redshift", math.nan))):
                            continue
                    except Exception:
                        pass
                if key == "z":
                    try:
                        if math.isfinite(float(meta.get("redshift", math.nan))):
                            continue
                    except Exception:
                        pass
                if key == "mwebv":
                    try:
                        if math.isfinite(float(meta.get("mwebv", math.nan))):
                            continue
                    except Exception:
                        pass
                value = detection.get(key)
                if value is None:
                    continue
                try:
                    value = float(value)
                except Exception:
                    continue
                if not math.isfinite(value):
                    continue
                if key == "z":
                    meta["redshift"] = value
                else:
                    meta[key] = value

        if not rows:
            raise ValueError(f"No usable observations for {object_id} after ParSNIP conversion.")

        rows.sort(key=lambda row: row[0])
        table = Table(rows=rows, names=("time", "flux", "fluxerr", "band"))
        if not math.isfinite(float(meta.get("mwebv", math.nan))):
            meta["mwebv"] = 0.0
        table.meta = meta
        return table

    def _prediction_row_to_dict(self, prediction: Any) -> dict[str, Any]:
        if isinstance(prediction, dict):
            return {str(k): _to_builtin(v) for k, v in prediction.items()}

        if isinstance(prediction, list) and prediction and isinstance(prediction[0], dict):
            return {str(k): _to_builtin(v) for k, v in prediction[0].items()}

        colnames = getattr(prediction, "colnames", None)
        if colnames:
            row = {}
            for col in colnames:
                try:
                    value = prediction[col][0]
                except Exception:
                    continue
                row[str(col)] = _to_builtin(value)
            return row

        if hasattr(prediction, "keys"):
            row = {}
            for key in prediction.keys():
                try:
                    row[str(key)] = _to_builtin(prediction[key])
                except Exception:
                    continue
            return row

        return {"repr": repr(prediction)}

    def _extract_probabilities(self, classifications: Any) -> dict[str, float]:
        row = self._prediction_row_to_dict(classifications)
        probs = {}
        for class_name in self.CLASSES:
            if class_name not in row:
                continue
            try:
                probs[class_name] = float(row[class_name])
            except Exception:
                continue
        if not probs:
            raise ValueError("ParSNIP classifier output did not contain any class probabilities.")
        return probs

    def _run_parsnip(self, object_id: str, lightcurve: Any) -> tuple[dict[str, float], dict[str, Any]]:
        if self._model is None or self._classifier is None:
            raise RuntimeError("ParSNIP model/classifier not loaded.")

        table = self._to_parsnip_light_curve(object_id, lightcurve)
        settings = getattr(self._model, "settings", {})
        predict_redshift = bool(settings.get("predict_redshift"))
        if not predict_redshift:
            redshift = table.meta.get("redshift", math.nan)
            if redshift is None or not math.isfinite(float(redshift)):
                raise ValueError(
                    "ParSNIP model requires redshift metadata, but the DEBASS lightcurve cache "
                    "does not provide one. Use a photo-z ParSNIP model or attach redshift metadata."
                )

        supported_bands = set(self._supported_bands())
        if supported_bands:
            table = table[[row["band"] in supported_bands for row in table]]
            if len(table) == 0:
                raise ValueError(
                    f"Lightcurve bands do not overlap ParSNIP model bands {sorted(supported_bands)}."
                )

        predictions = self._model.predict([table])
        classifications = self._classifier.classify(predictions)
        probs = self._extract_probabilities(classifications)
        raw_output = {
            "n_det": len(self._iter_detections(lightcurve)),
            "prediction": self._prediction_row_to_dict(predictions),
            "classification": self._prediction_row_to_dict(classifications),
            "bands": sorted({str(_to_builtin(band)) for band in table['band']}),
        }
        return probs, raw_output
