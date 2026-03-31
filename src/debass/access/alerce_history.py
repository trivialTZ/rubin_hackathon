"""ALeRCE object-snapshot history fetcher.

ALeRCE stores classifier probabilities at the *object* level (not per-alert),
so there is no way to recover "what was the probability at detection 4"
from the historical API.

Strategy:
  1. Fetch the full lightcurve to establish the per-epoch JD values.
  2. Fetch the *current* object-level classifier probabilities (latest run).
  3. Synthesise a list of EpochScore records — one per detection epoch —
     where every epoch carries the *same* latest-snapshot probabilities.

Semantic contract:
  - ``event_scope = 'object_snapshot'``
  - ``temporal_exactness = 'latest_object_unsafe'``
  These are set on every emitted field so downstream consumers can
  recognise and exclude these records from epoch-safe trust training.

For the SCC GPU path, the lightcurve truncated to N detections is also
returned so SuperNNova / ParSNIP can be re-run at each epoch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LC_DIR = Path("data/lightcurves")

# Columns to fetch from ALeRCE lightcurve endpoint
_LC_COLUMNS = "magpsf,sigmapsf,fid,mjd,jd,ra,dec,ndethist,isdiffpos"

_MJD_TO_JD_OFFSET = 2400000.5  # standard conversion


@dataclass
class EpochScore:
    """Classifier scores at a single detection epoch for one object."""
    object_id: str
    n_det: int          # number of detections seen at this epoch
    alert_jd: float     # JD of the n_det-th detection
    broker: str
    # Per-field scores at this epoch
    fields: list[dict[str, Any]] = field(default_factory=list)
    # Truncated lightcurve rows (dicts) up to and including this detection
    lightcurve_slice: list[dict[str, Any]] = field(default_factory=list)


class AlerceHistoryFetcher:
    """Fetch per-epoch scores and lightcurve slices from ALeRCE."""

    def __init__(self, lc_dir: Path = _LC_DIR) -> None:
        self.lc_dir = Path(lc_dir)
        try:
            from alerce.core import Alerce
            self._client = Alerce()
        except ImportError:
            self._client = None

    def fetch_epoch_scores(
        self,
        object_id: str,
        max_epochs: int = 20,
    ) -> list[EpochScore]:
        """Return EpochScore list for each detection epoch up to max_epochs.

        Since ALeRCE probabilities are object-level (not per-alert), the same
        latest object snapshot is attached to every epoch. This is useful for
        SCC lightcurve slicing, but the emitted fields remain explicitly marked
        ``latest_object_unsafe`` so downstream trust training can exclude them.
        """
        lc = self._fetch_lightcurve(object_id)
        if not lc:
            return []

        probs = self._fetch_probabilities(object_id)

        # Sort detections by MJD/JD (ALeRCE uses mjd)
        detections = sorted(
            [r for r in lc if r.get("isdiffpos") in ("t", "1", 1, True, "true")],
            key=lambda r: r.get("mjd") or r.get("jd") or 0
        )
        if not detections:
            detections = sorted(lc, key=lambda r: r.get("mjd") or r.get("jd") or 0)

        epochs: list[EpochScore] = []
        for i, det in enumerate(detections[:max_epochs]):
            n_det = i + 1
            alert_jd = self._alert_jd(det)
            lc_slice = detections[: n_det]

            # Attach the latest object snapshot to every epoch with explicit
            # unsafe temporal semantics.
            epoch_fields = [
                {
                    **f,
                    "n_det": n_det,
                    "alert_jd": alert_jd,
                    "event_scope": "object_snapshot",
                    "temporal_exactness": "latest_object_unsafe",
                    "event_time_jd": None,
                }
                for f in probs
            ]

            epochs.append(EpochScore(
                object_id=object_id,
                n_det=n_det,
                alert_jd=alert_jd,
                broker="alerce",
                fields=epoch_fields,
                lightcurve_slice=lc_slice,
            ))

        # Cache lightcurve to disk for SCC GPU re-scoring
        self._cache_lightcurve(object_id, detections)
        return epochs

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _alert_jd(self, record: dict[str, Any]) -> float:
        if record.get("jd") is not None:
            return float(record["jd"])
        if record.get("mjd") is not None:
            return float(record["mjd"]) + _MJD_TO_JD_OFFSET
        return 0.0

    def _fetch_lightcurve(self, object_id: str) -> list[dict]:
        cache = self.lc_dir / f"{object_id}.json"
        if cache.exists():
            import json
            with open(cache) as fh:
                return json.load(fh)
        if self._client is None:
            return []
        try:
            df = self._client.query_lightcurve(oid=object_id, format="pandas")
            # ALeRCE returns a single-row DataFrame with a 'detections' column
            # containing a list of dicts
            if "detections" in df.columns:
                records = df["detections"].iloc[0]
                if not isinstance(records, list):
                    records = []
            else:
                records = df.to_dict(orient="records")
            self._cache_lightcurve(object_id, records)
            return records
        except Exception:
            return []

    def _fetch_probabilities(self, object_id: str) -> list[dict]:
        """Return per-field prob dicts (same format as AlerceAdapter.fields)."""
        if self._client is None:
            return []
        try:
            df = self._client.query_probabilities(oid=object_id, format="pandas")
            fields = []
            for _, row in df.iterrows():
                cls_name = row.get("class_name", "unknown")
                prob_val = row.get("probability")
                classifier = row.get("classifier_name", "unknown")
                fields.append({
                    "field": f"{classifier}_{cls_name}",
                    "raw_label_or_score": prob_val,
                    "semantic_type": "probability",
                    "canonical_projection": float(prob_val) if prob_val is not None else None,
                    "classifier": classifier,
                    "class_name": cls_name,
                    "class": cls_name,
                    "event_scope": "object_snapshot",
                    "temporal_exactness": "latest_object_unsafe",
                    "event_time_jd": None,
                })
            return fields
        except Exception:
            return []

    def _cache_lightcurve(self, object_id: str, records: list[dict]) -> None:
        import json
        self.lc_dir.mkdir(parents=True, exist_ok=True)
        path = self.lc_dir / f"{object_id}.json"
        # Convert non-serialisable types
        clean = []
        for r in records:
            clean.append({k: (float(v) if hasattr(v, 'item') else v) for k, v in r.items()})
        with open(path, "w") as fh:
            json.dump(clean, fh)
