"""Labelling rules of scripts/build_lsst_catalog_context.py (network-free: X-Match is faked)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "build_lsst_catalog_context", Path(__file__).resolve().parents[1] / "scripts/build_lsst_catalog_context.py")
bcc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bcc)


def _fake_xmatch(tables: dict[str, pd.DataFrame]):
    def xmatch(df, cat, radius_arcsec, label, chunk=20000):
        h = tables.get(cat, pd.DataFrame(columns=["object_id", "angDist"]))
        return h[h["object_id"].isin(set(df["object_id"]))].copy()
    return xmatch


def test_labels(monkeypatch) -> None:
    obj = pd.DataFrame({"object_id": ["star", "movingstar", "faintplx", "rr", "qso", "liner", "galaxy", "sn", "none"],
                        "ra": range(9), "dec": range(9)})
    gaia = pd.DataFrame({
        "object_id": ["star", "movingstar", "faintplx", "galaxy"],
        "angDist": [0.1, 0.2, 0.3, 0.4],
        "Plx": [10.0, 0.1, 1.0, 0.2], "e_Plx": [0.5, 0.5, 0.5, 0.5],       # star: 20 sigma; faintplx: 2 sigma
        "pmRA": [0.0, 30.0, 0.0, 0.1], "e_pmRA": [1.0, 1.0, 1.0, 1.0],
        "pmDE": [0.0, 0.0, 0.0, 0.1], "e_pmDE": [1.0, 1.0, 1.0, 1.0],
        "Source": [1, 2, 3, 4],
    })
    gvar = pd.DataFrame({"object_id": ["rr"], "angDist": [0.2], "Class": ["RR"]})
    simbad = pd.DataFrame({"object_id": ["qso", "liner", "galaxy", "sn"], "angDist": [0.3, 0.3, 0.5, 0.5],
                           "main_type": ["QSO", "LINER", "Galaxy", "SN"]})
    monkeypatch.setattr(bcc, "xmatch", _fake_xmatch({bcc.GAIA_CAT: gaia, bcc.GAIA_VAR_CAT: gvar, "simbad": simbad}))
    out = bcc.label_objects(obj).set_index("object_id")

    assert out.loc["star", "label_source"] == "catalog:gaia_star"
    assert out.loc["movingstar", "label_source"] == "catalog:gaia_star"        # proper motion alone suffices
    assert out.loc["rr", "label_source"] == "catalog:gaia_var:RR"
    assert out.loc["qso", "label_source"] == "catalog:simbad:QSO"
    for oid in ("star", "movingstar", "rr", "qso"):
        assert out.loc[oid, "final_class_ternary"] == "other"
        assert out.loc[oid, "label_quality"] == "context"
    # never "other": a 2-sigma parallax, a LINER nucleus, a galaxy, a SIMBAD SN, no match
    for oid in ("faintplx", "liner", "galaxy", "sn", "none"):
        assert out.loc[oid, "final_class_ternary"] is None, oid
        assert out.loc[oid, "label_source"] is None, oid
