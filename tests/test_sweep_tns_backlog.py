"""Unit tests for WP1 TNS backlog sweep (synthetic; no live network)."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.sweep_tns_backlog import (
    OUTPUT_COLUMNS,
    ROUTE_ALERCE,
    ROUTE_LASAIR,
    authoritative_sep,
    build_xm_index,
    cone_hits_cached,
    filter_tns_candidates,
    finalize_matches,
    hits_to_matches,
    match_route_a,
    normalize_tns_name,
    tns_candidates_from_df,
)


# ------------------------------------------------------------------ #
# Name normalization                                                  #
# ------------------------------------------------------------------ #


def test_normalize_strips_prefix_whitespace_case():
    assert normalize_tns_name("SN 2024xyz") == "2024xyz"
    assert normalize_tns_name("AT2024xyz") == "2024xyz"
    assert normalize_tns_name("  2024XYZ ") == "2024xyz"
    assert normalize_tns_name("sn2024xyz") == "2024xyz"
    # only the leading AT/SN is stripped, once
    assert normalize_tns_name("2015an") == "2015an"
    assert normalize_tns_name("SN 2015an") == normalize_tns_name("2015an")


def test_normalize_empty_and_nan():
    assert normalize_tns_name(None) == ""
    assert normalize_tns_name("") == ""
    assert normalize_tns_name("nan") == ""
    assert normalize_tns_name("   ") == ""


def test_normalize_does_not_eat_non_prefix():
    # 'asassn' must not lose 'as' — only exact AT/SN leading tokens go.
    assert normalize_tns_name("ASASSN-15xyz") == "asassn-15xyz"


# ------------------------------------------------------------------ #
# Authoritative separation                                            #
# ------------------------------------------------------------------ #


def test_authoritative_sep_zero_and_known():
    assert authoritative_sep(10.0, -5.0, 10.0, -5.0) == pytest.approx(0.0)
    # 1 arcsec in dec = 1/3600 deg
    sep = authoritative_sep(10.0, -5.0, 10.0, -5.0 + 1.0 / 3600.0)
    assert sep == pytest.approx(1.0, abs=1e-6)


def test_authoritative_sep_missing_coords():
    assert authoritative_sep(None, -5.0, 10.0, -5.0) is None
    assert authoritative_sep(10.0, -5.0, float("nan"), -5.0) is None
    assert authoritative_sep(10.0, -5.0, "x", -5.0) is None


# ------------------------------------------------------------------ #
# TNS filter + candidate extraction                                   #
# ------------------------------------------------------------------ #


def _tns_df():
    return pd.DataFrame(
        {
            "name_prefix": ["SN", "AT", "SN", "SN"],
            "objname": ["2025aa", "2025bb", "2020old", "2025cc"],
            "type": ["SN Ia", "", "SN II", "SN Ic"],
            "ra": [10.0, 20.0, 30.0, 40.0],
            "declination": [-5.0, 10.0, -3.0, 50.0],  # last is dec>15
            "redshift": [0.05, None, 0.1, 0.2],
            "discoverydate": [
                "2025-07-01 00:00:00.000",
                "2025-08-01 00:00:00.000",
                "2020-01-01 00:00:00.000",  # too old
                "2025-09-01 00:00:00.000",
            ],
        }
    )


def test_filter_tns_candidates():
    out = filter_tns_candidates(_tns_df(), max_dec=15.0, since="2025-06-01")
    names = set(out["objname"])
    # 2025aa passes; 2025bb dropped (empty type); 2020old dropped (old);
    # 2025cc dropped (dec>15)
    assert names == {"2025aa"}


def test_tns_candidates_from_df_builds_full_name():
    df = _tns_df().iloc[[0]]
    cands = tns_candidates_from_df(df)
    assert cands[0]["objname"] == "2025aa"
    assert cands[0]["tns_name"] == "SN 2025aa"
    assert cands[0]["tns_type"] == "SN Ia"


# ------------------------------------------------------------------ #
# Route A xm index + matching                                         #
# ------------------------------------------------------------------ #


def test_build_xm_index_groups_by_normalized_name():
    records = [
        {"dia_id": "111", "obj_ra": 10.0, "obj_decl": -5.0, "tns_name": "SN 2025aa"},
        {"dia_id": "222", "obj_ra": 10.0, "obj_decl": -5.0, "tns_name": "AT2025aa"},
        {"dia_id": "333", "obj_ra": 99.0, "obj_decl": 0.0, "tns_name": None},
    ]
    idx = build_xm_index(records)
    assert set(idx.keys()) == {"2025aa"}
    assert {h["dia_id"] for h in idx["2025aa"]} == {"111", "222"}


def test_match_route_a_authoritative_sep_and_radius():
    cands = [
        {
            "objname": "2025aa",
            "tns_name": "SN 2025aa",
            "tns_type": "SN Ia",
            "ra": 10.0,
            "dec": -5.0,
            "tns_redshift": 0.05,
            "tns_discovery_date": "2025-07-01 00:00:00.000",
        }
    ]
    # one hit within 2", one far away (should be dropped)
    idx = {
        "2025aa": [
            {"dia_id": "111", "ra": 10.0, "dec": -5.0 + 0.5 / 3600.0},
            {"dia_id": "999", "ra": 10.0, "dec": -5.0 + 30.0 / 3600.0},
        ]
    }
    raw, matched = match_route_a(cands, idx, max_sep=2.0)
    ids = {r["object_id"] for r in raw}
    assert ids == {"111"}
    assert matched == {"2025aa"}
    assert raw[0]["route"] == ROUTE_LASAIR
    assert raw[0]["sep_arcsec"] == pytest.approx(0.5, abs=1e-6)


# ------------------------------------------------------------------ #
# Route B hit matching                                                #
# ------------------------------------------------------------------ #


def test_hits_to_matches_enforces_radius():
    cand = {
        "objname": "2025bb",
        "tns_name": "AT 2025bb",
        "tns_type": "SN II",
        "ra": 20.0,
        "dec": 3.0,
        "tns_redshift": None,
        "tns_discovery_date": "2025-08-01 00:00:00.000",
    }
    hits = [
        {"oid": "555", "meanra": 20.0, "meandec": 3.0 + 1.0 / 3600.0},
        {"oid": "666", "meanra": 20.0, "meandec": 3.0 + 10.0 / 3600.0},
    ]
    raw = hits_to_matches(cand, hits, max_sep=2.0)
    assert {r["object_id"] for r in raw} == {"555"}
    assert raw[0]["route"] == ROUTE_ALERCE


# ------------------------------------------------------------------ #
# Finalize: ternary mapping, dedup, hash_route, match_count           #
# ------------------------------------------------------------------ #


def test_finalize_maps_ternary_and_drops_unmappable():
    raw = [
        {"object_id": "1", "tns_name": "SN 2025aa", "tns_type": "SN Ia",
         "sep_arcsec": 0.4, "route": ROUTE_LASAIR, "tns_redshift": 0.05,
         "tns_discovery_date": "2025-07-01"},
        {"object_id": "2", "tns_name": "SN 2025bb", "tns_type": "SN II",
         "sep_arcsec": 0.3, "route": ROUTE_ALERCE, "tns_redshift": None,
         "tns_discovery_date": "2025-08-01"},
        {"object_id": "3", "tns_name": "SN 2025cc", "tns_type": "Totally Unknown",
         "sep_arcsec": 0.2, "route": ROUTE_LASAIR, "tns_redshift": None,
         "tns_discovery_date": "2025-09-01"},
    ]
    rows = finalize_matches(raw, max_sep=2.0)
    by_id = {r["object_id"]: r for r in rows}
    assert set(by_id) == {"1", "2"}  # unmappable type dropped
    assert by_id["1"]["final_class_ternary"] == "snia"
    assert by_id["2"]["final_class_ternary"] == "nonIa_snlike"
    assert all(r["label_quality"] == "spectroscopic" for r in rows)
    assert all(r["label_source"] == "tns_backlog_sweep" for r in rows)


def test_finalize_dedup_keeps_smallest_sep():
    raw = [
        {"object_id": "7", "tns_name": "SN X", "tns_type": "SN Ia",
         "sep_arcsec": 1.9, "route": ROUTE_LASAIR, "tns_redshift": None,
         "tns_discovery_date": None},
        {"object_id": "7", "tns_name": "SN X", "tns_type": "SN Ia",
         "sep_arcsec": 0.2, "route": ROUTE_ALERCE, "tns_redshift": None,
         "tns_discovery_date": None},
    ]
    rows = finalize_matches(raw, max_sep=2.0)
    assert len(rows) == 1
    assert rows[0]["sep_arcsec"] == pytest.approx(0.2)
    assert rows[0]["route"] == ROUTE_ALERCE


def test_finalize_hash_route_column_present_and_valid():
    raw = [
        {"object_id": "42", "tns_name": "SN Y", "tns_type": "SN Ia",
         "sep_arcsec": 0.1, "route": ROUTE_LASAIR, "tns_redshift": None,
         "tns_discovery_date": None},
    ]
    rows = finalize_matches(raw, max_sep=2.0)
    assert "hash_route" in rows[0]
    assert rows[0]["hash_route"] in {"train", "test"}
    # matches the canonical hash_route helper
    from scripts.build_truth_lsst_live import hash_route
    assert rows[0]["hash_route"] == hash_route("42")


def test_finalize_match_count_counts_redetections():
    # one TNS name -> two distinct diaObjectIds (LSST re-detections)
    raw = [
        {"object_id": "a1", "tns_name": "SN Z", "tns_type": "SN Ia",
         "sep_arcsec": 0.3, "route": ROUTE_LASAIR, "tns_redshift": None,
         "tns_discovery_date": None},
        {"object_id": "a2", "tns_name": "SN Z", "tns_type": "SN Ia",
         "sep_arcsec": 0.4, "route": ROUTE_LASAIR, "tns_redshift": None,
         "tns_discovery_date": None},
        {"object_id": "b1", "tns_name": "SN Q", "tns_type": "SN Ia",
         "sep_arcsec": 0.5, "route": ROUTE_LASAIR, "tns_redshift": None,
         "tns_discovery_date": None},
    ]
    rows = finalize_matches(raw, max_sep=2.0)
    assert len(rows) == 3  # both re-detections kept
    counts = {r["object_id"]: r["match_count"] for r in rows}
    assert counts["a1"] == 2 and counts["a2"] == 2
    assert counts["b1"] == 1


def test_output_columns_are_complete():
    raw = [
        {"object_id": "1", "tns_name": "SN A", "tns_type": "SN Ia",
         "sep_arcsec": 0.1, "route": ROUTE_LASAIR, "tns_redshift": 0.05,
         "tns_discovery_date": "2025-07-01"},
    ]
    rows = finalize_matches(raw, max_sep=2.0)
    assert set(OUTPUT_COLUMNS).issubset(set(rows[0].keys()))


# ------------------------------------------------------------------ #
# Route B cache reuse                                                 #
# ------------------------------------------------------------------ #


def test_cone_hits_cached_reuses_cache(tmp_path):
    calls = {"n": 0}

    def fetch(ra, dec, radius):
        calls["n"] += 1
        return [{"oid": "1", "meanra": ra, "meandec": dec}]

    sleeps: list[float] = []

    hits1, from_cache1 = cone_hits_cached(
        "2025aa", 10.0, -5.0, 2.0,
        cache_dir=tmp_path, fetch=fetch, sleep_fn=sleeps.append,
    )
    assert from_cache1 is False
    assert calls["n"] == 1
    assert sleeps == [0.2]  # slept once after the live call

    # second call hits the disk cache: no fetch, no sleep
    hits2, from_cache2 = cone_hits_cached(
        "2025aa", 10.0, -5.0, 2.0,
        cache_dir=tmp_path, fetch=fetch, sleep_fn=sleeps.append,
    )
    assert from_cache2 is True
    assert calls["n"] == 1  # fetch NOT called again
    assert hits1 == hits2
    assert sleeps == [0.2]  # no additional sleep
