"""Unit tests for the fusion_v11 P1 truth/label engine (synthetic; no network)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.build_truth_lsst_live import (
    aggregate_fink_xm,
    build_manifest,
    catalog_negative_rows,
    clean_cohort_truth,
    cohort_manifest_test_ids,
    discovery_date_to_mjd,
    hash_route,
    in_discovery_window,
    make_catalog_negative_row,
    make_truth_row,
    map_fink_xm,
    merge_live_into_truth,
    resolve_label_quality,
    sep_ok,
    truth_rows_to_df,
    window_position,
    TRUTH_COLUMNS,
)
from scripts.download_tns_bulk import upsert_tns_bulk
from scripts.harvest_ztf_lsst_associations import (
    ASSOCIATION_COLUMNS,
    parse_ztf_ids,
    record_to_association_rows,
    records_to_spec_rows,
)
from scripts.rederive_spec_truth import (
    DELTA_COLUMNS,
    build_tns_type_lookups,
    is_bts_untyped,
    rederive,
    rederive_label,
    resolve_tns_type,
)


# ------------------------------------------------------------------ #
# Epoch-window stale demotion                                         #
# ------------------------------------------------------------------ #


def test_epoch_window_inside_and_outside():
    # baseline [61000, 61200]; window [60910, 61230]
    assert in_discovery_window(61050.0, 61000.0, 61200.0) is True
    assert in_discovery_window(60915.0, 61000.0, 61200.0) is True   # within 90d pre
    assert in_discovery_window(60800.0, 61000.0, 61200.0) is False  # too early -> stale
    assert in_discovery_window(61300.0, 61000.0, 61200.0) is False  # too late -> stale
    # unknown MJDs cannot prove staleness
    assert in_discovery_window(None, 61000.0, 61200.0) is True
    assert in_discovery_window(61050.0, None, None) is True


def test_resolve_label_quality_demotion():
    # typed + in window -> spectroscopic
    assert resolve_label_quality("SN Ia", "2025abc", 61050.0, 61000.0, 61200.0) == "spectroscopic"
    # typed but discovery is 200d before first det -> stale
    assert resolve_label_quality("SN Ia", "2025abc", 60700.0, 61000.0, 61200.0) == "stale_xmatch"
    # name only, in window -> tns_untyped
    assert resolve_label_quality(None, "2025abc", 61050.0, 61000.0, 61200.0) == "tns_untyped"
    # ambiguous generic SN -> tns_ambiguous
    assert resolve_label_quality("SN", "2025abc", 61050.0, 61000.0, 61200.0) == "tns_ambiguous"
    # nothing -> None
    assert resolve_label_quality(None, None, None, None, None) is None


def test_window_position_and_tail_demotion():
    # baseline [61000, 61200]; window [60910, 61230]
    assert window_position(61050.0, 61000.0, 61200.0) == "in"
    assert window_position(60700.0, 61000.0, 61200.0) == "before"   # stale
    assert window_position(61300.0, 61000.0, 61200.0) == "after"    # tail
    assert window_position(None, 61000.0, 61200.0) == "in"          # unknown -> in
    # discovery AFTER the LSST detections -> tail_xmatch (late same-transient)
    assert resolve_label_quality("SN Ia", "2026aok", 61300.0, 61000.0, 61200.0) == "tail_xmatch"


def test_make_truth_row_tail_demotion_no_ternary():
    df = truth_rows_to_df([
        make_truth_row(object_id="7", tns_name="2026aok", tns_type="SN Ia",
                       discovery_date="2026-05-15 00:00:00",  # ~+130d after baseline
                       first_mjd=60940.0, last_mjd=60980.0),
    ])
    r = df.iloc[0]
    assert r["label_quality"] == "tail_xmatch"
    assert r["final_class_ternary"] is None  # excluded from train AND eval
    assert r["follow_proxy"] == 0


# ------------------------------------------------------------------ #
# Fink xm max-MJD cross-alert aggregation                             #
# ------------------------------------------------------------------ #


def test_aggregate_fink_xm_max_mjd_non_nan():
    alerts = [
        {"i:midpointMjdTai": 61000.0, "f:xm_tns_type": "SN Ia"},
        {"i:midpointMjdTai": 61010.0, "f:xm_tns_type": "Fail"},   # latest, but missing
        {"i:midpointMjdTai": 61005.0, "f:xm_tns_type": "SN Ib"},  # newest non-nan
        {"i:midpointMjdTai": 60990.0, "f:xm_tns_type": "nan"},
    ]
    # picks the max-MJD alert whose value is non-nan/non-Fail (61005 -> SN Ib)
    assert aggregate_fink_xm(alerts, "f:xm_tns_type") == "SN Ib"
    # all missing -> None
    allbad = [{"i:midpointMjdTai": 1.0, "f:xm_tns_type": "Fail"},
              {"i:midpointMjdTai": 2.0, "f:xm_tns_type": "nan"}]
    assert aggregate_fink_xm(allbad, "f:xm_tns_type") is None
    # empty -> None
    assert aggregate_fink_xm([], "f:xm_tns_type") is None


# ------------------------------------------------------------------ #
# Catalog negatives (B5)                                              #
# ------------------------------------------------------------------ #


def test_catalog_negative_rows_frozen_exclusion_and_dedup():
    records = [
        {"diaObjectId": "1", "classification": "VS"},
        {"diaObjectId": "2", "classification": "AGN"},
        {"diaObjectId": "3", "classification": "SN"},     # transient -> not a negative
        {"diaObjectId": "4", "classification": "ORPHAN"},  # not a negative
        {"diaObjectId": "5", "classification": "VS"},
        {"diaObjectId": "5", "classification": "VS"},      # dup
    ]
    rows = catalog_negative_rows(records, frozen_ids={"2"})  # 2 is benchmark-only
    ids = {r["object_id"] for r in rows}
    assert ids == {"1", "5"}  # 2 excluded, 3/4 not negatives, 5 deduped
    for r in rows:
        assert r["final_class_ternary"] == "other"
        assert r["label_quality"] == "context"


def test_discovery_date_to_mjd_roundtrip():
    mjd = discovery_date_to_mjd("2025-10-01 00:00:00")
    assert mjd is not None
    assert 60900 < mjd < 61100  # 2025-10-01 is MJD ~60949
    assert discovery_date_to_mjd(None) is None
    assert discovery_date_to_mjd("nan") is None


# ------------------------------------------------------------------ #
# 2-arcsec cut                                                        #
# ------------------------------------------------------------------ #


def test_sep_two_arcsec_cut():
    assert sep_ok(0.28) is True
    assert sep_ok(2.0) is True
    assert sep_ok(2.97) is False
    assert sep_ok(None) is True  # missing sep is not a rejection here


# ------------------------------------------------------------------ #
# Fink "Fail" sentinel                                                #
# ------------------------------------------------------------------ #


def test_fink_fail_mapping():
    assert map_fink_xm("Fail") is None
    assert map_fink_xm("") is None
    assert map_fink_xm(None) is None
    assert map_fink_xm("nan") is None
    assert map_fink_xm("SN Ia") == "SN Ia"
    assert map_fink_xm(0.87) == 0.87


# ------------------------------------------------------------------ #
# Manifest schema                                                     #
# ------------------------------------------------------------------ #


def test_manifest_schema_and_dedup():
    m = build_manifest(
        ["100", "200", "100", " 300 "],
        frozen_utc="2026-07-05T00:00:00Z",
        policy="hash",
        source="unit-test",
    )
    assert set(m.keys()) == {"test_ids", "frozen_utc", "policy", "source"}
    assert m["test_ids"] == ["100", "200", "300"]  # dedup, order-stable, trimmed
    assert m["frozen_utc"] == "2026-07-05T00:00:00Z"
    # JSON-serialisable
    json.loads(json.dumps(m))


# ------------------------------------------------------------------ #
# Hash routing determinism                                           #
# ------------------------------------------------------------------ #


def test_hash_routing_deterministic_and_split():
    ids = [str(i) for i in range(2000)]
    routes = {i: hash_route(i) for i in ids}
    # deterministic
    assert all(hash_route(i) == routes[i] for i in ids)
    # both buckets populated, roughly balanced
    n_test = sum(1 for v in routes.values() if v == "test")
    assert 700 < n_test < 1300
    assert set(routes.values()) == {"test", "train"}
    # whitespace-insensitive (frozen ids never move)
    assert hash_route(" 42 ") == hash_route("42")


# ------------------------------------------------------------------ #
# TNS diff upsert                                                     #
# ------------------------------------------------------------------ #


def test_diff_upsert_keeps_newest_lastmodified():
    master = pd.DataFrame({
        "objid": [1, 2, 3],
        "type": ["SN Ia", None, "SN II"],
        "lastmodified": ["2025-01-01 00:00:00", "2025-01-01 00:00:00", "2025-01-01 00:00:00"],
    })
    diff = pd.DataFrame({
        "objid": [2, 4],
        "type": ["SN Ic", "AGN"],
        "lastmodified": ["2025-06-01 00:00:00", "2025-06-01 00:00:00"],
    })
    out = upsert_tns_bulk(master, diff)
    assert set(out["objid"]) == {1, 2, 3, 4}
    # objid 2 upgraded to the newer diff row
    assert out.loc[out["objid"] == 2, "type"].iloc[0] == "SN Ic"
    # objid 1/3 preserved
    assert out.loc[out["objid"] == 1, "type"].iloc[0] == "SN Ia"


def test_diff_upsert_stale_diff_does_not_override_newer_master():
    master = pd.DataFrame({
        "objid": [5],
        "type": ["SN Ia"],
        "lastmodified": ["2025-06-01 00:00:00"],
    })
    stale_diff = pd.DataFrame({
        "objid": [5],
        "type": ["Unknown"],
        "lastmodified": ["2025-01-01 00:00:00"],
    })
    out = upsert_tns_bulk(master, stale_diff)
    assert len(out) == 1
    assert out.loc[out["objid"] == 5, "type"].iloc[0] == "SN Ia"


# ------------------------------------------------------------------ #
# Association harvest (inverted / disc_int_name parsing)              #
# ------------------------------------------------------------------ #


def test_parse_ztf_ids():
    assert parse_ztf_ids("ZTF25aaa;ATLAS25x;ZTF25bbb") == ["ZTF25aaa", "ZTF25bbb"]
    assert parse_ztf_ids("ZTF25aaa, ZTF25aaa") == ["ZTF25aaa"]  # dedup
    assert parse_ztf_ids(None) == []
    assert parse_ztf_ids("nan") == []
    assert parse_ztf_ids("Gaia25x;ATLAS25y") == []


def test_record_to_association_rows_schema():
    rec = {
        "diaObjectId": "170028485847285898",
        "ra": 59.34, "decl": -48.20,
        "crossmatch_tns.ra": 59.3401, "crossmatch_tns.decl": -48.2001,
        "disc_int_name": "ZTF25abc;ZTF25xyz",
        "type": "SN Ia", "tns_name": "2025abc",
    }
    rows = record_to_association_rows(rec)
    assert len(rows) == 2
    for r in rows:
        assert set(ASSOCIATION_COLUMNS).issubset(r.keys())
        assert r["lsst_object_id"] == "170028485847285898"
        assert r["match_status"] == "matched"
        assert r["association_source"] == "lasair_crossmatch_tns"
        assert r["match_count"] == 2
        assert r["sep_arcsec"] is not None and r["sep_arcsec"] < 2.0
    assert {r["ztf_object_id"] for r in rows} == {"ZTF25abc", "ZTF25xyz"}


def test_record_to_association_rows_no_ztf_id():
    rec = {"diaObjectId": "1", "disc_int_name": "ATLAS25x"}
    assert record_to_association_rows(rec) == []


def test_records_to_spec_rows_label_source():
    recs = [
        {"diaObjectId": "1", "type": "SN Ia", "tns_name": "2025a", "disc_int_name": "ZTF25a"},
        {"diaObjectId": "2", "type": None, "tns_name": "2025b"},  # untyped -> dropped
    ]
    spec = records_to_spec_rows(recs)
    assert len(spec) == 1
    assert spec[0]["label_source"] == "ztf_assoc_spec"
    assert spec[0]["final_class_ternary"] == "snia"
    assert spec[0]["ztf_object_id"] == "ZTF25a"


# ------------------------------------------------------------------ #
# Truth schema / dtypes                                              #
# ------------------------------------------------------------------ #


def test_truth_rows_to_df_schema_and_dtypes():
    rows = [
        make_truth_row(object_id="1", tns_name="2025a", tns_type="SN Ia",
                       discovery_date="2025-10-05 00:00:00",
                       first_mjd=60940.0, last_mjd=61000.0),
        make_catalog_negative_row(object_id="2", basis="gaia_plx"),
    ]
    df = truth_rows_to_df(rows)
    assert list(df.columns) == TRUTH_COLUMNS
    assert df["follow_proxy"].dtype == "int64"
    assert df["tns_redshift"].dtype == "float64"
    assert df["truth_timestamp"].dtype == "float64"
    # SN Ia in-window -> snia/spectroscopic/follow_proxy=1
    r0 = df[df["object_id"] == "1"].iloc[0]
    assert r0["final_class_ternary"] == "snia"
    assert r0["label_quality"] == "spectroscopic"
    assert r0["follow_proxy"] == 1
    # catalog negative
    r1 = df[df["object_id"] == "2"].iloc[0]
    assert r1["final_class_ternary"] == "other"
    assert r1["label_quality"] == "context"


def test_make_truth_row_stale_demotion_no_ternary():
    df = truth_rows_to_df([
        make_truth_row(object_id="9", tns_name="2020old", tns_type="SN Ia",
                       discovery_date="2020-01-01 00:00:00",
                       first_mjd=61000.0, last_mjd=61200.0),
    ])
    r = df.iloc[0]
    assert r["label_quality"] == "stale_xmatch"
    assert r["final_class_ternary"] is None  # excluded from train AND eval
    assert r["follow_proxy"] == 0


# ------------------------------------------------------------------ #
# Cohort re-derivation + manifest test_ids                            #
# ------------------------------------------------------------------ #


def test_clean_cohort_truth_and_manifest():
    cohort = pd.DataFrame({
        "object_id": ["100", "200", "300", "400"],
        "tns_name": ["2025a", "2025b", "2020old", "2025d"],
        "tns_type": [None, None, None, None],
        "first_det_mjd": [60940.0, 60940.0, 61000.0, 60940.0],
        "last_det_mjd": [61000.0, 61000.0, 61200.0, 61000.0],
        "cohort": ["transients", "others", "transients", "transients"],
        "label_basis": [None, "simbad", None, None],
    })
    tns_auth = {
        "2025a": {"type": "SN Ia", "discoverydate": "2025-10-05 00:00:00", "redshift": 0.05},
        "2025b": {"type": "AGN", "discoverydate": "2025-10-05 00:00:00", "redshift": None},
        # 2020old: typed but discovery long before baseline -> stale
        "2020old": {"type": "SN II", "discoverydate": "2020-01-01 00:00:00", "redshift": None},
    }
    cleaned = clean_cohort_truth(cohort_df=cohort, tns_auth=tns_auth, authoritative_truth=None)
    assert len(cleaned) == len(cohort)  # row-per-object parity (RUNBOOK)
    q = dict(zip(cleaned["object_id"], cleaned["label_quality"]))
    assert q["100"] == "spectroscopic"   # SN Ia in window
    assert q["200"] == "context"          # others -> catalog negative
    assert q["300"] == "stale_xmatch"     # 2020 disc vs 2025 baseline
    assert q["400"] == "tns_untyped"      # name only, no type

    test_ids = cohort_manifest_test_ids(cleaned)
    assert set(test_ids) == {"100", "200"}  # spec survivor + catalog other
    assert "300" not in test_ids  # stale excluded
    assert "400" not in test_ids  # untyped -> SSL only


# ------------------------------------------------------------------ #
# B0 rederive: untyped never spectroscopic; bts_untyped tier; delta   #
# ------------------------------------------------------------------ #


def test_is_bts_untyped():
    for t in ("-", "", None, "nan", "None"):
        assert is_bts_untyped(t) is True
    for t in ("SN Ia", "SN II", "CV"):
        assert is_bts_untyped(t) is False


def test_rederive_label_untyped_never_spectroscopic():
    # BTS-untyped filler, no TNS type -> demote to bts_untyped (NOT spectroscopic)
    res = rederive_label(bts_type="-", existing_tns_type=None, resolved_tns_type=None,
                         orig_quality="spectroscopic", orig_ternary="nonIa_snlike")
    assert res["label_quality"] == "bts_untyped"
    assert res["final_class_ternary"] is None
    assert res["changed"] is True
    assert res["reason"] == "bts_untyped_demote"


def test_rederive_label_tns_corrected_subtype():
    # BTS-untyped filler, TNS resolves SN Ia -> corrected snia, stays spectroscopic
    res = rederive_label(bts_type="-", existing_tns_type=None, resolved_tns_type="SN Ia",
                         orig_quality="spectroscopic", orig_ternary="nonIa_snlike")
    assert res["label_quality"] == "spectroscopic"
    assert res["final_class_ternary"] == "snia"
    assert res["resolved_tns_type"] == "SN Ia"
    assert res["changed"] is True
    assert res["reason"] == "tns_corrected"
    # TNS resolves a non-SN (CV) -> corrected to other, still typed
    res_cv = rederive_label(bts_type="-", existing_tns_type=None, resolved_tns_type="CV",
                            orig_quality="spectroscopic", orig_ternary="nonIa_snlike")
    assert res_cv["final_class_ternary"] == "other"
    assert res_cv["reason"] == "tns_corrected"


def test_rederive_label_ambiguous_sn_demotes():
    # generic "SN" is ambiguous -> must NOT become spectroscopic+subtype
    res = rederive_label(bts_type="-", existing_tns_type=None, resolved_tns_type="SN",
                         orig_quality="spectroscopic", orig_ternary="nonIa_snlike")
    assert res["label_quality"] == "bts_untyped"
    assert res["final_class_ternary"] is None


def test_rederive_label_typed_rows_unchanged():
    # concrete BTS type -> unchanged
    res = rederive_label(bts_type="SN II", existing_tns_type=None, resolved_tns_type="SN Ia",
                         orig_quality="spectroscopic", orig_ternary="nonIa_snlike")
    assert res["changed"] is False
    # already-typed via TNS -> unchanged even if bts_type is '-'
    res2 = rederive_label(bts_type="-", existing_tns_type="SN Ia", resolved_tns_type=None,
                          orig_quality="spectroscopic", orig_ternary="snia")
    assert res2["changed"] is False
    # weak rows untouched
    res3 = rederive_label(bts_type="-", existing_tns_type=None, resolved_tns_type=None,
                          orig_quality="weak", orig_ternary="other")
    assert res3["changed"] is False


def test_build_tns_type_lookups_and_resolve():
    tns_df = pd.DataFrame({
        "objname": ["2025abc", "2025xyz"],
        "internal_names": ["ZTF25aaa;ATLAS25x", "ZTF25bbb"],
        "type": ["SN Ia", None],
    })
    ztf_map, name_map = build_tns_type_lookups(tns_df)
    assert ztf_map["ZTF25aaa"] == "SN Ia"
    assert name_map["2025abc"] == "SN Ia"
    # resolve via ZTF id
    assert resolve_tns_type("ZTF25aaa", None, ztf_map, name_map) == "SN Ia"
    # resolve via TNS name when ZTF id absent
    assert resolve_tns_type("ZTF99zzz", "SN 2025abc", ztf_map, name_map) == "SN Ia"
    # untyped TNS record -> None
    assert resolve_tns_type("ZTF25bbb", "2025xyz", ztf_map, name_map) is None


def test_rederive_end_to_end_and_delta():
    truth = pd.DataFrame({
        "object_id": ["ZTF25aaa", "ZTF25ttt", "ZTF25bbb", "ZTF25www"],
        "final_class_ternary": ["nonIa_snlike", "snia", "nonIa_snlike", "other"],
        "follow_proxy": [0, 1, 0, 0],
        "label_source": ["ztf_bts"] * 4,
        "label_quality": ["spectroscopic", "spectroscopic", "spectroscopic", "weak"],
        "bts_type": ["-", "SN Ia", "-", "-"],
        "tns_name": ["2025abc", "2025typed", "2025none", None],
        "redshift": [None] * 4,
        "final_class_raw": [None] * 4,
        "truth_timestamp": [0.0] * 4,
        "tns_prefix": [None] * 4,
        "tns_type": [None, None, None, None],
        "tns_has_spectra": [False] * 4,
        "tns_redshift": [float("nan")] * 4,
        "tns_ra": [float("nan")] * 4,
        "tns_dec": [float("nan")] * 4,
        "tns_discovery_date": [None] * 4,
        "consensus_experts": [float("nan")] * 4,
        "consensus_n_agree": [float("nan")] * 4,
        "consensus_n_total": [float("nan")] * 4,
    })
    tns_df = pd.DataFrame({
        "objname": ["2025abc"],
        "internal_names": ["ZTF25aaa"],
        "type": ["SN Ia"],
    })
    corrected, delta = rederive(
        truth_df=truth, tns_df=tns_df, bts_map={}, locked_test_ids={"ZTF25aaa"},
    )
    by_id = corrected.set_index("object_id")
    # untyped filler resolved by TNS -> snia, spectroscopic, tns_type stamped
    assert by_id.loc["ZTF25aaa", "final_class_ternary"] == "snia"
    assert by_id.loc["ZTF25aaa", "label_quality"] == "spectroscopic"
    assert by_id.loc["ZTF25aaa", "tns_type"] == "SN Ia"
    assert by_id.loc["ZTF25aaa", "follow_proxy"] == 1
    # concrete BTS type -> untouched
    assert by_id.loc["ZTF25ttt", "label_quality"] == "spectroscopic"
    # untyped filler, unresolved -> bts_untyped tier, ternary None
    assert by_id.loc["ZTF25bbb", "label_quality"] == "bts_untyped"
    assert by_id.loc["ZTF25bbb", "final_class_ternary"] is None
    # weak row untouched
    assert by_id.loc["ZTF25www", "label_quality"] == "weak"
    # follow_proxy keeps the pinned int dtype
    assert corrected["follow_proxy"].dtype == "int64"
    # delta emitted with locked-test flag
    assert list(delta.columns) == DELTA_COLUMNS
    assert set(delta["object_id"]) == {"ZTF25aaa", "ZTF25bbb"}
    d_aaa = delta.set_index("object_id").loc["ZTF25aaa"]
    assert bool(d_aaa["in_locked_test"]) is True
    assert d_aaa["reason"] == "tns_corrected"


# ------------------------------------------------------------------ #
# TRUTH PLUMBING PIN: merge lsst_live rows into object_truth_v11       #
# ------------------------------------------------------------------ #


def test_merge_live_into_truth_live_rows_win_and_carry_demoted():
    base = truth_rows_to_df([
        make_truth_row(object_id="ZTF1", tns_name="2025a", tns_type="SN Ia",
                       discovery_date="2025-10-05 00:00:00",
                       first_mjd=60940.0, last_mjd=61000.0),
        make_truth_row(object_id="LSST_OLD", tns_name="2025x", tns_type="SN II",
                       discovery_date="2025-10-05 00:00:00",
                       first_mjd=60940.0, last_mjd=61000.0),
    ])
    live = truth_rows_to_df([
        # LSST-live row for the same id -> must WIN over the base row
        make_truth_row(object_id="LSST_OLD", tns_name="2025x", tns_type="SN Ia",
                       discovery_date="2025-10-05 00:00:00",
                       first_mjd=60940.0, last_mjd=61000.0),
        # a demoted stale row -> carried through with its demoted quality
        make_truth_row(object_id="LSST_STALE", tns_name="2020old", tns_type="SN Ia",
                       discovery_date="2020-01-01 00:00:00",
                       first_mjd=61000.0, last_mjd=61200.0),
    ])
    merged = merge_live_into_truth(base, live)
    assert list(merged.columns) == TRUTH_COLUMNS
    by_id = merged.set_index("object_id")
    # ZTF-only base row preserved
    assert by_id.loc["ZTF1", "label_quality"] == "spectroscopic"
    # live row wins for its id (type flipped SN II -> SN Ia)
    assert by_id.loc["LSST_OLD", "final_class_ternary"] == "snia"
    # no duplicate ids
    assert merged["object_id"].is_unique
    # demoted stale row carried with its quality (builder excludes it later)
    assert by_id.loc["LSST_STALE", "label_quality"] == "stale_xmatch"


def test_merge_live_into_truth_empty_live_is_base():
    base = truth_rows_to_df([
        make_truth_row(object_id="ZTF1", tns_name="2025a", tns_type="SN Ia",
                       discovery_date="2025-10-05 00:00:00",
                       first_mjd=60940.0, last_mjd=61000.0),
    ])
    empty = truth_rows_to_df([])
    merged = merge_live_into_truth(base, empty)
    assert len(merged) == 1
    assert list(merged.columns) == TRUTH_COLUMNS


def test_rederive_no_tns_all_untyped_demote():
    # No TNS bulk: every untyped filler spec row demotes to bts_untyped.
    truth = pd.DataFrame({
        "object_id": ["ZTF1"],
        "final_class_ternary": ["nonIa_snlike"],
        "follow_proxy": [0],
        "label_source": ["ztf_bts"],
        "label_quality": ["spectroscopic"],
        "bts_type": ["-"],
        "tns_name": ["2025abc"],
        "redshift": [None], "final_class_raw": [None], "truth_timestamp": [0.0],
        "tns_prefix": [None], "tns_type": [None], "tns_has_spectra": [False],
        "tns_redshift": [float("nan")], "tns_ra": [float("nan")], "tns_dec": [float("nan")],
        "tns_discovery_date": [None], "consensus_experts": [float("nan")],
        "consensus_n_agree": [float("nan")], "consensus_n_total": [float("nan")],
    })
    corrected, delta = rederive(truth_df=truth, tns_df=None)
    assert corrected.iloc[0]["label_quality"] == "bts_untyped"
    assert len(delta) == 1
    assert delta.iloc[0]["reason"] == "bts_untyped_demote"


# ------------------------------------------------------------------ #
# sep_arcsec fix (2026-07-05 smoke defect): aliased SELECT + TNS-bulk #
# authoritative separation                                            #
# ------------------------------------------------------------------ #


def test_harvest_selected_aliases_colliding_coord_columns():
    """Lasair /query/ flattens qualified names to bare keys, so objects.ra and
    crossmatch_tns.ra collide unless aliased — the ≤2″ cut is a silent no-op
    without these aliases (2026-07-05 smoke finding)."""
    from scripts.harvest_ztf_lsst_associations import _SELECTED

    for alias in ("AS obj_ra", "AS obj_decl", "AS cm_ra", "AS cm_decl"):
        assert alias in _SELECTED
    # the bare colliding forms must not remain un-aliased
    assert "objects.ra," not in _SELECTED
    assert "crossmatch_tns.ra," not in _SELECTED


def test_record_sep_arcsec_reads_aliased_keys():
    from scripts.harvest_ztf_lsst_associations import _record_sep_arcsec

    rec = {"obj_ra": 52.8238467, "obj_decl": -29.6916002,
           "cm_ra": 52.823846, "cm_decl": -29.6916}
    sep = _record_sep_arcsec(rec)
    assert sep is not None and 0.0 <= sep < 0.02  # sub-arcsec match


def test_auth_sep_prefers_tns_bulk_coordinates():
    from scripts.build_truth_lsst_live import _auth_sep_arcsec

    rec = {"obj_ra": 150.0, "obj_decl": 2.0}
    # ~3.6" offset in ra at dec=2 (1e-3 deg * cos(dec) * 3600)
    auth = {"ra": 150.001, "declination": 2.0}
    sep = _auth_sep_arcsec(rec, auth)
    assert sep is not None and 3.0 < sep < 4.0
    # missing auth coords -> None (caller falls back to the Lasair sep)
    assert _auth_sep_arcsec(rec, {"ra": None, "declination": None}) is None
    assert _auth_sep_arcsec(rec, None) is None


# ------------------------------------------------------------------ #
# Duplicate-object_id dedup (2026-07-05 SCC build crash: merged truth #
# must have unique ids for gold's _load_truth_lookup)                 #
# ------------------------------------------------------------------ #


def _mk_truth(rows):
    return truth_rows_to_df([make_truth_row(**r) for r in rows])


def test_dedupe_truth_frame_quality_priority():
    from scripts.build_truth_lsst_live import dedupe_truth_frame

    df = pd.concat([
        _mk_truth([dict(object_id="A", tns_name="2026aa", tns_type="SN Ia",
                        discovery_date="2026-06-01 00:00:00",
                        first_mjd=61190.0, last_mjd=61220.0)]),  # spectroscopic
        _mk_truth([dict(object_id="A", tns_name=None, tns_type=None,
                        discovery_date=None, first_mjd=61190.0,
                        last_mjd=61220.0)]),                      # unlabeled dup
        _mk_truth([dict(object_id="B", tns_name="2026bb", tns_type=None,
                        discovery_date="2026-06-01 00:00:00",
                        first_mjd=61190.0, last_mjd=61220.0)]),   # tns_untyped
    ], ignore_index=True)
    # a context dup of B (catalog negative colliding with the seed row)
    dup_b = df.iloc[[2]].copy()
    dup_b["label_quality"] = "context"
    dup_b["final_class_ternary"] = "other"
    df = pd.concat([df, dup_b], ignore_index=True)

    out = dedupe_truth_frame(df)
    assert out["object_id"].is_unique and len(out) == 2
    a = out[out.object_id == "A"].iloc[0]
    assert a["label_quality"] == "spectroscopic"    # spec beats unlabeled
    b = out[out.object_id == "B"].iloc[0]
    assert b["label_quality"] == "tns_untyped"      # untyped beats context


def test_dedupe_truth_frame_noop_on_unique():
    from scripts.build_truth_lsst_live import dedupe_truth_frame

    df = _mk_truth([
        dict(object_id="X", tns_name="2026xx", tns_type="SN II",
             discovery_date="2026-06-01 00:00:00", first_mjd=61190.0,
             last_mjd=61220.0),
    ])
    out = dedupe_truth_frame(df)
    assert out is df  # unique frame passes through untouched


def test_merge_live_into_truth_dedupes_and_asserts_unique():
    live = pd.concat([
        _mk_truth([dict(object_id="L1", tns_name="2026cc", tns_type="SN Ia",
                        discovery_date="2026-06-01 00:00:00",
                        first_mjd=61190.0, last_mjd=61220.0)]),
        _mk_truth([dict(object_id="L1", tns_name=None, tns_type=None,
                        discovery_date=None, first_mjd=None, last_mjd=None)]),
    ], ignore_index=True)
    base = _mk_truth([
        dict(object_id="Z1", tns_name="2025zz", tns_type="SN II",
             discovery_date="2025-06-01 00:00:00", first_mjd=60800.0,
             last_mjd=60830.0),
    ])
    merged = merge_live_into_truth(base, live)
    assert merged["object_id"].is_unique
    assert set(merged["object_id"]) == {"Z1", "L1"}
    l1 = merged[merged.object_id == "L1"].iloc[0]
    assert l1["label_quality"] == "spectroscopic"


def test_build_manifest_excluded_ids_never_reenter():
    """G6 counterpart integrity: excluded ids are the sanctioned removal path —
    they never appear in test_ids (even via prior/new lists) and are carried
    forward append-only."""
    m1 = build_manifest(
        ["A", "B", "C"], policy="p", source="s",
        excluded_ids={"B": "ztf_counterpart_in_locked_train"},
    )
    assert m1["test_ids"] == ["A", "C"]
    assert "B" in m1["excluded_ids"]
    # a later cohort-clean rebuild passes prior ids + a live seed that still
    # contains B — it must stay out, and the exclusion must survive the merge
    m2 = build_manifest(
        ["A", "B", "C", "D"], policy="p", source="s",
        frozen_utc=m1["frozen_utc"], prior_test_ids=m1["test_ids"],
        excluded_ids=m1.get("excluded_ids", {}),
    )
    assert m2["test_ids"] == ["A", "C", "D"]
    assert m2["excluded_ids"] == m1["excluded_ids"]
    assert m2["frozen_utc"] == m1["frozen_utc"]
