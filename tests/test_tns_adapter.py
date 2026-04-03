"""Tests for the TNS access adapter and three-tier quality logic."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# Allow importing crossmatch_tns from scripts/ (not a package)
import importlib.util
_crossmatch_spec = importlib.util.spec_from_file_location(
    "crossmatch_tns",
    str(Path(__file__).resolve().parent.parent / "scripts" / "crossmatch_tns.py"),
)
_crossmatch_mod = importlib.util.module_from_spec(_crossmatch_spec)
sys.modules["crossmatch_tns"] = _crossmatch_mod
_crossmatch_spec.loader.exec_module(_crossmatch_mod)

from debass_meta.access.tns import (
    TNSClient,
    TNSCredentials,
    TNSResult,
    map_tns_type_to_ternary,
    is_ambiguous_type,
    load_tns_credentials,
    extract_classification_credibility,
    is_credible_classification,
    TNS_CREDIBLE_GROUPS,
    TNS_AMBIGUOUS_TYPES,
    _extract_type,
    _has_spectra,
    _extract_redshift,
)
from debass_meta.truth.quality import (
    QualityAssignment,
    assign_quality_from_tns,
    collect_broker_votes,
    CONSENSUS_MIN_EXPERTS,
)


# ------------------------------------------------------------------ #
# TNS ternary mapping                                                  #
# ------------------------------------------------------------------ #


class TestTNSTernaryMapping:
    def test_snia_types(self):
        assert map_tns_type_to_ternary("SN Ia") == "snia"
        assert map_tns_type_to_ternary("SN Ia-91T-like") == "snia"
        assert map_tns_type_to_ternary("SN Ia-91bg-like") == "snia"
        assert map_tns_type_to_ternary("SN Ia-CSM") == "snia"
        assert map_tns_type_to_ternary("SN Iax[02cx-like]") == "snia"

    def test_cc_types(self):
        assert map_tns_type_to_ternary("SN II") == "nonIa_snlike"
        assert map_tns_type_to_ternary("SN IIb") == "nonIa_snlike"
        assert map_tns_type_to_ternary("SN Ib") == "nonIa_snlike"
        assert map_tns_type_to_ternary("SN Ic-BL") == "nonIa_snlike"
        assert map_tns_type_to_ternary("SLSN-I") == "nonIa_snlike"

    def test_other_types(self):
        assert map_tns_type_to_ternary("TDE") == "other"
        assert map_tns_type_to_ternary("AGN") == "other"
        assert map_tns_type_to_ternary("Nova") == "other"

    def test_unknown_type(self):
        assert map_tns_type_to_ternary(None) is None
        assert map_tns_type_to_ternary("") is None
        assert map_tns_type_to_ternary("UnknownType42") is None

    def test_fuzzy_snia(self):
        assert map_tns_type_to_ternary("SN Ia-weird-variant") == "snia"

    def test_fuzzy_sn(self):
        assert map_tns_type_to_ternary("SN IIn-weird") == "nonIa_snlike"
        assert map_tns_type_to_ternary("SLSN-new") == "nonIa_snlike"


# ------------------------------------------------------------------ #
# TNS response parsing                                                 #
# ------------------------------------------------------------------ #


class TestTNSParsing:
    def test_extract_type_dict(self):
        detail = {"object_type": {"name": "SN Ia"}}
        assert _extract_type(detail) == "SN Ia"

    def test_extract_type_string(self):
        detail = {"type": "SN II"}
        assert _extract_type(detail) == "SN II"

    def test_extract_type_empty(self):
        assert _extract_type({}) is None

    def test_has_spectra_list(self):
        assert _has_spectra({"spectra": [{"id": 1}]}) is True
        assert _has_spectra({"spectra": []}) is False

    def test_has_spectra_classification(self):
        assert _has_spectra({"classification_spectra": [{"id": 1}]}) is True
        assert _has_spectra({"classification_spectra": []}) is False

    def test_has_spectra_empty(self):
        assert _has_spectra({}) is False

    def test_extract_redshift(self):
        assert _extract_redshift({"redshift": 0.05}) == 0.05
        assert _extract_redshift({"host_redshift": 0.03}) == 0.03
        assert _extract_redshift({}) is None
        assert _extract_redshift({"redshift": "nan"}) is None


# ------------------------------------------------------------------ #
# Credential security                                                  #
# ------------------------------------------------------------------ #


class TestCredentialSecurity:
    def test_repr_redacts_api_key(self):
        creds = TNSCredentials(
            api_key="super_secret_key_12345",
            tns_id="999",
            marker_type="bot",
            marker_name="testbot",
        )
        rep = repr(creds)
        assert "super_secret_key_12345" not in rep
        assert "***" in rep
        assert "testbot" in rep

    def test_load_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="Missing TNS_API_KEY"):
                load_tns_credentials()

    def test_load_requires_tns_id(self):
        with patch.dict(os.environ, {"TNS_API_KEY": "key"}, clear=True):
            with pytest.raises(RuntimeError, match="Missing TNS_TNS_ID"):
                load_tns_credentials()

    def test_load_requires_marker_name(self):
        env = {"TNS_API_KEY": "key", "TNS_TNS_ID": "123"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="Missing TNS_MARKER_NAME"):
                load_tns_credentials()

    def test_load_success(self):
        env = {
            "TNS_API_KEY": "key",
            "TNS_TNS_ID": "123",
            "TNS_MARKER_NAME": "mybot",
        }
        with patch.dict(os.environ, env, clear=True):
            creds = load_tns_credentials()
            assert creds.api_key == "key"
            assert creds.tns_id == "123"
            assert creds.marker_name == "mybot"
            assert creds.marker_type == "user"

    def test_user_agent_format(self):
        creds = TNSCredentials(
            api_key="k",
            tns_id="42",
            marker_type="bot",
            marker_name="mybot",
        )
        ua = creds.user_agent
        assert ua.startswith("tns_marker")
        parsed = json.loads(ua.replace("tns_marker", ""))
        assert parsed["tns_id"] == "42"
        assert parsed["name"] == "mybot"


# ------------------------------------------------------------------ #
# Three-tier quality assignment                                        #
# ------------------------------------------------------------------ #


class TestQualityAssignment:
    def test_spectroscopic(self):
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=True,
        )
        assert qa.quality == "spectroscopic"
        assert qa.label_source == "tns_spectroscopic"
        assert qa.final_class_ternary == "snia"

    def test_spectroscopic_cc(self):
        qa = assign_quality_from_tns(
            tns_type="SN II",
            tns_ternary="nonIa_snlike",
            has_spectra=True,
        )
        assert qa.quality == "spectroscopic"
        assert qa.final_class_ternary == "nonIa_snlike"

    def test_consensus_tns_plus_brokers(self):
        votes = {
            "fink/snn": "snia",
            "fink/rf_ia": "snia",
            "alerce/lc_classifier_transient": "snia",
        }
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=False,
            broker_votes=votes,
        )
        assert qa.quality == "consensus"
        assert qa.label_source == "tns_consensus"
        assert qa.consensus_n_agree == 3

    def test_consensus_insufficient_brokers(self):
        votes = {
            "fink/snn": "snia",
            "fink/rf_ia": "nonIa_snlike",
        }
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=False,
            broker_votes=votes,
        )
        # Only 1 broker agrees, not enough
        assert qa.quality == "weak"
        assert qa.label_source == "tns_unconfirmed"

    def test_consensus_broker_only(self):
        votes = {
            "fink/snn": "nonIa_snlike",
            "fink/rf_ia": "nonIa_snlike",
            "alerce/lc_classifier_transient": "nonIa_snlike",
            "alerce/stamp_classifier": "nonIa_snlike",
        }
        qa = assign_quality_from_tns(
            tns_type=None,
            tns_ternary=None,
            has_spectra=False,
            broker_votes=votes,
        )
        assert qa.quality == "consensus"
        assert qa.label_source == "broker_consensus"
        assert qa.final_class_ternary == "nonIa_snlike"
        assert qa.consensus_n_agree == 4

    def test_weak_tns_unconfirmed(self):
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=False,
        )
        assert qa.quality == "weak"
        assert qa.label_source == "tns_unconfirmed"

    def test_weak_no_match(self):
        qa = assign_quality_from_tns(
            tns_type=None,
            tns_ternary=None,
            has_spectra=False,
        )
        assert qa.quality == "weak"
        assert qa.label_source == "no_tns_match"
        assert qa.final_class_ternary is None

    def test_spectra_overrides_consensus(self):
        """Spectroscopic tier takes priority even if broker votes disagree."""
        votes = {
            "fink/snn": "nonIa_snlike",
            "fink/rf_ia": "nonIa_snlike",
            "alerce/lc_classifier_transient": "nonIa_snlike",
        }
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=True,
            broker_votes=votes,
        )
        assert qa.quality == "spectroscopic"
        assert qa.final_class_ternary == "snia"


# ------------------------------------------------------------------ #
# Broker vote collection                                               #
# ------------------------------------------------------------------ #


class TestBrokerVoteCollection:
    def test_collect_from_events(self):
        import pandas as pd

        events = pd.DataFrame([
            {
                "object_id": "ZTF23abc",
                "expert_key": "fink/snn",
                "n_det": 15,
                "event_time_jd": 2460000.5,
                "class_name": "snia",
                "canonical_projection": 0.9,
                "raw_label_or_score": 0.9,
            },
            {
                "object_id": "ZTF23abc",
                "expert_key": "fink/rf_ia",
                "n_det": 15,
                "event_time_jd": 2460000.5,
                "class_name": "snia",
                "canonical_projection": 0.85,
                "raw_label_or_score": 0.85,
            },
        ])
        votes = collect_broker_votes(events, "ZTF23abc", min_n_det=10)
        assert votes.get("fink/snn") == "snia"
        assert votes.get("fink/rf_ia") == "snia"

    def test_temporal_firewall(self):
        """Events with n_det < threshold are excluded."""
        import pandas as pd

        events = pd.DataFrame([
            {
                "object_id": "ZTF23abc",
                "expert_key": "fink/snn",
                "n_det": 3,
                "event_time_jd": 2460000.5,
                "class_name": "snia",
                "canonical_projection": 0.9,
                "raw_label_or_score": 0.9,
            },
        ])
        votes = collect_broker_votes(events, "ZTF23abc", min_n_det=10)
        assert len(votes) == 0

    def test_no_events(self):
        import pandas as pd

        votes = collect_broker_votes(pd.DataFrame(), "ZTF23abc")
        assert votes == {}

    def test_low_prob_excluded(self):
        import pandas as pd

        events = pd.DataFrame([
            {
                "object_id": "ZTF23abc",
                "expert_key": "fink/snn",
                "n_det": 15,
                "event_time_jd": 2460000.5,
                "class_name": "snia",
                "canonical_projection": 0.3,
                "raw_label_or_score": 0.3,
            },
        ])
        votes = collect_broker_votes(events, "ZTF23abc", min_n_det=10, min_prob=0.5)
        assert len(votes) == 0


# ------------------------------------------------------------------ #
# Validate truth table accepts three-tier quality                      #
# ------------------------------------------------------------------ #


class TestValidateTruthThreeTier:
    def test_spectroscopic_quality_accepted(self):
        import pandas as pd
        from validate_truth_table import validate_truth_frame

        df = pd.DataFrame([{
            "object_id": "ZTF23abc",
            "final_class_raw": "SN Ia",
            "final_class_ternary": "snia",
            "follow_proxy": 1,
            "label_source": "tns_spectroscopic",
            "label_quality": "spectroscopic",
        }])
        result = validate_truth_frame(df)
        assert result["passed"] is True

    def test_consensus_quality_accepted(self):
        import pandas as pd
        from validate_truth_table import validate_truth_frame

        df = pd.DataFrame([{
            "object_id": "ZTF23abc",
            "final_class_raw": "SN Ia",
            "final_class_ternary": "snia",
            "follow_proxy": 1,
            "label_source": "tns_consensus",
            "label_quality": "consensus",
        }])
        result = validate_truth_frame(df)
        assert result["passed"] is True

    def test_mixed_quality_accepted(self):
        import pandas as pd
        from validate_truth_table import validate_truth_frame

        df = pd.DataFrame([
            {
                "object_id": "ZTF23abc",
                "final_class_raw": "SN Ia",
                "final_class_ternary": "snia",
                "follow_proxy": 1,
                "label_source": "tns_spectroscopic",
                "label_quality": "spectroscopic",
            },
            {
                "object_id": "ZTF23def",
                "final_class_raw": "SN II",
                "final_class_ternary": "nonIa_snlike",
                "follow_proxy": 0,
                "label_source": "tns_consensus",
                "label_quality": "consensus",
            },
            {
                "object_id": "ZTF23ghi",
                "final_class_raw": "other",
                "final_class_ternary": "other",
                "follow_proxy": 0,
                "label_source": "alerce_self_label",
                "label_quality": "weak",
            },
        ])
        result = validate_truth_frame(df)
        assert result["passed"] is True


# ------------------------------------------------------------------ #
# TNS crossmatch cache                                                #
# ------------------------------------------------------------------ #


class TestTNSCache:
    def test_cache_roundtrip(self):
        from crossmatch_tns import _load_cached, _save_cached

        result = TNSResult(
            object_id="ZTF23abc",
            tns_name="2023xyz",
            tns_prefix="SN",
            type_name="SN Ia",
            has_spectra=True,
            redshift=0.05,
            ra=180.0,
            dec=-30.0,
            discovery_date="2023-01-15",
            raw={"test": True},
        )
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            _save_cached(cache_dir, result)
            loaded = _load_cached(cache_dir, "ZTF23abc")
            assert loaded is not None
            assert loaded.object_id == "ZTF23abc"
            assert loaded.tns_name == "2023xyz"
            assert loaded.type_name == "SN Ia"
            assert loaded.has_spectra is True
            assert loaded.redshift == 0.05

    def test_cache_miss(self):
        from crossmatch_tns import _load_cached

        with tempfile.TemporaryDirectory() as td:
            assert _load_cached(Path(td), "nonexistent") is None


# ------------------------------------------------------------------ #
# Build truth from crossmatch results                                 #
# ------------------------------------------------------------------ #


class TestBuildTruthFromCrossmatch:
    def test_spectroscopic_result(self):
        from crossmatch_tns import build_truth_from_crossmatch

        results = [
            TNSResult(
                object_id="ZTF23abc",
                tns_name="2023xyz",
                tns_prefix="SN",
                type_name="SN Ia",
                has_spectra=True,
                redshift=0.05,
                ra=180.0,
                dec=-30.0,
                discovery_date="2023-01-15",
                raw={},
            ),
        ]
        df = build_truth_from_crossmatch(tns_results=results, use_consensus=False)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["label_quality"] == "spectroscopic"
        assert row["final_class_ternary"] == "snia"
        assert row["follow_proxy"] == 1
        assert bool(row["tns_has_spectra"]) is True

    def test_no_match_with_fallback(self):
        from crossmatch_tns import build_truth_from_crossmatch

        results = [
            TNSResult(
                object_id="ZTF23abc",
                tns_name=None,
                tns_prefix=None,
                type_name=None,
                has_spectra=False,
                redshift=None,
                ra=None,
                dec=None,
                discovery_date=None,
                raw={},
            ),
        ]
        df = build_truth_from_crossmatch(
            tns_results=results,
            use_consensus=False,
            existing_labels={"ZTF23abc": "snia"},
        )
        assert len(df) == 1
        assert df.iloc[0]["label_quality"] == "weak"
        assert df.iloc[0]["label_source"] == "alerce_self_label"

    def test_no_match_no_fallback_skipped(self):
        from crossmatch_tns import build_truth_from_crossmatch

        results = [
            TNSResult(
                object_id="ZTF23abc",
                tns_name=None,
                tns_prefix=None,
                type_name=None,
                has_spectra=False,
                redshift=None,
                ra=None,
                dec=None,
                discovery_date=None,
                raw={},
            ),
        ]
        df = build_truth_from_crossmatch(tns_results=results, use_consensus=False)
        assert len(df) == 0


# ------------------------------------------------------------------ #
# Credibility framework                                                #
# ------------------------------------------------------------------ #


class TestCredibilityExtraction:
    """Tests for extract_classification_credibility()."""

    def test_professional_group_recognized(self):
        """ZTF group with SEDM instrument → professional."""
        detail = {
            "spectra": [{
                "source_group_name": "ZTF",
                "instrument": {"name": "SEDM"},
                "observer": "SEDmRobot",
            }]
        }
        cred = extract_classification_credibility(detail)
        assert cred.tier == "professional"
        assert cred.source_group == "ZTF"

    def test_epessto_recognized(self):
        detail = {
            "spectra": [{
                "source_group_name": "ePESSTO+",
                "instrument": {"name": "EFOSC2"},
            }]
        }
        cred = extract_classification_credibility(detail)
        assert cred.tier == "professional"
        assert cred.source_group == "ePESSTO+"

    def test_unlisted_group_with_recognized_instrument(self):
        """Group not in registry but recognized instrument → likely_professional."""
        detail = {
            "spectra": [{
                "source_group_name": "NewSurveyProgram",
                "instrument": {"name": "ALFOSC"},
            }]
        }
        cred = extract_classification_credibility(detail)
        assert cred.tier == "likely_professional"
        assert cred.instrument == "ALFOSC"

    def test_unlisted_group_and_instrument(self):
        """Group and instrument both not in registry → unverified."""
        detail = {
            "spectra": [{
                "source_group_name": "UnregisteredGroup",
                "instrument": {"name": "CustomSetup"},
            }]
        }
        cred = extract_classification_credibility(detail)
        assert cred.tier == "unverified"

    def test_no_spectra(self):
        cred = extract_classification_credibility({})
        assert cred.tier == "unverified"
        assert cred.reason == "no_spectra_in_response"

    def test_empty_spectra_list(self):
        cred = extract_classification_credibility({"spectra": []})
        assert cred.tier == "unverified"

    def test_multiple_spectra_one_recognized(self):
        """If ANY spectrum is from a recognized source, classify as professional."""
        detail = {
            "spectra": [
                {
                    "source_group_name": "UnregisteredGroup",
                    "instrument": {"name": "Unknown"},
                },
                {
                    "source_group_name": "SCAT",
                    "instrument": {"name": "SNIFS"},
                },
            ]
        }
        cred = extract_classification_credibility(detail)
        assert cred.tier == "professional"
        assert cred.source_group == "SCAT"

    def test_is_credible_helper(self):
        assert is_credible_classification({
            "spectra": [{"source_group_name": "ZTF", "instrument": {"name": "SEDM"}}]
        }) is True
        assert is_credible_classification({
            "spectra": [{"source_group_name": "UnlistedGroup", "instrument": {"name": "UnlistedInstrument"}}]
        }) is False


class TestAmbiguousTypes:
    """Tests for ambiguous type detection."""

    def test_generic_sn_is_ambiguous(self):
        assert is_ambiguous_type("SN") is True

    def test_specific_snia_not_ambiguous(self):
        assert is_ambiguous_type("SN Ia") is False

    def test_specific_cc_not_ambiguous(self):
        assert is_ambiguous_type("SN II") is False

    def test_none_is_ambiguous(self):
        assert is_ambiguous_type(None) is True

    def test_new_rare_types_mapped(self):
        """Ensure new rare types have ternary mappings."""
        assert map_tns_type_to_ternary("SN Ibn") == "nonIa_snlike"
        assert map_tns_type_to_ternary("SN Icn") == "nonIa_snlike"
        assert map_tns_type_to_ternary("FBOT") == "other"
        assert map_tns_type_to_ternary("LRN") == "other"
        assert map_tns_type_to_ternary("PISN") == "nonIa_snlike"


class TestCredibilityQualityDowngrade:
    """Tests that unverified sources and ambiguous types are downgraded."""

    def test_professional_source_gets_spectroscopic(self):
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=True,
            credibility_tier="professional",
            source_group="ZTF",
        )
        assert qa.quality == "spectroscopic"
        assert qa.downgrade_reason is None

    def test_unverified_source_downgraded_to_weak(self):
        """Unverified source is assigned weak quality without broker consensus."""
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=True,
            credibility_tier="unverified",
            source_group="UnregisteredGroup",
        )
        # Without broker consensus, falls to weak
        assert qa.quality == "weak"
        assert qa.label_source == "tns_source_unverified"
        assert "unverified_source" in (qa.downgrade_reason or "")

    def test_unverified_source_with_consensus_gets_consensus(self):
        """Unverified source + broker agreement → consensus (corroborated)."""
        votes = {
            "fink/snn": "snia",
            "fink/rf_ia": "snia",
            "alerce/lc_classifier_transient": "snia",
        }
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=True,
            credibility_tier="unverified",
            source_group="UnregisteredGroup",
            broker_votes=votes,
        )
        assert qa.quality == "consensus"
        assert "downgraded" in qa.label_source

    def test_ambiguous_sn_downgraded(self):
        """Generic 'SN' type with spectra is NOT spectroscopic quality."""
        qa = assign_quality_from_tns(
            tns_type="SN",
            tns_ternary="nonIa_snlike",
            has_spectra=True,
            credibility_tier="professional",
            source_group="ZTF",
            type_is_ambiguous=True,
        )
        assert qa.quality == "weak"
        assert qa.label_source == "tns_ambiguous_type"
        assert "ambiguous_type" in (qa.downgrade_reason or "")

    def test_likely_professional_accepted(self):
        """likely_professional tier gets spectroscopic quality."""
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=True,
            credibility_tier="likely_professional",
            source_group="SmallPIProgram",
        )
        assert qa.quality == "spectroscopic"

    def test_no_credibility_info_backward_compat(self):
        """Bulk CSV mode: credibility_tier=None → accept as credible."""
        qa = assign_quality_from_tns(
            tns_type="SN Ia",
            tns_ternary="snia",
            has_spectra=True,
            credibility_tier=None,  # bulk CSV, no metadata
        )
        assert qa.quality == "spectroscopic"

    def test_cache_roundtrip_preserves_credibility(self):
        """Cache save/load preserves credibility fields."""
        from crossmatch_tns import _load_cached, _save_cached

        result = TNSResult(
            object_id="ZTF23abc",
            tns_name="2023xyz",
            tns_prefix="SN",
            type_name="SN Ia",
            has_spectra=True,
            redshift=0.05,
            ra=180.0,
            dec=-30.0,
            discovery_date="2023-01-15",
            raw={},
            source_group="ZTF",
            classification_instrument="SEDM",
            credibility_tier="professional",
            credibility_reason="credible_group:ZTF",
        )
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            _save_cached(cache_dir, result)
            loaded = _load_cached(cache_dir, "ZTF23abc")
            assert loaded is not None
            assert loaded.source_group == "ZTF"
            assert loaded.classification_instrument == "SEDM"
            assert loaded.credibility_tier == "professional"
            assert loaded.credibility_reason == "credible_group:ZTF"
