"""DEBASS Dashboard — on-demand broker re-query for a single object.

Wraps the existing broker adapters in src/debass_meta/access/ to provide
a "Refresh Now" capability in the dashboard.

Usage:
    result = refresh_object("ZTF21abcdef")
    # result is a dict matching the JSONL score record format
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# Add project root to path so we can import debass_meta
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import streamlit as st


@st.cache_data(ttl=60, show_spinner="Querying brokers...")
def refresh_object(object_id: str) -> dict:
    """Query all brokers for a single object and return a score-like dict.

    This is a lightweight version of the full scoring pipeline,
    suitable for on-demand dashboard refresh.  It calls the broker
    adapters directly but does NOT retrain any models.
    """
    results = {
        "object_id": object_id,
        "query_time": time.time(),
        "brokers": {},
        "errors": [],
    }

    # ── Try each broker ───────────────────────────────────────────
    broker_queries = [
        ("ALeRCE", _query_alerce),
        ("Fink", _query_fink),
        ("Lasair", _query_lasair),
    ]

    for name, fn in broker_queries:
        try:
            data = fn(object_id)
            results["brokers"][name] = data
        except Exception as e:
            results["errors"].append(f"{name}: {e}")

    return results


def _query_alerce(object_id: str) -> dict:
    """Query ALeRCE for classifier outputs."""
    try:
        from debass_meta.access.alerce import AlerceAdapter
        adapter = AlerceAdapter()
        outputs = adapter.query_object(object_id)
        return {
            "status": "ok",
            "n_outputs": len(outputs) if outputs else 0,
            "classifiers": _extract_fields(outputs),
        }
    except ImportError:
        return {"status": "adapter_unavailable"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _query_fink(object_id: str) -> dict:
    """Query Fink for per-alert classifier scores."""
    try:
        from debass_meta.access.fink import FinkAdapter
        adapter = FinkAdapter()
        outputs = adapter.query_object(object_id)
        return {
            "status": "ok",
            "n_outputs": len(outputs) if outputs else 0,
            "classifiers": _extract_fields(outputs),
        }
    except ImportError:
        return {"status": "adapter_unavailable"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _query_lasair(object_id: str) -> dict:
    """Query Lasair for Sherlock context."""
    try:
        from debass_meta.access.lasair import LasairAdapter
        adapter = LasairAdapter()
        outputs = adapter.query_object(object_id)
        return {
            "status": "ok",
            "n_outputs": len(outputs) if outputs else 0,
            "classifiers": _extract_fields(outputs),
        }
    except ImportError:
        return {"status": "adapter_unavailable"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _extract_fields(outputs) -> list[dict]:
    """Extract simplified field info from BrokerOutput objects."""
    if not outputs:
        return []
    fields = []
    for out in outputs:
        if not hasattr(out, "fields"):
            continue
        for f in (out.fields or []):
            fields.append({
                "expert_key": f.get("expert_key", "?"),
                "raw_label_or_score": f.get("raw_label_or_score"),
                "semantic_type": f.get("semantic_type"),
                "canonical_projection": f.get("canonical_projection"),
                "temporal_exactness": f.get("temporal_exactness"),
            })
    return fields
