"""scripts/probe.py — Run liveness probes on all broker adapters.

Usage:
    python scripts/probe.py
    python scripts/probe.py --brokers alerce fink
    python scripts/probe.py --object ZTF21abbzjeq
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debass.access import ALL_ADAPTERS

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


def _print_table(rows: list[dict]) -> None:
    if console is None:
        for r in rows:
            print(r)
        return
    t = Table(title="Broker Probe Results")
    cols = ["broker", "phase", "semantic_type", "status", "detail"]
    for c in cols:
        t.add_column(c, overflow="fold")
    for r in rows:
        detail = r.get("reason") or r.get("rows") or r.get("http") or ""
        status = r.get("status", "?")
        color = {"ok": "green", "error": "red", "stubbed": "yellow",
                 "fixture_fallback": "cyan", "unavailable": "red"}.get(status, "white")
        t.add_row(
            r["broker"],
            str(r.get("phase", "?")),
            r.get("semantic_type", "?"),
            f"[{color}]{status}[/{color}]",
            str(detail),
        )
    console.print(t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe all broker adapters")
    parser.add_argument("--brokers", nargs="+", help="Restrict to named brokers")
    parser.add_argument("--object", default="ZTF21abbzjeq", help="Test object ID")
    args = parser.parse_args()

    rows = []
    for cls in ALL_ADAPTERS:
        adapter = cls()
        if args.brokers and adapter.name not in args.brokers:
            continue
        result = adapter.probe()
        result["phase"] = adapter.phase
        result["semantic_type"] = adapter.semantic_type
        rows.append(result)

        # If live access, also fetch the test object
        if result.get("status") == "ok" and args.object:
            try:
                out = adapter.fetch_object(args.object)
                print(f"  [{adapter.name}] {args.object}: {len(out.fields)} fields extracted"
                      f"{' (fixture)' if out.fixture_used else ''}")
            except Exception as exc:
                print(f"  [{adapter.name}] fetch_object error: {exc}")

    _print_table(rows)


if __name__ == "__main__":
    main()
