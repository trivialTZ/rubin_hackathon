"""Helpers for loading repo-root script modules through the src package path."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"_debass_script_{script_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load script module {script_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
