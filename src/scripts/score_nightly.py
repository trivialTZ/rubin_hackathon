from ._loader import load_script_module

_module = load_script_module("score_nightly")

_resolve_snapshot_path = _module._resolve_snapshot_path
build_score_payload = _module.build_score_payload
main = _module.main

__all__ = ["_resolve_snapshot_path", "build_score_payload", "main"]


def __getattr__(name: str):
    return getattr(_module, name)
