from ._loader import load_script_module

_module = load_script_module("score_nightly")

build_score_payload = _module.build_score_payload
main = _module.main

__all__ = ["build_score_payload", "main"]
