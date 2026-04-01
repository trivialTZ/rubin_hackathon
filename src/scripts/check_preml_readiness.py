from ._loader import load_script_module

_module = load_script_module("check_preml_readiness")

evaluate_preml_readiness = _module.evaluate_preml_readiness
main = _module.main

__all__ = ["evaluate_preml_readiness", "main"]
