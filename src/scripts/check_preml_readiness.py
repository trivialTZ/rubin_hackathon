from ._loader import load_script_module

_module = load_script_module("check_preml_readiness")

LasairAdapter = _module.LasairAdapter


def evaluate_preml_readiness(*args, **kwargs):
    _module.LasairAdapter = LasairAdapter
    return _module.evaluate_preml_readiness(*args, **kwargs)


main = _module.main

__all__ = ["LasairAdapter", "evaluate_preml_readiness", "main"]


def __getattr__(name: str):
    return getattr(_module, name)
