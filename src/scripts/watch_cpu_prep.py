from ._loader import load_script_module

_module = load_script_module("watch_cpu_prep")

_has_failure = _module._has_failure
_log_phase_hint = _module._log_phase_hint
main = _module.main

__all__ = ["_has_failure", "_log_phase_hint", "main"]


def __getattr__(name: str):
    return getattr(_module, name)
