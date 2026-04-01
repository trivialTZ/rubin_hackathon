from ._loader import load_script_module

_module = load_script_module("audit_data_readiness")

audit_data_readiness = _module.audit_data_readiness
main = _module.main

__all__ = ["audit_data_readiness", "main"]
