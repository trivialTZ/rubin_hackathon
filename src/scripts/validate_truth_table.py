from ._loader import load_script_module

_module = load_script_module("validate_truth_table")

validate_truth_frame = _module.validate_truth_frame
main = _module.main

__all__ = ["validate_truth_frame", "main"]
