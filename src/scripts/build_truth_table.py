from ._loader import load_script_module

_module = load_script_module("build_truth_table")

build_truth_table = _module.build_truth_table
main = _module.main

__all__ = ["build_truth_table", "main"]
