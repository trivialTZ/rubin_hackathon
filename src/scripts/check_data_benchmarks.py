from ._loader import load_script_module

_module = load_script_module("check_data_benchmarks")

DEFAULT_BENCHMARKS = _module.DEFAULT_BENCHMARKS
evaluate_run = _module.evaluate_run
load_benchmark_spec = _module.load_benchmark_spec
main = _module.main

__all__ = ["DEFAULT_BENCHMARKS", "evaluate_run", "load_benchmark_spec", "main"]
