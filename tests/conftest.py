from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

for path in (str(_SRC), str(_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

existing_scripts = sys.modules.get("scripts")
if existing_scripts is not None:
    file_path = str(getattr(existing_scripts, "__file__", "") or "")
    package_paths = [str(path) for path in getattr(existing_scripts, "__path__", [])]
    if str(_ROOT) not in file_path and not any(str(_ROOT) in path for path in package_paths):
        del sys.modules["scripts"]
