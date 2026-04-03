"""Test-friendly shim for importing repo-root script modules via ``src``.

When tests import ``scripts`` after prepending ``src/`` to ``sys.path``, Python
resolves the package to ``src/scripts`` first. Extend the package search path to
also include the repo-root ``scripts/`` directory so ``import scripts.foo``
works regardless of import order.
"""

from __future__ import annotations

from pathlib import Path

_REPO_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if _REPO_SCRIPTS.exists():
    __path__.append(str(_REPO_SCRIPTS))
