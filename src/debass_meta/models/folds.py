"""Grouped K-fold with a platform-independent fold map (fusion v13f).

``sklearn.model_selection.GroupKFold`` (shuffle=False) orders the groups by size
with ``np.argsort``; NumPy 2 dispatches that sort to SIMD kernels (AVX-512 on
some SCC nodes, not on others) whose order among equal sizes differs, so objects
with the same number of rows landed in different folds depending on the node.
The out-of-fold predictions, the calibrators fitted on them and the blend α
changed between identical runs (docs/fusion_v13_plan.md, v13f).

:class:`StableGroupKFold` runs the same greedy balancing (largest group first,
into the currently lightest fold) with a stable sort, so ties are broken by
group id and the fold map depends only on the groups.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class StableGroupKFold:
    """Drop-in for ``GroupKFold(n_splits)`` whose folds do not depend on the CPU
    or on the row order."""

    def __init__(self, n_splits: int = 5) -> None:
        if int(n_splits) < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = int(n_splits)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def fold_of_rows(self, groups) -> np.ndarray:
        """Fold index of every row."""
        groups = np.asarray(groups)
        unique_groups, inverse = np.unique(groups, return_inverse=True)
        if self.n_splits > len(unique_groups):
            raise ValueError(f"Cannot have n_splits={self.n_splits} greater than the number "
                             f"of groups: {len(unique_groups)}.")
        sizes = np.bincount(inverse.ravel())
        order = np.argsort(-sizes, kind="stable")      # largest first, ties by group id
        load = np.zeros(self.n_splits)
        fold_of_group = np.empty(len(unique_groups), dtype=int)
        for g in order:
            f = int(np.argmin(load))                   # first lightest fold
            fold_of_group[g] = f
            load[f] += sizes[g]
        return fold_of_group[inverse.ravel()]

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        fold = self.fold_of_rows(groups)
        for f in range(self.n_splits):
            yield np.flatnonzero(fold != f), np.flatnonzero(fold == f)
