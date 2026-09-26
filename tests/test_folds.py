"""StableGroupKFold: GroupKFold's greedy balancing with a fold map that depends only on the groups
(not on the CPU's sort kernels or the row order) — fusion v13f."""
from __future__ import annotations

import numpy as np
import pytest

from debass_meta.models.folds import StableGroupKFold


def _groups(seed: int = 0) -> np.ndarray:
    sizes = np.random.default_rng(seed).integers(1, 21, 600)   # many equal group sizes
    return np.repeat(np.array([f"obj{i:04d}" for i in range(600)]), sizes)


def test_folds_are_a_partition_with_whole_groups_and_balanced_rows():
    g = _groups()
    splits = list(StableGroupKFold(5).split(None, None, g))
    test_rows = np.concatenate([te for _, te in splits])
    assert np.array_equal(np.sort(test_rows), np.arange(len(g)))
    for tr, te in splits:
        assert not set(g[tr]) & set(g[te])
    sizes = [len(te) for _, te in splits]
    assert max(sizes) - min(sizes) <= 20                       # greedy: within one largest group


def test_fold_map_ignores_row_order_and_breaks_ties_by_group_id():
    g = _groups()
    kf = StableGroupKFold(5)
    fold = dict(zip(g, kf.fold_of_rows(g)))
    perm = np.random.default_rng(1).permutation(len(g))
    fold_perm = dict(zip(g[perm], kf.fold_of_rows(g[perm])))
    assert fold == fold_perm
    # equal-size groups are placed in group-id order
    two = np.array(["b", "a", "c", "d"])
    f = StableGroupKFold(2).fold_of_rows(two)
    assert dict(zip(two, f)) == {"a": 0, "b": 1, "c": 0, "d": 1}


def test_too_many_splits_raises():
    with pytest.raises(ValueError):
        list(StableGroupKFold(5).split(None, None, np.array(["a", "b"])))
