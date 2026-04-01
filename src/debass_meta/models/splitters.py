"""Object-level data split helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupSplit:
    train_ids: set[str]
    cal_ids: set[str]
    test_ids: set[str]


def group_train_cal_test_split(
    object_to_label: dict[str, str | None],
    *,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    seed: int = 42,
) -> GroupSplit:
    """Split object IDs by label while keeping all epochs of an object together."""
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[str]] = {}
    unlabeled: list[str] = []
    for object_id, label in object_to_label.items():
        if label is None:
            unlabeled.append(str(object_id))
            continue
        by_label.setdefault(str(label), []).append(str(object_id))

    train_ids: set[str] = set()
    cal_ids: set[str] = set()
    test_ids: set[str] = set()

    def _split_ids(ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        ids = ids.copy()
        rng.shuffle(ids)
        n = len(ids)
        if n <= 2:
            if n == 1:
                return ids, [], []
            return [ids[0]], [ids[1]], []
        n_train = max(1, int(round(n * train_frac)))
        n_cal = max(1, int(round(n * cal_frac)))
        if n_train + n_cal >= n:
            overflow = n_train + n_cal - (n - 1)
            if overflow > 0:
                n_train = max(1, n_train - overflow)
        n_test = n - n_train - n_cal
        if n_test < 1:
            n_cal = max(0, n_cal - 1)
            n_test = n - n_train - n_cal
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = n - n_train - n_cal
        return ids[:n_train], ids[n_train:n_train + n_cal], ids[n_train + n_cal:]

    for ids in by_label.values():
        train, cal, test = _split_ids(ids)
        train_ids.update(train)
        cal_ids.update(cal)
        test_ids.update(test)

    if unlabeled:
        train, cal, test = _split_ids(unlabeled)
        train_ids.update(train)
        cal_ids.update(cal)
        test_ids.update(test)

    return GroupSplit(train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids)
