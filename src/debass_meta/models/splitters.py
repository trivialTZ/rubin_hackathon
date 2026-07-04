"""Object-level data split helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupSplit:
    train_ids: set[str]
    cal_ids: set[str]
    test_ids: set[str]


def _association_clusters(
    id_order: list[str],
    association_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Union-find over association edges → (id → cluster rep, rep → members).

    Edges are undirected (LSST diaObjectId ↔ ZTF oid) and may pass through
    ids absent from ``id_order`` (two objects sharing one counterpart still
    land in the same cluster).  Reps and member order are deterministic:
    the rep is the lexicographically smallest member; members keep the
    caller's id order.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo

    for a, b in sorted((str(k), str(v)) for k, v in association_map.items()):
        if a and b:
            union(a, b)

    rep_of: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    for oid in id_order:
        rep = find(oid) if oid in parent else oid
        rep_of[oid] = rep
        members.setdefault(rep, []).append(oid)
    # Normalize the rep to the smallest member actually present in id_order
    # so downstream never keys on an id outside the split population.
    normalized: dict[str, str] = {}
    out_members: dict[str, list[str]] = {}
    for rep, ids in members.items():
        canon = min(ids)
        out_members[canon] = ids
        for oid in ids:
            normalized[oid] = canon
    return normalized, out_members


def group_train_cal_test_split(
    object_to_label: dict[str, str | None],
    *,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    seed: int = 42,
    association_map: dict[str, str] | None = None,
) -> GroupSplit:
    """Split object IDs by label while keeping all epochs of an object together.

    ``association_map`` (optional, v10): object_id → associated counterpart id
    (e.g. LSST diaObjectId → ZTF oid).  Associated objects share photometry, so
    they are unioned into one cluster and the WHOLE cluster is assigned to a
    single split — the duplicate-photometry leak defense.  Without association
    data the behaviour (including RNG consumption) is exactly the historical
    per-object split.
    """
    rng = np.random.default_rng(seed)
    id_order = [str(o) for o in object_to_label]
    label_of = {
        str(o): (str(lab) if lab is not None else None)
        for o, lab in object_to_label.items()
    }
    if association_map:
        rep_of, members = _association_clusters(id_order, association_map)
    else:
        rep_of = {oid: oid for oid in id_order}
        members = {oid: [oid] for oid in id_order}

    def _cluster_label(rep: str) -> str | None:
        labels = sorted({label_of[m] for m in members[rep] if label_of[m] is not None})
        return labels[0] if labels else None

    by_label: dict[str, list[str]] = {}
    unlabeled: list[str] = []
    seen: set[str] = set()
    for oid in id_order:
        rep = rep_of[oid]
        if rep in seen:
            continue
        seen.add(rep)
        label = _cluster_label(rep)
        if label is None:
            unlabeled.append(rep)
            continue
        by_label.setdefault(label, []).append(rep)

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

    def _expand(reps: list[str]) -> set[str]:
        return {m for rep in reps for m in members[rep]}

    for ids in by_label.values():
        train, cal, test = _split_ids(ids)
        train_ids.update(_expand(train))
        cal_ids.update(_expand(cal))
        test_ids.update(_expand(test))

    if unlabeled:
        train, cal, test = _split_ids(unlabeled)
        train_ids.update(_expand(train))
        cal_ids.update(_expand(cal))
        test_ids.update(_expand(test))

    return GroupSplit(train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids)
