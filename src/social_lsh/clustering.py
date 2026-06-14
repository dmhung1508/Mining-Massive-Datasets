from __future__ import annotations

import pandas as pd


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def connected_components(tweet_ids: list[int], verified_pairs: pd.DataFrame) -> pd.DataFrame:
    union_find = UnionFind(tweet_ids)

    for row in verified_pairs[["tweet_id_left", "tweet_id_right"]].itertuples(index=False):
        union_find.union(int(row.tweet_id_left), int(row.tweet_id_right))

    roots = {tweet_id: union_find.find(tweet_id) for tweet_id in tweet_ids}
    root_order = {root: idx + 1 for idx, root in enumerate(sorted(set(roots.values())))}
    component_sizes: dict[int, int] = {}
    for root in roots.values():
        component_sizes[root] = component_sizes.get(root, 0) + 1

    rows = [
        {
            "tweet_id": tweet_id,
            "cluster_id": root_order[root],
            "cluster_size": component_sizes[root],
        }
        for tweet_id, root in sorted(roots.items())
    ]
    return pd.DataFrame(rows)
