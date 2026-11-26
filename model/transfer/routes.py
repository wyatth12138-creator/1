"""Serializable endpoint topology, independent of task/label-space identifiers."""
from __future__ import annotations

from copy import deepcopy

from .adapters import BASE_RANKS, BLOCK_INDICES


class TaskRoutes:
    def __init__(self):
        self.versions = {i: {"v0": {"rank": BASE_RANKS[i], "kind": "base", "frozen": False,
                                    "archived": False}} for i in BLOCK_INDICES}
        self.live = {i: ("v0",) for i in BLOCK_INDICES}
        self.endpoints = {}

    def add_endpoint(self, endpoint):
        if endpoint in self.endpoints:
            raise ValueError(f"Endpoint already registered: {endpoint}")
        self.endpoints[endpoint] = tuple(self.live[i] for i in BLOCK_INDICES)

    def route(self, endpoint):
        try:
            return self.endpoints[endpoint]
        except KeyError as exc:
            raise KeyError(f"Unknown endpoint {endpoint!r}; register it before use") from exc

    def add_version(self, block, rank, kind):
        versions = self.versions[block]
        name = f"v{len(versions)}"
        versions[name] = {"rank": int(rank), "kind": kind, "frozen": False, "archived": False}
        return name

    def export(self):
        return {"versions": {str(i): deepcopy(v) for i, v in self.versions.items()},
                "live": {str(i): list(v) for i, v in self.live.items()},
                "endpoints": {name: [list(chain) for chain in route]
                              for name, route in self.endpoints.items()}}

    @classmethod
    def from_spec(cls, spec):
        result = cls()
        result.versions = {int(i): deepcopy(v) for i, v in spec["versions"].items()}
        result.live = {int(i): tuple(v) for i, v in spec["live"].items()}
        result.endpoints = {name: tuple(tuple(c) for c in route)
                            for name, route in spec["endpoints"].items()}
        if set(result.versions) != set(BLOCK_INDICES) or set(result.live) != set(BLOCK_INDICES):
            raise ValueError("Invalid adapter topology: block indices differ from Small configuration")
        for route in (*result.endpoints.values(), tuple(result.live[i] for i in BLOCK_INDICES)):
            if len(route) != 13:
                raise ValueError("Invalid route length")
            for block, chain in zip(BLOCK_INDICES, route):
                if not chain or any(version not in result.versions[block] for version in chain):
                    raise ValueError(f"Invalid route version at block {block}")
        return result
