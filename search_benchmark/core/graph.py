"""
core/graph.py
-------------
Weighted undirected graph environment and its Problem subclass.
"""

from __future__ import annotations
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

try:
    from .problem import Node, Problem
    from .heuristics import euclidean
except ImportError:
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.core.problem import Node, Problem
    from search_benchmark.core.heuristics import euclidean

NodeID = int
Weight = float
Position = Tuple[float, float]


class WeightedGraph:
    """Undirected weighted graph with 2-D node positions."""

    def __init__(
        self,
        n_nodes: int,
        positions: Optional[Dict[NodeID, Position]] = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.positions = positions or {}
        self._adj: Dict[NodeID, List[Tuple[NodeID, Weight]]] = defaultdict(list)

    def add_edge(self, u: NodeID, v: NodeID, weight: Weight) -> None:
        self._adj[u].append((v, weight))
        self._adj[v].append((u, weight))

    def neighbours(self, node: NodeID) -> List[Tuple[NodeID, Weight]]:
        return self._adj.get(node, [])

    def nodes(self) -> List[NodeID]:
        return list(range(self.n_nodes))

    def position(self, node: NodeID) -> Optional[Position]:
        return self.positions.get(node)

    def distance(self, u: NodeID, v: NodeID) -> float:
        pu, pv = self.positions.get(u), self.positions.get(v)
        if pu is None or pv is None:
            return 0.0
        return euclidean(pu, pv)

    def to_dict(self) -> dict:
        edges = []
        seen = set()
        for u in range(self.n_nodes):
            for v, w in self._adj[u]:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    edges.append({"u": u, "v": v, "weight": round(w, 3)})
        return {
            "n_nodes": self.n_nodes,
            "positions": {str(k): list(v) for k, v in self.positions.items()},
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WeightedGraph":
        positions = {int(k): tuple(v) for k, v in data["positions"].items()}
        g = cls(n_nodes=data["n_nodes"], positions=positions)
        for e in data["edges"]:
            g.add_edge(e["u"], e["v"], e["weight"])
        return g

    def __repr__(self) -> str:
        n_edges = sum(len(v) for v in self._adj.values()) // 2
        return f"WeightedGraph(nodes={self.n_nodes}, edges={n_edges})"


class GraphProblem(Problem):
    """Concrete Problem over a WeightedGraph."""

    def __init__(
        self,
        graph: WeightedGraph,
        initial_state: NodeID,
        goal_state: NodeID,
        heuristic_scale: float = 1.0,
    ) -> None:
        super().__init__(initial_state, goal_state)
        self.graph = graph
        self.heuristic_scale = heuristic_scale

    def actions(self, state: NodeID) -> List[Tuple[NodeID, Weight]]:
        return self.graph.neighbours(state)

    def result(self, state: NodeID, action: Tuple[NodeID, Weight]) -> NodeID:
        return action[0]

    def step_cost(
        self,
        state: NodeID,
        action: Tuple[NodeID, Weight],
        next_state: NodeID,
    ) -> float:
        return action[1]

    def h(self, node: Node) -> float:
        dist = self.graph.distance(node.state, self.goal_state)
        return dist * self.heuristic_scale

    def to_dict(self) -> dict:
        return {
            "type": "graph",
            "graph": self.graph.to_dict(),
            "initial_state": self.initial_state,
            "goal_state": self.goal_state,
            "heuristic_scale": self.heuristic_scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphProblem":
        return cls(
            graph=WeightedGraph.from_dict(data["graph"]),
            initial_state=data["initial_state"],
            goal_state=data["goal_state"],
            heuristic_scale=data.get("heuristic_scale", 1.0),
        )


def generate_random_graph(
    n_nodes: int = 20,
    edge_density: float = 0.2,
    weight_range: Tuple[float, float] = (1.0, 10.0),
    seed: int = 42,
) -> Tuple[WeightedGraph, NodeID, NodeID]:
    rng = random.Random(seed)
    positions = {i: (rng.random(), rng.random()) for i in range(n_nodes)}
    g = WeightedGraph(n_nodes=n_nodes, positions=positions)

    w_min, w_max = weight_range
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if rng.random() < edge_density:
                weight = round(rng.uniform(w_min, w_max), 3)
                g.add_edge(u, v, weight)

    start, goal = 0, n_nodes - 1
    if not _bfs_reachable(g, start, goal):
        path = list(range(n_nodes))
        rng.shuffle(path)
        path.remove(start)
        path.remove(goal)
        path = [start] + path + [goal]
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            weight = round(rng.uniform(w_min, w_max), 3)
            g.add_edge(u, v, weight)

    return g, start, goal


def _bfs_reachable(graph: WeightedGraph, start: NodeID, goal: NodeID) -> bool:
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        for neighbour, _ in graph.neighbours(node):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return False

