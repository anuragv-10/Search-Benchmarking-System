"""
core/grid.py
------------
2-D grid / maze environment and its Problem subclass.
"""

from __future__ import annotations
import random
from collections import deque
from typing import List, Tuple

import numpy as np

try:
    from .problem import Node, Problem
    from .heuristics import manhattan, euclidean, chebyshev, zero
except ImportError:
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.core.problem import Node, Problem
    from search_benchmark.core.heuristics import manhattan, euclidean, chebyshev, zero

State = Tuple[int, int]
Action = Tuple[int, int]

_CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ALL_EIGHT = _CARDINAL + _DIAGONAL


class Grid:
    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=np.uint8)
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]

    def is_free(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols and self.data[row, col] == 0

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "data": self.data.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Grid":
        return cls(np.array(data["data"], dtype=np.uint8))

    def __repr__(self) -> str:
        walls = int(self.data.sum())
        free = self.rows * self.cols - walls
        return f"Grid({self.rows}x{self.cols}, free={free}, walls={walls})"


class GridProblem(Problem):
    _HEURISTIC_MAP = {
        "manhattan": manhattan,
        "euclidean": euclidean,
        "chebyshev": chebyshev,
        "zero": zero,
    }

    def __init__(
        self,
        grid: Grid,
        initial_state: State,
        goal_state: State,
        eight_directional: bool = False,
        heuristic_name: str = "manhattan",
        heuristic_scale: float = 1.0,
    ) -> None:
        super().__init__(initial_state, goal_state)
        self.grid = grid
        self.eight_directional = eight_directional
        self.heuristic_scale = heuristic_scale

        if heuristic_name not in self._HEURISTIC_MAP:
            raise ValueError(
                f"Unknown heuristic '{heuristic_name}'. "
                f"Choose from {list(self._HEURISTIC_MAP)}."
            )
        self._heuristic_fn = self._HEURISTIC_MAP[heuristic_name]
        self.heuristic_name = heuristic_name

    def actions(self, state: State) -> List[Action]:
        moves = _ALL_EIGHT if self.eight_directional else _CARDINAL
        row, col = state
        valid = []
        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            if self.grid.is_free(nr, nc):
                valid.append((dr, dc))
        return valid

    def result(self, state: State, action: Action) -> State:
        return (state[0] + action[0], state[1] + action[1])

    def step_cost(self, state: State, action: Action, next_state: State) -> float:
        dr, dc = action
        return 1.414 if (dr != 0 and dc != 0) else 1.0

    def h(self, node: Node) -> float:
        return self._heuristic_fn(node.state, self.goal_state) * self.heuristic_scale

    def to_dict(self) -> dict:
        return {
            "type": "grid",
            "grid": self.grid.to_dict(),
            "initial_state": list(self.initial_state),
            "goal_state": list(self.goal_state),
            "eight_directional": self.eight_directional,
            "heuristic_name": self.heuristic_name,
            "heuristic_scale": self.heuristic_scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridProblem":
        return cls(
            grid=Grid.from_dict(data["grid"]),
            initial_state=tuple(data["initial_state"]),
            goal_state=tuple(data["goal_state"]),
            eight_directional=data.get("eight_directional", False),
            heuristic_name=data.get("heuristic_name", "manhattan"),
            heuristic_scale=data.get("heuristic_scale", 1.0),
        )


def generate_maze(
    rows: int = 10,
    cols: int = 10,
    wall_density: float = 0.3,
    seed: int = 42,
    heuristic: str = "manhattan",
) -> Tuple[GridProblem, Grid]:
    wall_density = min(wall_density, 0.65)
    rng = random.Random(seed)

    start: State = (0, 0)
    goal: State = (rows - 1, cols - 1)

    for _ in range(100):
        data = np.zeros((rows, cols), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                if (r, c) in (start, goal):
                    continue
                if rng.random() < wall_density:
                    data[r, c] = 1

        if _flood_reachable(data, start, goal, rows, cols):
            grid = Grid(data)
            problem = GridProblem(
                grid=grid,
                initial_state=start,
                goal_state=goal,
                heuristic_name=heuristic,
            )
            return problem, grid

    data = np.ones((rows, cols), dtype=np.uint8)
    for c in range(cols):
        data[0, c] = 0
    for r in range(rows):
        data[r, cols - 1] = 0

    grid = Grid(data)
    problem = GridProblem(
        grid=grid,
        initial_state=start,
        goal_state=goal,
        heuristic_name=heuristic,
    )
    return problem, grid


def generate_unsolvable_maze(rows: int = 10, cols: int = 10) -> Tuple[GridProblem, Grid]:
    data = np.zeros((rows, cols), dtype=np.uint8)
    goal_r, goal_c = rows - 1, cols - 1

    for dr in [-1, 0]:
        for dc in [-1, 0]:
            r, c = goal_r + dr, goal_c + dc
            if (r, c) != (0, 0) and 0 <= r < rows and 0 <= c < cols:
                data[r, c] = 1

    grid = Grid(data)
    problem = GridProblem(
        grid=grid,
        initial_state=(0, 0),
        goal_state=(goal_r, goal_c),
    )
    return problem, grid


def _flood_reachable(
    data: np.ndarray,
    start: State,
    goal: State,
    rows: int,
    cols: int,
) -> bool:
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in _CARDINAL:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and data[nr, nc] == 0
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False

