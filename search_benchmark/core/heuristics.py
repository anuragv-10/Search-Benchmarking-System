"""
core/heuristics.py
------------------
Standalone heuristic functions used by GraphProblem, GridProblem,
and injected into algorithms at call-time.

Every function has the signature:
    heuristic(state, goal, **env_data) -> float

env_data is problem-specific context (e.g. node positions for graphs).
Problems wrap these into their h(node) method automatically.
"""

from __future__ import annotations
import math
from typing import Tuple

Coord = Tuple[float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Generic distance heuristics
# ─────────────────────────────────────────────────────────────────────────────

def euclidean(a: Coord, b: Coord) -> float:
    """
    Straight-line (Euclidean) distance between two 2-D coordinates.
    Admissible for any problem where the step cost >= actual distance moved.
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def manhattan(a: Coord, b: Coord) -> float:
    """
    Manhattan (taxicab) distance — sum of absolute coordinate differences.
    Admissible for 4-directional grid movement with unit step costs.
    """
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def chebyshev(a: Coord, b: Coord) -> float:
    """
    Chebyshev distance — maximum of absolute coordinate differences.
    Admissible for 8-directional grid movement with unit step costs.
    """
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


def zero(a: Coord, b: Coord) -> float:
    """
    Zero heuristic — makes A* behave identically to UCS.
    Always admissible; useful as a correctness baseline.
    """
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Scaled / inadmissible variants  (used in Experiment C — stress testing)
# ─────────────────────────────────────────────────────────────────────────────

def scaled_manhattan(scale: float):
    """
    Returns a Manhattan heuristic scaled by *scale*.
    scale > 1.0  →  inadmissible  (over-estimates; A* no longer optimal).
    scale < 1.0  →  still admissible but less informed.
    """

    def h(a: Coord, b: Coord) -> float:
        return scale * manhattan(a, b)

    h.__name__ = f"manhattan_x{scale}"
    return h


def scaled_euclidean(scale: float):
    """
    Returns a Euclidean heuristic scaled by *scale*.
    scale > 1.0  →  inadmissible.
    """

    def h(a: Coord, b: Coord) -> float:
        return scale * euclidean(a, b)

    h.__name__ = f"euclidean_x{scale}"
    return h


# ─────────────────────────────────────────────────────────────────────────────
# Registry  — maps string names to callables (used by API / agent)
# ─────────────────────────────────────────────────────────────────────────────

HEURISTIC_REGISTRY = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "chebyshev": chebyshev,
    "zero": zero,
    "manhattan_x1.5": scaled_manhattan(1.5),
    "manhattan_x2.0": scaled_manhattan(2.0),
    "manhattan_x3.0": scaled_manhattan(3.0),
}


def get_heuristic(name: str):
    """Look up a heuristic by string name. Raises KeyError for unknown names."""
    if name not in HEURISTIC_REGISTRY:
        raise KeyError(
            f"Unknown heuristic '{name}'. "
            f"Available: {list(HEURISTIC_REGISTRY.keys())}"
        )
    return HEURISTIC_REGISTRY[name]

