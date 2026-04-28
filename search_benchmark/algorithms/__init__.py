"""Algorithm implementations and search statistics utilities.

Public API
----------
- SearchStats, SearchResult  (from .stats)
- instrument, finalise, log_step  (from ._utils)
"""

from .stats import SearchStats, SearchResult
from ._utils import instrument, finalise, log_step

from . import bfs, dfs, ucs, astar, gbfs

ALGORITHMS = {
    "BFS": bfs,
    "DFS": dfs,
    "UCS": ucs,
    "Greedy": gbfs,
    "A*": astar,
}

__all__ = [
    "SearchStats",
    "SearchResult",
    "instrument",
    "finalise",
    "log_step",
    "ALGORITHMS",
]
