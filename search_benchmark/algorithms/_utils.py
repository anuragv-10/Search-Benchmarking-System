"""
algorithms/_utils.py
---------------------
Internal helpers shared by all five search algorithms.

Not part of the public API — import from algorithms/ directly.
"""

from __future__ import annotations
import time
import tracemalloc
from contextlib import contextmanager
from typing import Generator
import os
import sys

try:
    # Normal case: imported as part of the package.
    from search_benchmark.algorithms.stats import SearchStats
except ImportError:
    # Fallback for running this module directly (no package context).
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.algorithms.stats import SearchStats


@contextmanager
def instrument(stats: SearchStats) -> Generator[None, None, None]:
    """
    Context manager that starts/stops timing and memory tracking.

    Usage
    -----
    stats = SearchStats()
    with instrument(stats):
        ... run search ...
    # stats.runtime_ms and stats.peak_memory_kb are now populated
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.runtime_ms    = (time.perf_counter() - t0) * 1000
        _, peak             = tracemalloc.get_traced_memory()
        stats.peak_memory_kb = peak / 1024
        tracemalloc.stop()


def finalise(stats: SearchStats, node, algo_name: str):
    """
    Called when a solution node is found.
    Populates solution_found, path_cost, solution_depth on stats.
    Returns (path_states, actions).
    """
    stats.solution_found = True
    stats.path_cost      = node.path_cost
    stats.solution_depth = node.depth
    path_nodes  = node.path()
    path_states = [n.state for n in path_nodes]
    actions     = node.solution()
    return path_states, actions


def log_step(stats: SearchStats, expanded_state, frontier_iterable):
    """Append one entry to expansion_log (only when record_log=True)."""
    if stats.record_log:
        stats.expansion_log.append({
            "expanded": expanded_state,
            "frontier": list(frontier_iterable),
        })