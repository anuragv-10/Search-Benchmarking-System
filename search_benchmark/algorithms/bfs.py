"""
algorithms/bfs.py
-----------------
Breadth-First Search

Properties
----------
Complete  : Yes (for finite state spaces)
Optimal   : Yes (when all step costs are equal / unit cost)
Time      : O(b^d)
Space     : O(b^d)  — frontier holds an entire level at once

Implementation notes
--------------------
- Frontier  : collections.deque  (O(1) popleft)
- Visited   : set of states  (prevents re-expansion)
- Goal test : applied at *expansion* time, not generation time.
              This matches the theoretical guarantee.
- The expansion_log records (expanded_state, frontier_states) at
  every step when stats.record_log is True — used by the frontend
  visualiser to animate the search.
"""

from __future__ import annotations
from collections import deque
from typing import Optional

try:
    from ..core.problem import Problem
    from .stats import SearchStats, SearchResult
    from ._utils import instrument, finalise, log_step
except ImportError:
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.core.problem import Problem
    from search_benchmark.algorithms.stats import SearchStats, SearchResult
    from search_benchmark.algorithms._utils import instrument, finalise, log_step

ALGO_NAME = "BFS"


def solve(
    problem:    Problem,
    heuristic=None,           # accepted but ignored — BFS is uninformed
    timeout:    float = 10.0, # seconds; enforced externally by BenchmarkRunner
    record_log: bool  = False,
) -> SearchResult:
    """
    Run Breadth-First Search on *problem*.

    Parameters
    ----------
    problem    : Any Problem subclass (GraphProblem or GridProblem).
    heuristic  : Ignored.  Present so all algorithms share the same signature.
    timeout    : Wall-clock time limit in seconds.
                 BFS itself does not enforce this — the BenchmarkRunner wraps
                 the call in a thread with a timeout.
    record_log : If True, populate stats.expansion_log for visualisation.

    Returns
    -------
    SearchResult with populated stats and path (if solution found).
    """
    stats            = SearchStats()
    stats.record_log = record_log

    root    = problem.root_node()
    visited = {root.state}
    frontier: deque = deque([root])

    with instrument(stats):
        while frontier:
            # ── Track peak frontier size ───────────────────────
            if len(frontier) > stats.max_frontier_size:
                stats.max_frontier_size = len(frontier)

            node = frontier.popleft()
            stats.nodes_expanded += 1

            # ── Expansion log (for visualiser) ────────────────
            log_step(stats, node.state, [n.state for n in frontier])

            # ── Goal test ─────────────────────────────────────
            if problem.goal_test(node.state):
                path_states, actions = finalise(stats, node, ALGO_NAME)
                return SearchResult(
                    stats     = stats,
                    path      = path_states,
                    actions   = actions,
                    algo_name = ALGO_NAME,
                )

            # ── Expand ────────────────────────────────────────
            for child in problem.expand(node):
                stats.nodes_generated += 1
                if child.state not in visited:
                    visited.add(child.state)
                    frontier.append(child)

    # Frontier exhausted — no solution
    stats.failure_reason = "no_path"
    return SearchResult(stats=stats, algo_name=ALGO_NAME)