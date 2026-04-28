"""
algorithms/dfs.py
-----------------
Depth-First Search

Properties (graph-search variant, visited set enabled)
-------------------------------------------------------
Complete  : Yes (for finite, acyclic state spaces)
Optimal   : No
Time      : O(b^m)   m = maximum depth
Space     : O(b·m)   — only the current path + siblings are in memory

Properties (tree-search variant, no visited set)
-------------------------------------------------
Complete  : No  (may loop indefinitely on cyclic graphs)
Optimal   : No

Implementation notes
--------------------
- Frontier    : Python list used as a LIFO stack (append / pop).
- graph_search: When True (default), a visited set prevents revisiting states.
                When False, tree-search mode is used — intentionally incomplete,
                used in Experiment D to demonstrate the failure case.
- depth_limit : Optional integer.  Nodes deeper than this are not expanded.
                Returns failure_reason='depth_limit' if limit is hit without
                finding the goal.
- The expansion_log records at every step when record_log=True.
"""

from __future__ import annotations
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

ALGO_NAME = "DFS"


def solve(
    problem:      Problem,
    heuristic=None,
    timeout:      float = 10.0,
    record_log:   bool  = False,
    graph_search: bool  = True,
    depth_limit:  Optional[int] = None,
) -> SearchResult:
    """
    Run Depth-First Search on *problem*.

    Parameters
    ----------
    problem      : Any Problem subclass.
    heuristic    : Ignored.
    timeout      : Wall-clock limit (enforced by BenchmarkRunner).
    record_log   : Populate expansion_log for visualiser.
    graph_search : True  → use visited set (safe on cyclic graphs).
                   False → tree-search (no visited set — may loop forever).
    depth_limit  : Maximum node depth to expand.  None = unlimited.

    Returns
    -------
    SearchResult.  failure_reason is 'no_path', 'depth_limit', or None.
    """
    stats            = SearchStats()
    stats.record_log = record_log

    root     = problem.root_node()
    frontier = [root]                         # LIFO stack
    visited  = {root.state} if graph_search else set()
    hit_limit = False

    with instrument(stats):
        while frontier:
            # ── Peak frontier size ────────────────────────────
            if len(frontier) > stats.max_frontier_size:
                stats.max_frontier_size = len(frontier)

            node = frontier.pop()
            stats.nodes_expanded += 1

            log_step(stats, node.state, [n.state for n in frontier])

            # ── Goal test ────────────────────────────────────
            if problem.goal_test(node.state):
                path_states, actions = finalise(stats, node, ALGO_NAME)
                return SearchResult(
                    stats     = stats,
                    path      = path_states,
                    actions   = actions,
                    algo_name = ALGO_NAME,
                )

            # ── Depth limit check ─────────────────────────────
            if depth_limit is not None and node.depth >= depth_limit:
                hit_limit = True
                continue

            # ── Expand (push in reverse so leftmost child pops first) ──
            for child in reversed(problem.expand(node)):
                stats.nodes_generated += 1
                if graph_search:
                    if child.state not in visited:
                        visited.add(child.state)
                        frontier.append(child)
                else:
                    frontier.append(child)   # tree-search: always push

    if hit_limit:
        stats.failure_reason = "depth_limit"
    else:
        stats.failure_reason = "no_path"

    return SearchResult(stats=stats, algo_name=ALGO_NAME)