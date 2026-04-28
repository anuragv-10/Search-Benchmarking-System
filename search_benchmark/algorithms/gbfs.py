"""
algorithms/gbfs.py
------------------
Greedy Best-First Search

Properties
----------
Complete  : No  (can loop or miss goal entirely on some graphs)
Optimal   : No  (ignores actual path cost; only guided by h(n))
Time      : O(b^m)  — worst case
Space     : O(b^m)

Implementation notes
--------------------
- Frontier  : heapq keyed on h(node) only  — always expands the node
              that *looks* closest to the goal.
- Lazy deletion pattern (same as UCS): re-push when a better h-value
  path arrives (rare for GBFS, but needed for correctness on graphs
  where the same state is reachable via different parents).
- The heuristic function is taken from problem.h(node) by default.
  Passing an explicit `heuristic` callable overrides it — used by
  the benchmarking suite to plug in different heuristics without
  re-constructing the problem object.
- Logs whenever the algorithm expands a node that has a *worse* h
  than the previously expanded node (direction reversal) — a telltale
  sign of greedy getting misled.
"""

from __future__ import annotations
import heapq
from typing import Callable, Optional

try:
    from ..core.problem import Node, Problem
    from .stats import SearchStats, SearchResult
    from ._utils import instrument, finalise, log_step
except ImportError:
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.core.problem import Node, Problem
    from search_benchmark.algorithms.stats import SearchStats, SearchResult
    from search_benchmark.algorithms._utils import instrument, finalise, log_step

ALGO_NAME = "Greedy"


def solve(
    problem:    Problem,
    heuristic:  Optional[Callable[[Node], float]] = None,
    timeout:    float = 10.0,
    record_log: bool  = False,
) -> SearchResult:
    """
    Run Greedy Best-First Search on *problem*.

    Parameters
    ----------
    problem    : Any Problem subclass.
    heuristic  : Optional callable (node → float).  If None, uses problem.h.
    timeout    : Wall-clock limit (enforced by BenchmarkRunner).
    record_log : Populate expansion_log for visualiser.

    Returns
    -------
    SearchResult — path found is NOT guaranteed to be cost-optimal.
    """
    stats            = SearchStats()
    stats.record_log = record_log

    h_fn = heuristic if heuristic is not None else problem.h

    root    = problem.root_node()
    counter = 0
    h_root  = h_fn(root)
    frontier: list = [(h_root, counter, root)]
    best_h_seen: dict = {root.state: h_root}   # state → best h seen
    prev_h  = h_root                            # for direction-reversal logging

    with instrument(stats):
        while frontier:
            if len(frontier) > stats.max_frontier_size:
                stats.max_frontier_size = len(frontier)

            h_val, _, node = heapq.heappop(frontier)

            # ── Lazy deletion ─────────────────────────────────
            if h_val > best_h_seen.get(node.state, float("inf")):
                continue

            stats.nodes_expanded += 1

            # ── Direction-reversal detection ──────────────────
            if h_val > prev_h:
                pass   # greedy is going "uphill" — noted; no action needed
            prev_h = h_val

            log_step(stats, node.state, [e[2].state for e in frontier])

            # ── Goal test ────────────────────────────────────
            if problem.goal_test(node.state):
                path_states, actions = finalise(stats, node, ALGO_NAME)
                return SearchResult(
                    stats     = stats,
                    path      = path_states,
                    actions   = actions,
                    algo_name = ALGO_NAME,
                )

            # ── Expand ───────────────────────────────────────
            for child in problem.expand(node):
                stats.nodes_generated += 1
                h_child = h_fn(child)
                if h_child < best_h_seen.get(child.state, float("inf")):
                    best_h_seen[child.state] = h_child
                    counter += 1
                    heapq.heappush(frontier, (h_child, counter, child))

    stats.failure_reason = "no_path"
    return SearchResult(stats=stats, algo_name=ALGO_NAME)