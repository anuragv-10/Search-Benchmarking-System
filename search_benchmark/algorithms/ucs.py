"""
algorithms/ucs.py
-----------------
Uniform Cost Search  (Dijkstra-style)

Properties
----------
Complete  : Yes
Optimal   : Yes  (minimises total path cost)
Time      : O(b ^ (1 + floor(C* / ε)))   C* = optimal cost, ε = min edge cost
Space     : O(b ^ (1 + floor(C* / ε)))

Implementation notes
--------------------
- Frontier  : heapq of (path_cost, counter, node).
              The counter breaks ties deterministically so Node.__lt__
              is never reached with equal keys (avoids heapq comparison
              of non-comparable states on some problem types).
- Lazy deletion pattern:
    When a cheaper path to a state is found we push a new entry.
    On pop, we skip the entry if we've already expanded that state
    at a lower cost (cost_so_far check).
- visited dict maps state → best known g(state).
- UCS is used as the *ground truth* path-cost oracle:
    The benchmarking layer calls UCS to obtain h*(n) for heuristic
    admissibility checking.
"""

from __future__ import annotations
import heapq
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

ALGO_NAME = "UCS"


def solve(
    problem:    Problem,
    heuristic=None,
    timeout:    float = 10.0,
    record_log: bool  = False,
) -> SearchResult:
    """
    Run Uniform Cost Search on *problem*.

    Parameters
    ----------
    problem    : Any Problem subclass.
    heuristic  : Ignored (UCS is uninformed).
    timeout    : Wall-clock limit (enforced by BenchmarkRunner).
    record_log : Populate expansion_log for visualiser.

    Returns
    -------
    SearchResult with cost-optimal path (if reachable).
    """
    stats            = SearchStats()
    stats.record_log = record_log

    root      = problem.root_node()
    counter   = 0                             # tie-breaker
    frontier  = [(0.0, counter, root)]        # min-heap
    cost_so_far: dict = {root.state: 0.0}    # state → best g seen so far

    with instrument(stats):
        while frontier:
            if len(frontier) > stats.max_frontier_size:
                stats.max_frontier_size = len(frontier)

            g, _, node = heapq.heappop(frontier)

            # ── Lazy deletion: skip stale entries ────────────
            if g > cost_so_far.get(node.state, float("inf")):
                continue

            stats.nodes_expanded += 1
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
                new_cost = child.path_cost
                if new_cost < cost_so_far.get(child.state, float("inf")):
                    cost_so_far[child.state] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, child))

    stats.failure_reason = "no_path"
    return SearchResult(stats=stats, algo_name=ALGO_NAME)