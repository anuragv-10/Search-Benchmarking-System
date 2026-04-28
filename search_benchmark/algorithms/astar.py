"""
algorithms/astar.py
-------------------
A* Search

Properties
----------
Complete  : Yes
Optimal   : Yes, IF h(n) is admissible  (h(n) ≤ h*(n) for all n)
Time      : O(b^d)  with a perfect heuristic; degrades to O(b^(C*/ε))
            with h=0 (identical to UCS)
Space     : O(b^d)  — must keep all generated nodes in memory

Implementation notes
--------------------
- Frontier  : heapq keyed on f(n) = g(n) + h(n).
- Lazy deletion: same pattern as UCS/GBFS.
- Re-expansion tracking: if a node is popped whose g > best_g[state]
  that means we already expanded it via a cheaper path — this only
  happens when h is *inconsistent*.  We count these as re_expansions.
- Post-solve heuristic error:
    After finding the solution, we compute h*(n) for every node on the
    solution path by running a mini UCS from that node to the goal.
    This lets us report  mean |h(n) − h*(n)|  — the true heuristic error.
    We cap this at 20 nodes to avoid excessive post-processing time on
    long paths; the sample is still statistically meaningful.
"""

from __future__ import annotations
import heapq
from typing import Callable, List, Optional

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

ALGO_NAME = "A*"


def solve(
    problem:    Problem,
    heuristic:  Optional[Callable[[Node], float]] = None,
    timeout:    float = 10.0,
    record_log: bool  = False,
) -> SearchResult:
    """
    Run A* Search on *problem*.

    Parameters
    ----------
    problem    : Any Problem subclass.
    heuristic  : Optional callable (node → float).  If None, uses problem.h.
                 Pass a scaled/inadmissible heuristic for stress experiments.
    timeout    : Wall-clock limit (enforced by BenchmarkRunner).
    record_log : Populate expansion_log for visualiser.

    Returns
    -------
    SearchResult — optimal path if h is admissible.
    """
    stats            = SearchStats()
    stats.record_log = record_log

    h_fn = heuristic if heuristic is not None else problem.h

    root    = problem.root_node()
    counter = 0
    f_root  = h_fn(root)
    frontier: list = [(f_root, counter, root)]
    best_g: dict   = {root.state: 0.0}   # state → best g(n) seen so far

    with instrument(stats):
        while frontier:
            if len(frontier) > stats.max_frontier_size:
                stats.max_frontier_size = len(frontier)

            f_val, _, node = heapq.heappop(frontier)

            # ── Re-expansion / lazy deletion check ───────────
            current_best_g = best_g.get(node.state, float("inf"))
            if node.path_cost > current_best_g:
                stats.re_expansions += 1
                continue

            stats.nodes_expanded += 1
            log_step(stats, node.state, [e[2].state for e in frontier])

            # ── Goal test ────────────────────────────────────
            if problem.goal_test(node.state):
                path_states, actions = finalise(stats, node, ALGO_NAME)
                # ── Post-solve: compute heuristic error ──────
                stats.heuristic_error = _heuristic_error(
                    node, problem, h_fn
                )
                return SearchResult(
                    stats     = stats,
                    path      = path_states,
                    actions   = actions,
                    algo_name = ALGO_NAME,
                )

            # ── Expand ───────────────────────────────────────
            for child in problem.expand(node):
                stats.nodes_generated += 1
                new_g = child.path_cost
                if new_g < best_g.get(child.state, float("inf")):
                    best_g[child.state] = new_g
                    h_child = h_fn(child)
                    f_child = new_g + h_child
                    counter += 1
                    heapq.heappush(frontier, (f_child, counter, child))

    stats.failure_reason = "no_path"
    return SearchResult(stats=stats, algo_name=ALGO_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic error computation
# ─────────────────────────────────────────────────────────────────────────────

def _ucs_cost(problem: Problem, start_state) -> Optional[float]:
    """
    Run a mini UCS from *start_state* to the goal.
    Returns the true optimal cost h*(start_state), or None if unreachable.
    We reuse the same Problem interface but override the initial state.
    """
    # Build a lightweight temporary problem with new start
    try:
        from ..core.problem import Node as _Node
    except ImportError:
        from search_benchmark.core.problem import Node as _Node
    root      = _Node(state=start_state)
    counter   = 0
    frontier  = [(0.0, counter, root)]
    cost_so_far = {start_state: 0.0}

    while frontier:
        g, _, node = heapq.heappop(frontier)
        if g > cost_so_far.get(node.state, float("inf")):
            continue
        if problem.goal_test(node.state):
            return g
        for child in problem.expand(node):
            ng = child.path_cost
            if ng < cost_so_far.get(child.state, float("inf")):
                cost_so_far[child.state] = ng
                counter += 1
                heapq.heappush(frontier, (ng, counter, child))
    return None


def _heuristic_error(
    solution_node: Node,
    problem:       Problem,
    h_fn:          Callable[[Node], float],
    max_sample:    int = 20,
) -> Optional[float]:
    """
    Compute mean |h(n) − h*(n)| over nodes on the solution path.

    We sample up to *max_sample* evenly-spaced nodes from the path
    to keep post-processing time bounded on long paths.
    """
    path_nodes = solution_node.path()
    if len(path_nodes) <= 1:
        return 0.0

    # Evenly sample up to max_sample nodes (always include root + goal)
    indices = _sample_indices(len(path_nodes), max_sample)
    errors  = []

    for i in indices:
        n        = path_nodes[i]
        h_est    = h_fn(n)
        h_actual = _ucs_cost(problem, n.state)
        if h_actual is not None:
            errors.append(abs(h_est - h_actual))

    if not errors:
        return None
    return round(sum(errors) / len(errors), 6)


def _sample_indices(total: int, k: int) -> List[int]:
    """Return up to k evenly-spaced indices in [0, total-1]."""
    if total <= k:
        return list(range(total))
    step = total / k
    return sorted(set(
        [0, total - 1] + [round(i * step) for i in range(1, k - 1)]
    ))