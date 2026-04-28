"""
analysis/heuristic_analysis.py
--------------------------------
Heuristic quality analysis tools.

Functions
---------
check_admissibility(problem, solution_path_nodes, h_fn)
    → AdmissibilityReport

check_consistency(problem, explored_edges, h_fn)
    → ConsistencyReport

compute_accuracy(reports)
    → float  (accuracy score 0–1)

full_heuristic_report(problem, result, h_fn)
    → dict  (all metrics combined — used by API)
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

try:
    from ..core.problem import Node, Problem
except ImportError:
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.core.problem import Node, Problem


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdmissibilityReport:
    """Result of checking h(n) ≤ h*(n) on solution path nodes."""
    total_nodes:        int   = 0
    admissible_nodes:   int   = 0
    violations:         int   = 0
    admissibility_rate: float = 1.0        # admissible_nodes / total_nodes
    mean_overestimate:  float = 0.0        # mean(max(h(n)-h*(n), 0)) over violations
    max_overestimate:   float = 0.0        # worst single violation
    is_admissible:      bool  = True       # True iff zero violations

    def as_dict(self) -> dict:
        return {
            "total_nodes":        self.total_nodes,
            "admissible_nodes":   self.admissible_nodes,
            "violations":         self.violations,
            "admissibility_rate": round(self.admissibility_rate, 4),
            "mean_overestimate":  round(self.mean_overestimate, 6),
            "max_overestimate":   round(self.max_overestimate, 6),
            "is_admissible":      self.is_admissible,
        }


@dataclass
class ConsistencyReport:
    """Result of checking h(n) ≤ c(n,a,n') + h(n') on explored edges."""
    total_edges:        int   = 0
    consistent_edges:   int   = 0
    violations:         int   = 0
    consistency_rate:   float = 1.0
    max_inconsistency:  float = 0.0        # worst delta: h(n) - c - h(n')
    is_consistent:      bool  = True

    def as_dict(self) -> dict:
        return {
            "total_edges":       self.total_edges,
            "consistent_edges":  self.consistent_edges,
            "violations":        self.violations,
            "consistency_rate":  round(self.consistency_rate, 4),
            "max_inconsistency": round(self.max_inconsistency, 6),
            "is_consistent":     self.is_consistent,
        }


# ─────────────────────────────────────────────────────────────────────────────
# UCS oracle — compute h*(state) for any state
# ─────────────────────────────────────────────────────────────────────────────

def _h_star(problem: Problem, start_state: Any) -> Optional[float]:
    """
    Return the true optimal cost from *start_state* to the goal via mini-UCS.
    Returns None if the goal is unreachable from start_state.
    """
    root        = Node(state=start_state)
    counter     = 0
    frontier    = [(0.0, counter, root)]
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


# ─────────────────────────────────────────────────────────────────────────────
# Admissibility checker
# ─────────────────────────────────────────────────────────────────────────────

def check_admissibility(
    problem:           Problem,
    solution_nodes:    List[Node],
    h_fn:              Callable[[Node], float],
    max_sample:        int = 30,
) -> AdmissibilityReport:
    """
    Check h(n) ≤ h*(n) for nodes on the solution path.

    Parameters
    ----------
    problem        : The problem instance (used to run mini-UCS for h*).
    solution_nodes : List of Node objects from root to goal (node.path()).
    h_fn           : Heuristic function (node → float).
    max_sample     : Maximum number of nodes to sample (even sampling).

    Returns
    -------
    AdmissibilityReport
    """
    if not solution_nodes:
        return AdmissibilityReport()

    # Sample evenly
    n     = len(solution_nodes)
    step  = max(1, n // max_sample)
    nodes = solution_nodes[::step]
    # Always include the goal
    if solution_nodes[-1] not in nodes:
        nodes.append(solution_nodes[-1])

    report      = AdmissibilityReport(total_nodes=len(nodes))
    overestimates = []

    for node in nodes:
        h_est    = h_fn(node)
        h_actual = _h_star(problem, node.state)
        if h_actual is None:
            continue
        if h_est <= h_actual + 1e-9:
            report.admissible_nodes += 1
        else:
            report.violations     += 1
            overestimates.append(h_est - h_actual)

    if report.total_nodes > 0:
        report.admissibility_rate = report.admissible_nodes / report.total_nodes
    if overestimates:
        report.mean_overestimate = sum(overestimates) / len(overestimates)
        report.max_overestimate  = max(overestimates)
    report.is_admissible = (report.violations == 0)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Consistency checker
# ─────────────────────────────────────────────────────────────────────────────

def check_consistency(
    problem:        Problem,
    nodes_to_check: List[Node],
    h_fn:           Callable[[Node], float],
) -> ConsistencyReport:
    """
    Check h(n) ≤ step_cost(n,a,n') + h(n') for edges reachable from nodes_to_check.

    Parameters
    ----------
    problem        : The problem instance.
    nodes_to_check : Nodes whose outgoing edges are checked.
    h_fn           : Heuristic function.

    Returns
    -------
    ConsistencyReport
    """
    report = ConsistencyReport()
    inconsistencies = []

    for node in nodes_to_check:
        h_n = h_fn(node)
        for child in problem.expand(node):
            cost     = problem.step_cost(node.state, child.action, child.state)
            h_child  = h_fn(child)
            lhs      = h_n
            rhs      = cost + h_child
            report.total_edges += 1
            if lhs <= rhs + 1e-9:
                report.consistent_edges += 1
            else:
                report.violations      += 1
                inconsistencies.append(lhs - rhs)

    if report.total_edges > 0:
        report.consistency_rate = report.consistent_edges / report.total_edges
    if inconsistencies:
        report.max_inconsistency = max(inconsistencies)
    report.is_consistent = (report.violations == 0)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy score
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(
    problem:        Problem,
    solution_nodes: List[Node],
    h_fn:           Callable[[Node], float],
    max_sample:     int = 20,
) -> float:
    """
    Heuristic accuracy = 1 − mean_relative_error.

    relative_error(n) = |h(n) − h*(n)| / max(h*(n), 1)

    Returns a value in [0, 1].  1.0 = perfect heuristic.
    """
    if not solution_nodes:
        return 1.0

    n    = len(solution_nodes)
    step = max(1, n // max_sample)
    nodes = solution_nodes[::step]

    errors = []
    for node in nodes:
        h_est    = h_fn(node)
        h_actual = _h_star(problem, node.state)
        if h_actual is not None:
            rel_err = abs(h_est - h_actual) / max(h_actual, 1.0)
            errors.append(rel_err)

    if not errors:
        return 1.0
    return round(1.0 - sum(errors) / len(errors), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Combined report (for API and notebooks)
# ─────────────────────────────────────────────────────────────────────────────

def full_heuristic_report(
    problem:     Problem,
    result,                      # SearchResult
    h_fn:        Optional[Callable[[Node], float]] = None,
    max_sample:  int = 20,
) -> dict:
    """
    Run all three analyses on a solved problem and return a combined dict.

    Parameters
    ----------
    problem    : Problem instance.
    result     : SearchResult from any algorithm.
    h_fn       : Heuristic to analyse.  Defaults to problem.h.
    max_sample : Max nodes sampled per check.

    Returns
    -------
    dict with keys: admissibility, consistency, accuracy, heuristic_error
    """
    h_fn = h_fn or problem.h

    if not result.found or result.path is None:
        return {
            "admissibility": None,
            "consistency":   None,
            "accuracy":      None,
            "note":          "No solution found — cannot analyse heuristic.",
        }

    # Reconstruct Node path from result.path (states only)
    # We need Node objects for h_fn; rebuild cheaply
    nodes = []
    cost  = 0.0
    for i, state in enumerate(result.path):
        node = Node(state=state, depth=i, path_cost=cost)
        if i > 0:
            prev  = result.path[i - 1]
            acts  = problem.actions(prev)
            # find matching action
            for a in acts:
                if problem.result(prev, a) == state:
                    cost += problem.step_cost(prev, a, state)
                    node  = Node(state=state, depth=i, path_cost=cost)
                    break
        nodes.append(node)

    adm  = check_admissibility(problem, nodes, h_fn, max_sample)
    con  = check_consistency(problem, nodes[:max_sample], h_fn)
    acc  = compute_accuracy(problem, nodes, h_fn, max_sample)

    return {
        "admissibility":    adm.as_dict(),
        "consistency":      con.as_dict(),
        "accuracy":         acc,
        "heuristic_error":  result.stats.heuristic_error,
    }
