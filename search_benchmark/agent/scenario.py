"""
agent/scenarios.py
------------------
Four concrete demo scenarios that show the agent making different
decisions based on different constraints.

Each scenario function returns a ScenarioResult containing:
  - the Recommendation from the selector
  - the actual SearchResult from running the recommended algorithm
  - a narrative summary of what happened
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

try:
    from ..core.graph import GraphProblem, generate_random_graph
    from ..core.grid  import Grid, GridProblem, generate_maze
    from ..algorithms import ALGORITHMS
    from ..algorithms.stats import SearchResult
    from ..benchmarking.runner import BenchmarkRunner
    from .profiles import ProfileStore
    from .selector import Recommendation, StrategySelector
except ImportError:
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.core.graph import GraphProblem, generate_random_graph
    from search_benchmark.core.grid  import Grid, GridProblem, generate_maze
    from search_benchmark.algorithms import ALGORITHMS
    from search_benchmark.algorithms.stats import SearchResult
    from search_benchmark.benchmarking.runner import BenchmarkRunner
    from search_benchmark.agent.profiles import ProfileStore
    from search_benchmark.agent.selector import Recommendation, StrategySelector


@dataclass
class ScenarioResult:
    scenario_name:  str
    recommendation: Recommendation
    search_result:  Optional[SearchResult]
    summary:        str


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_recommended(
    problem,
    rec: Recommendation,
    timeout: float = 10.0,
) -> SearchResult:
    """
    Run the recommended algorithm.  If it fails, try fallbacks in order.
    """
    runner = BenchmarkRunner(timeout_seconds=timeout)
    to_try = [rec.primary] + rec.fallback_order

    for algo in to_try:
        if algo not in ALGORITHMS:
            continue
        result = runner.run_single(problem, algo)
        if result.found:
            return result

    # All failed — return last result
    return runner.run_single(problem, rec.primary)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Speed Priority
# ─────────────────────────────────────────────────────────────────────────────

def scenario_speed_priority(
    profiles: Optional[ProfileStore] = None,
    verbose:  bool = True,
) -> ScenarioResult:
    """
    Problem   : 100-node weighted graph
    Constraint: time_limit=50ms, optimality NOT required
    Weights   : speed=0.7, memory=0.2, quality=0.1
    Expected  : Greedy Best-First (fastest, fewest nodes)
    """
    selector = StrategySelector(profiles or ProfileStore.default())
    rec = selector.recommend(
        env_type            = "graph",
        problem_size        = 100,
        optimality_required = False,
        time_limit_ms       = 50.0,
        speed_weight        = 0.7,
        memory_weight       = 0.2,
        quality_weight      = 0.1,
    )

    g, start, goal = generate_random_graph(100, seed=42)
    problem = GraphProblem(g, start, goal)
    result  = _run_recommended(problem, rec, timeout=5.0)

    summary = (
        f"Scenario 1 — Speed Priority\n"
        f"  Recommended : {rec.primary}\n"
        f"  Solution    : {'FOUND' if result.found else 'NOT FOUND'}\n"
        f"  Runtime     : {result.stats.runtime_ms:.2f} ms\n"
        f"  Nodes       : {result.stats.nodes_expanded}\n"
        f"  Path cost   : {result.stats.path_cost}\n"
        f"  Note        : Optimality NOT required — fast heuristic path acceptable"
    )
    if verbose:
        print(summary)

    return ScenarioResult(
        scenario_name  = "Speed Priority",
        recommendation = rec,
        search_result  = result,
        summary        = summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Optimality Priority
# ─────────────────────────────────────────────────────────────────────────────

def scenario_optimality_priority(
    profiles: Optional[ProfileStore] = None,
    verbose:  bool = True,
) -> ScenarioResult:
    """
    Problem   : 20×20 grid maze
    Constraint: optimality_required=True
    Weights   : quality=0.6, speed=0.3, memory=0.1
    Expected  : A* (optimal + efficient) or UCS (optimal but slower)
    """
    selector = StrategySelector(profiles or ProfileStore.default())
    rec = selector.recommend(
        env_type            = "grid",
        problem_size        = 20,
        optimality_required = True,
        speed_weight        = 0.3,
        memory_weight       = 0.1,
        quality_weight      = 0.6,
    )

    problem, _ = generate_maze(20, 20, wall_density=0.25, seed=7)
    result      = _run_recommended(problem, rec, timeout=10.0)

    # Verify optimality: compare with UCS
    runner    = BenchmarkRunner(timeout_seconds=10.0)
    ucs_result = runner.run_single(problem, "UCS")
    is_optimal = (
        result.found and ucs_result.found and
        abs(result.stats.path_cost - ucs_result.stats.path_cost) < 1e-6
    )

    summary = (
        f"Scenario 2 — Optimality Priority\n"
        f"  Recommended : {rec.primary}\n"
        f"  Solution    : {'FOUND' if result.found else 'NOT FOUND'}\n"
        f"  Path cost   : {result.stats.path_cost}\n"
        f"  UCS cost    : {ucs_result.stats.path_cost}\n"
        f"  Optimal     : {'YES ✓' if is_optimal else 'NO ✗'}\n"
        f"  Nodes       : {result.stats.nodes_expanded}\n"
        f"  Note        : DFS and Greedy were eliminated (not optimal)"
    )
    if verbose:
        print(summary)

    return ScenarioResult(
        scenario_name  = "Optimality Priority",
        recommendation = rec,
        search_result  = result,
        summary        = summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Memory Constrained
# ─────────────────────────────────────────────────────────────────────────────

def scenario_memory_constrained(
    profiles: Optional[ProfileStore] = None,
    verbose:  bool = True,
) -> ScenarioResult:
    """
    Problem   : 30×30 grid maze
    Constraint: memory_limit=80KB, optimality NOT required
    Weights   : memory=0.6, speed=0.3, quality=0.1
    Expected  : DFS (lowest memory footprint) or Greedy
    """
    selector = StrategySelector(profiles or ProfileStore.default())
    rec = selector.recommend(
        env_type            = "grid",
        problem_size        = 30,
        optimality_required = False,
        memory_limit_kb     = 80.0,
        speed_weight        = 0.3,
        memory_weight       = 0.6,
        quality_weight      = 0.1,
    )

    problem, _ = generate_maze(30, 30, wall_density=0.2, seed=3)
    result      = _run_recommended(problem, rec, timeout=10.0)

    summary = (
        f"Scenario 3 — Memory Constrained\n"
        f"  Recommended  : {rec.primary}\n"
        f"  Solution     : {'FOUND' if result.found else 'NOT FOUND'}\n"
        f"  Peak memory  : {result.stats.peak_memory_kb:.1f} KB\n"
        f"  Memory limit : 80 KB\n"
        f"  Within limit : {'YES ✓' if result.stats.peak_memory_kb <= 80 else 'NO ✗'}\n"
        f"  Nodes        : {result.stats.nodes_expanded}\n"
        f"  Path cost    : {result.stats.path_cost}\n"
        f"  Note         : Memory-heavy algos (BFS, UCS) were penalised in scoring"
    )
    if verbose:
        print(summary)

    return ScenarioResult(
        scenario_name  = "Memory Constrained",
        recommendation = rec,
        search_result  = result,
        summary        = summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — Unknown / New Problem Type
# ─────────────────────────────────────────────────────────────────────────────

def scenario_unknown_problem(
    profiles: Optional[ProfileStore] = None,
    verbose:  bool = True,
) -> ScenarioResult:
    """
    Problem   : A graph with unusual parameters (size=75, no prior profile)
    Constraint: no explicit constraints
    Expected  : A* (safe, general-purpose default)

    This scenario simulates what happens when the agent has never seen
    this exact problem type before. It should fall back gracefully to A*.
    """
    # Use a fresh ProfileStore with intentionally missing size bucket
    empty_profiles = ProfileStore.default()
    selector       = StrategySelector(empty_profiles)

    rec = selector.recommend(
        env_type            = "graph",
        problem_size        = 75,   # between "small" (50) and "medium" (100)
        optimality_required = False,
    )

    g, start, goal = generate_random_graph(75, seed=99)
    problem         = GraphProblem(g, start, goal)
    result          = _run_recommended(problem, rec, timeout=10.0)

    summary = (
        f"Scenario 4 — Unknown Problem Type\n"
        f"  Recommended : {rec.primary}\n"
        f"  Solution    : {'FOUND' if result.found else 'NOT FOUND'}\n"
        f"  Runtime     : {result.stats.runtime_ms:.2f} ms\n"
        f"  Path cost   : {result.stats.path_cost}\n"
        f"  Nodes       : {result.stats.nodes_expanded}\n"
        f"  Note        : No exact profile match — agent used default profiles\n"
        f"                A* chosen as the safest general-purpose algorithm"
    )
    if verbose:
        print(summary)

    return ScenarioResult(
        scenario_name  = "Unknown Problem Type",
        recommendation = rec,
        search_result  = result,
        summary        = summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run all four scenarios
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scenarios(
    profiles: Optional[ProfileStore] = None,
    verbose:  bool = True,
) -> list:
    """
    Run all four agent demo scenarios and return a list of ScenarioResult.
    """
    if verbose:
        print("╔══════════════════════════════════════════════╗")
        print("║   Agent Layer — Demo Scenarios               ║")
        print("╚══════════════════════════════════════════════╝\n")

    results = []
    for fn in [
        scenario_speed_priority,
        scenario_optimality_priority,
        scenario_memory_constrained,
        scenario_unknown_problem,
    ]:
        if verbose:
            print("─" * 50)
        results.append(fn(profiles=profiles, verbose=verbose))
        if verbose:
            print()

    return results
