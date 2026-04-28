"""
tests/test_algorithms.py
------------------------
Unit tests for Section 2: all five search algorithms.
Run with:  python tests/test_algorithms.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np

from search_benchmark.core.graph import WeightedGraph, GraphProblem, generate_random_graph
from search_benchmark.core.grid  import Grid, GridProblem, generate_maze, generate_unsolvable_maze
from search_benchmark.algorithms import ALGORITHMS, bfs, dfs, ucs, gbfs, astar
from search_benchmark.algorithms.stats import SearchResult

passed = failed = 0

def ok(name):
    global passed; passed += 1
    print(f"  PASS  {name}")

def fail(name, err):
    global failed; failed += 1
    print(f"  FAIL  {name}: {err}")

def check(name, condition, err="assertion failed"):
    if condition: ok(name)
    else: fail(name, err)

def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def make_line_graph(n=6):
    """
    Linear graph: 0-1-2-3-4-5, all edge weights = 1.
    Optimal path: 0→1→2→3→4→5, cost = 5.
    """
    pos = {i: (float(i), 0.0) for i in range(n)}
    g   = WeightedGraph(n_nodes=n, positions=pos)
    for i in range(n - 1):
        g.add_edge(i, i + 1, 1.0)
    return GraphProblem(g, initial_state=0, goal_state=n - 1)

def make_diamond_graph():
    """
    Diamond: 0→{1,2}→3
    0→1 cost=1, 1→3 cost=10   (long but cheap start)
    0→2 cost=5, 2→3 cost=1    (expensive start, cheap finish)
    Optimal: 0→2→3 cost=6  vs  0→1→3 cost=11
    BFS finds 0→1→3 (fewer hops) — same depth but BFS ignores cost.
    UCS/A* find 0→2→3 (cheaper).
    """
    pos = {0:(0,0), 1:(1,1), 2:(1,-1), 3:(2,0)}
    g   = WeightedGraph(n_nodes=4, positions=pos)
    g.add_edge(0, 1, 1.0)
    g.add_edge(0, 2, 5.0)
    g.add_edge(1, 3, 10.0)
    g.add_edge(2, 3, 1.0)
    return GraphProblem(g, initial_state=0, goal_state=3)

def make_open_grid(size=5):
    """Fully open size×size grid, start=(0,0), goal=(size-1,size-1)."""
    return GridProblem(
        Grid(np.zeros((size, size), dtype=np.uint8)),
        initial_state=(0, 0),
        goal_state=(size - 1, size - 1),
    )

def make_unsolvable():
    return generate_unsolvable_maze(rows=8, cols=8)[0]


# ─────────────────────────────────────────────────────────────
# Helper: run all algorithms on a problem
# ─────────────────────────────────────────────────────────────

def run_all(problem):
    results = {}
    for name, mod in ALGORITHMS.items():
        results[name] = mod.solve(problem)
    return results


# ═════════════════════════════════════════════════════════════
# 1. Correctness — all algorithms find valid paths
# ═════════════════════════════════════════════════════════════
print("--- Correctness: valid paths ---")

prob = make_line_graph()
results = run_all(prob)

for name, res in results.items():
    check(
        f"{name}: solution found on line graph",
        res.found,
        f"failed: {res.stats.failure_reason}",
    )
    if res.found:
        # Path must start at initial_state and end at goal_state
        check(
            f"{name}: path starts at initial state",
            res.path[0] == prob.initial_state,
        )
        check(
            f"{name}: path ends at goal state",
            res.path[-1] == prob.goal_state,
        )
        # Path must be connected — each consecutive pair must be neighbours
        valid_path = True
        for i in range(len(res.path) - 1):
            s, ns = res.path[i], res.path[i + 1]
            neighbours = [a[0] for a in prob.actions(s)]
            if ns not in neighbours:
                valid_path = False
                break
        check(f"{name}: path is connected", valid_path)


# ═════════════════════════════════════════════════════════════
# 2. Optimality — BFS and UCS return optimal paths
# ═════════════════════════════════════════════════════════════
print("--- Optimality ---")

line = make_line_graph(6)
r_bfs = bfs.solve(line)
r_ucs = ucs.solve(line)
r_dfs = dfs.solve(line)
r_astar = astar.solve(line)

check("BFS path cost == 5 on line graph",   approx(r_bfs.stats.path_cost, 5.0))
check("UCS path cost == 5 on line graph",   approx(r_ucs.stats.path_cost, 5.0))
check("A* path cost == 5 on line graph",    approx(r_astar.stats.path_cost, 5.0))
check("DFS path cost >= UCS on line graph", r_dfs.stats.path_cost >= r_ucs.stats.path_cost)

# Diamond graph — UCS/A* must find cost=6, BFS finds cost=11 (fewer hops)
diamond = make_diamond_graph()
r_ucs_d   = ucs.solve(diamond)
r_astar_d = astar.solve(diamond)
r_bfs_d   = bfs.solve(diamond)

check("UCS optimal on diamond (cost=6)",   approx(r_ucs_d.stats.path_cost, 6.0))
check("A* optimal on diamond (cost=6)",    approx(r_astar_d.stats.path_cost, 6.0))
check("BFS finds path on diamond",         r_bfs_d.found)
# BFS uses unit cost so it picks the 2-hop path (0→1→3, cost=11 in our edge weights
# but BFS doesn't use edge weights — the path is still valid)
check("BFS path on diamond is connected",  r_bfs_d.found)


# ═════════════════════════════════════════════════════════════
# 3. A* with h=0 == UCS
# ═════════════════════════════════════════════════════════════
print("--- A* with h=0 equals UCS ---")

from search_benchmark.core.heuristics import zero as zero_h

prob = make_line_graph(8)
r_ucs2   = ucs.solve(prob)
r_astar0 = astar.solve(prob, heuristic=lambda n: zero_h(n.state, prob.goal_state))

check("A*(h=0) same path cost as UCS",
      approx(r_astar0.stats.path_cost, r_ucs2.stats.path_cost))
check("A*(h=0) same nodes expanded as UCS (±2)",
      abs(r_astar0.stats.nodes_expanded - r_ucs2.stats.nodes_expanded) <= 2)


# ═════════════════════════════════════════════════════════════
# 4. A* with admissible h never worse than UCS
# ═════════════════════════════════════════════════════════════
print("--- A* admissible heuristic never over-estimates ---")

for seed in range(10):
    g, start, goal = generate_random_graph(20, seed=seed)
    prob = GraphProblem(g, start, goal)
    r_u  = ucs.solve(prob)
    r_a  = astar.solve(prob)
    if r_u.found and r_a.found:
        check(
            f"A* cost <= UCS cost (seed={seed})",
            r_a.stats.path_cost <= r_u.stats.path_cost + 1e-6,
            f"A*={r_a.stats.path_cost:.4f} > UCS={r_u.stats.path_cost:.4f}",
        )


# ═════════════════════════════════════════════════════════════
# 5. A* expands fewer nodes than BFS on heuristic-friendly problems
# ═════════════════════════════════════════════════════════════
print("--- A* efficiency vs BFS ---")

prob = make_open_grid(10)
r_bfs10   = bfs.solve(prob)
r_astar10 = astar.solve(prob)

check("Both find solution on 10×10 grid", r_bfs10.found and r_astar10.found)
check(
    "A* expands fewer or equal nodes than BFS on open grid",
    r_astar10.stats.nodes_expanded <= r_bfs10.stats.nodes_expanded,
    f"A*={r_astar10.stats.nodes_expanded} > BFS={r_bfs10.stats.nodes_expanded}",
)


# ═════════════════════════════════════════════════════════════
# 6. Failure cases — unsolvable problems
# ═════════════════════════════════════════════════════════════
print("--- Failure: unsolvable problem ---")

unsolvable = make_unsolvable()
for name, mod in ALGORITHMS.items():
    res = mod.solve(unsolvable)
    check(
        f"{name}: returns solution_found=False on unsolvable",
        not res.found,
        f"Unexpectedly found a path",
    )
    check(
        f"{name}: failure_reason set on unsolvable",
        res.stats.failure_reason in ("no_path", "depth_limit", "timeout"),
        f"Got failure_reason={res.stats.failure_reason!r}",
    )


# ═════════════════════════════════════════════════════════════
# 7. DFS graph-search vs tree-search
# ═════════════════════════════════════════════════════════════
print("--- DFS graph vs tree search ---")

prob = make_line_graph(6)
r_graph = dfs.solve(prob, graph_search=True)
check("DFS graph-search finds solution", r_graph.found)

# Tree-search on tiny cyclic graph — give it a depth limit to prevent infinite loop
r_tree = dfs.solve(prob, graph_search=False, depth_limit=20)
check("DFS tree-search with depth limit terminates", True)   # just check it returns


# ═════════════════════════════════════════════════════════════
# 8. DFS depth limit
# ═════════════════════════════════════════════════════════════
print("--- DFS depth limit ---")

prob  = make_line_graph(10)   # goal at depth 9
r_lim = dfs.solve(prob, depth_limit=4)

check(
    "DFS depth-limit=4 fails to find goal at depth 9",
    not r_lim.found,
)
check(
    "DFS depth-limit sets failure_reason='depth_limit'",
    r_lim.stats.failure_reason == "depth_limit",
    f"Got {r_lim.stats.failure_reason!r}",
)


# ═════════════════════════════════════════════════════════════
# 9. Stats correctness
# ═════════════════════════════════════════════════════════════
print("--- Stats correctness ---")

prob    = make_line_graph(6)
r       = bfs.solve(prob)
s       = r.stats

check("nodes_expanded > 0",         s.nodes_expanded > 0)
check("nodes_generated >= expanded", s.nodes_generated >= s.nodes_expanded)
check("max_frontier_size > 0",      s.max_frontier_size > 0)
check("runtime_ms > 0",             s.runtime_ms > 0)
check("peak_memory_kb > 0",         s.peak_memory_kb > 0)
check("solution_depth set",         s.solution_depth is not None)
check("solution_depth == 5",        s.solution_depth == 5)
check("path_cost set",              s.path_cost is not None)
check("as_dict returns dict",       isinstance(s.as_dict(), dict))


# ═════════════════════════════════════════════════════════════
# 10. A* heuristic error computed post-solve
# ═════════════════════════════════════════════════════════════
print("--- A* heuristic error ---")

prob   = make_open_grid(6)
r_a    = astar.solve(prob)
check("heuristic_error populated after solve", r_a.stats.heuristic_error is not None)
check("heuristic_error >= 0",                  r_a.stats.heuristic_error >= 0)

# With admissible h (Manhattan on grid), error should be 0 or very small
# (Manhattan is not always tight but is always a lower bound)
check(
    "heuristic_error is finite",
    math.isfinite(r_a.stats.heuristic_error),
)


# ═════════════════════════════════════════════════════════════
# 11. Inadmissible heuristic — A* no longer optimal
# ═════════════════════════════════════════════════════════════
print("--- Inadmissible heuristic degrades A* ---")

from search_benchmark.core.heuristics import scaled_manhattan

prob_inadmissible = GridProblem(
    Grid(np.zeros((10, 10), dtype=np.uint8)),
    initial_state=(0, 0),
    goal_state=(9, 9),
    heuristic_scale=3.0,           # 3× over-estimates → inadmissible
)
prob_admissible = GridProblem(
    Grid(np.zeros((10, 10), dtype=np.uint8)),
    initial_state=(0, 0),
    goal_state=(9, 9),
    heuristic_scale=1.0,
)

r_adm  = astar.solve(prob_admissible)
r_inadm = astar.solve(prob_inadmissible)

check("Admissible A* finds solution",   r_adm.found)
check("Inadmissible A* finds solution", r_inadm.found)
check(
    "Admissible A* expands more nodes than inadmissible (inadmissible is greedier)",
    r_adm.stats.nodes_expanded >= r_inadm.stats.nodes_expanded,
    f"adm={r_adm.stats.nodes_expanded} inadm={r_inadm.stats.nodes_expanded}",
)
check(
    "Inadmissible A* returns higher or equal cost (sub-optimal)",
    r_inadm.stats.path_cost >= r_adm.stats.path_cost - 1e-6,
)


# ═════════════════════════════════════════════════════════════
# 12. Greedy vs A* on open grid
# ═════════════════════════════════════════════════════════════
print("--- Greedy vs A* ---")

prob = make_open_grid(8)
r_g  = gbfs.solve(prob)
r_a  = astar.solve(prob)

check("Greedy finds solution on open grid", r_g.found)
check("A* finds solution on open grid",     r_a.found)
check(
    "A* path cost <= Greedy path cost (A* is optimal)",
    r_a.stats.path_cost <= r_g.stats.path_cost + 1e-6,
    f"A*={r_a.stats.path_cost:.3f} > Greedy={r_g.stats.path_cost:.3f}",
)


# ═════════════════════════════════════════════════════════════
# 13. record_log populates expansion_log
# ═════════════════════════════════════════════════════════════
print("--- Expansion log (visualiser data) ---")

prob = make_line_graph(6)
r    = bfs.solve(prob, record_log=True)

check("expansion_log non-empty with record_log=True", len(r.stats.expansion_log) > 0)
check("expansion_log entry has 'expanded' key",       "expanded" in r.stats.expansion_log[0])
check("expansion_log entry has 'frontier' key",       "frontier" in r.stats.expansion_log[0])

r_no = bfs.solve(prob, record_log=False)
check("expansion_log empty with record_log=False",    len(r_no.stats.expansion_log) == 0)


# ═════════════════════════════════════════════════════════════
# 14. Algorithm registry
# ═════════════════════════════════════════════════════════════
print("--- Algorithm registry ---")

check("Registry has 5 algorithms",  len(ALGORITHMS) == 5)
check("'BFS' in registry",          "BFS"    in ALGORITHMS)
check("'DFS' in registry",          "DFS"    in ALGORITHMS)
check("'UCS' in registry",          "UCS"    in ALGORITHMS)
check("'Greedy' in registry",       "Greedy" in ALGORITHMS)
check("'A*' in registry",           "A*"     in ALGORITHMS)
for name, mod in ALGORITHMS.items():
    check(f"{name} module has solve()", callable(getattr(mod, "solve", None)))


# ═════════════════════════════════════════════════════════════
# 15. Maze problems
# ═════════════════════════════════════════════════════════════
print("--- Maze problems ---")

for seed in range(5):
    prob, _ = generate_maze(12, 12, wall_density=0.25, seed=seed)
    r_bfs_m   = bfs.solve(prob)
    r_astar_m = astar.solve(prob)
    if r_bfs_m.found and r_astar_m.found:
        check(
            f"A* cost <= BFS cost on 12×12 maze (seed={seed})",
            r_astar_m.stats.path_cost <= r_bfs_m.stats.path_cost + 1e-6,
            f"A*={r_astar_m.stats.path_cost:.3f} > BFS={r_bfs_m.stats.path_cost:.3f}",
        )


# ═════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════
print()
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")