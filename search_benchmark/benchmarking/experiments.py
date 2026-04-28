"""
benchmarking/experiments.py
----------------------------
The four structured experiments defined in the plan.

Experiment A — Graph Scaling
    20 / 50 / 100 / 200 nodes × 5 seeds = 20 problems
    All five algorithms, Euclidean heuristic.

Experiment B — Grid / Maze Scaling
    10×10 / 20×20 / 30×30 / 50×50 × wall_density ∈ {0.1, 0.3, 0.5}
    = 12 problems.
    BFS/DFS/UCS run without heuristic.
    A* and Greedy run with Manhattan, Euclidean, Chebyshev (separate rows).

Experiment C — Heuristic Stress Test
    10×10 grid, admissible + inadmissible heuristic scales.
    Shows A* degradation and Greedy path-quality collapse.

Experiment D — Failure Cases
    Unsolvable graph, unsolvable maze,
    DFS tree-search on a maze (expect timeout / huge expansion),
    BFS on a dense 50×50 maze (expect frontier blow-up).

Each experiment function returns a pd.DataFrame.
Call run_all_experiments() to run everything and get one combined DataFrame.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

try:
    from ..core.graph import GraphProblem, generate_random_graph
    from ..core.grid  import Grid, GridProblem, generate_maze, generate_unsolvable_maze
    from ..core.heuristics import scaled_manhattan
    from .runner import BenchmarkRunner
except ImportError:
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.core.graph import GraphProblem, generate_random_graph
    from search_benchmark.core.grid  import Grid, GridProblem, generate_maze, generate_unsolvable_maze
    from search_benchmark.core.heuristics import scaled_manhattan
    from search_benchmark.benchmarking.runner import BenchmarkRunner


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A — Graph Scaling
# ─────────────────────────────────────────────────────────────────────────────

def experiment_a_graph_scaling(
    runner:  Optional[BenchmarkRunner] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all algorithms on weighted graphs of increasing size.

    Sizes   : 20, 50, 100, 200 nodes
    Seeds   : 0 – 4  (5 per size)
    Heuristic: Euclidean (admissible for straight-line graphs)
    """
    runner = runner or BenchmarkRunner(timeout_seconds=15.0)

    sizes   = [20, 50, 100, 200]
    n_seeds = 5
    problems = []

    for size in sizes:
        for seed in range(n_seeds):
            g, start, goal = generate_random_graph(
                n_nodes      = size,
                edge_density = 0.2,
                weight_range = (1.0, 10.0),
                seed         = seed,
            )
            prob = GraphProblem(g, initial_state=start, goal_state=goal)
            problems.append({
                "problem":    prob,
                "problem_id": f"graph_n{size}_s{seed}",
                "meta": {
                    "env_type":  "graph",
                    "size":      size,
                    "seed":      seed,
                    "heuristic_name": "euclidean",
                },
            })

    if verbose:
        print(f"\n=== Experiment A: Graph Scaling ({len(problems)} problems) ===")

    df = runner.run_suite(
        problems,
        label   = "exp_a_graph",
        verbose = verbose,
    )
    df["experiment"] = "A"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B — Grid / Maze Scaling
# ─────────────────────────────────────────────────────────────────────────────

def experiment_b_grid_scaling(
    runner:  Optional[BenchmarkRunner] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run algorithms on grids of increasing size and wall density.

    Grid sizes    : 10×10, 20×20, 30×30, 50×50
    Wall densities: 0.1, 0.3, 0.5
    Seeds         : 0 – 2 per (size × density)
    Heuristics    : manhattan (default), euclidean, chebyshev for A*/Greedy
    """
    runner = runner or BenchmarkRunner(timeout_seconds=20.0)

    sizes    = [10, 20, 30, 50]
    densities = [0.1, 0.3, 0.5]
    n_seeds  = 3
    heuristic_names = ["manhattan", "euclidean", "chebyshev"]

    all_dfs = []

    for size in sizes:
        for density in densities:
            for seed in range(n_seeds):
                for h_name in heuristic_names:
                    prob, _ = generate_maze(
                        rows         = size,
                        cols         = size,
                        wall_density = density,
                        seed         = seed,
                        heuristic    = h_name,
                    )

                    all_dfs.append({
                        "problem":    prob,
                        "problem_id": f"grid_{size}x{size}_d{int(density*10)}_s{seed}_h{h_name[:3]}",
                        "meta": {
                            "env_type":       "grid",
                            "size":           size,
                            "wall_density":   density,
                            "seed":           seed,
                            "heuristic_name": h_name,
                        },
                    })

    if verbose:
        print(f"\n=== Experiment B: Grid Scaling ({len(all_dfs)} problems) ===")

    df = runner.run_suite(
        all_dfs,
        label   = "exp_b_grid",
        verbose = verbose,
    )
    df["experiment"] = "B"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C — Heuristic Stress Test
# ─────────────────────────────────────────────────────────────────────────────

def experiment_c_heuristic_stress(
    runner:  Optional[BenchmarkRunner] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test A* and Greedy Best-First with admissible and inadmissible heuristics.

    Scale factors : 1.0 (admissible), 1.5×, 2.0×, 3.0×  (inadmissible)
    Grid          : 15×15, wall_density=0.25, 5 seeds
    """
    runner = runner or BenchmarkRunner(
        timeout_seconds = 15.0,
        algorithms      = ["UCS", "Greedy", "A*"],  # UCS = ground truth
    )

    scales  = [1.0, 1.5, 2.0, 3.0]
    n_seeds = 5
    size    = 15

    problems = []

    for seed in range(n_seeds):
        for scale in scales:
            prob, _ = generate_maze(
                rows         = size,
                cols         = size,
                wall_density = 0.25,
                seed         = seed,
                heuristic    = "manhattan",
            )
            # Override heuristic_scale on the problem
            prob.heuristic_scale = scale

            problems.append({
                "problem":    prob,
                "problem_id": f"stress_s{seed}_x{scale}",
                "meta": {
                    "env_type":        "grid",
                    "size":            size,
                    "seed":            seed,
                    "heuristic_scale": scale,
                    "heuristic_name":  f"manhattan_x{scale}",
                    "admissible":      scale <= 1.0,
                },
            })

    if verbose:
        print(f"\n=== Experiment C: Heuristic Stress ({len(problems)} problems) ===")

    df = runner.run_suite(
        problems,
        label   = "exp_c_stress",
        verbose = verbose,
    )
    df["experiment"] = "C"

    # Add optimality_gap vs UCS
    if "path_cost" in df.columns:
        ucs_costs = (
            df[df["algo"] == "UCS"]
            .set_index("problem_id")["path_cost"]
            .to_dict()
        )
        df["ucs_optimal_cost"] = df["problem_id"].map(ucs_costs)
        df["optimality_gap"] = (
            (df["path_cost"] - df["ucs_optimal_cost"])
            .clip(lower=0)
            .round(4)
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Experiment D — Failure Cases
# ─────────────────────────────────────────────────────────────────────────────

def experiment_d_failure_cases(
    runner:  Optional[BenchmarkRunner] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Deliberately trigger failure modes in each algorithm.

    Case 1 — Unsolvable graph (all algos → no_path)
    Case 2 — Unsolvable maze  (all algos → no_path)
    Case 3 — DFS tree-search on a maze (expect timeout / huge expansion)
    Case 4 — BFS on a dense 50×50 maze (frontier blow-up)
    Case 5 — A* with 3× inadmissible heuristic on 20×20 (sub-optimal path)
    """
    all_rows = []

    # ── Case 1: Unsolvable graph ──────────────────────────────
    try:
        from ..core.graph import WeightedGraph
    except ImportError:
        from search_benchmark.core.graph import WeightedGraph
    g = WeightedGraph(n_nodes=10, positions={i: (float(i), 0.0) for i in range(10)})
    for i in range(4):  g.add_edge(i, i+1, 1.0)
    for i in range(5, 9): g.add_edge(i, i+1, 1.0)
    # Node 4 and Node 5 are NOT connected → start=0, goal=9 unreachable
    prob_bad_graph = GraphProblem(g, initial_state=0, goal_state=9)

    runner_all = runner or BenchmarkRunner(timeout_seconds=5.0)

    if verbose: print("\n=== Experiment D: Failure Cases ===")
    if verbose: print("  Case 1: Unsolvable graph")
    results_c1 = runner_all.run_all(prob_bad_graph)
    for algo, res in results_c1.items():
        all_rows.append({
            "experiment":  "D",
            "case":        "D1_unsolvable_graph",
            "algo":        algo,
            "env_type":    "graph",
            **res.stats.as_dict(),
        })

    # ── Case 2: Unsolvable maze ───────────────────────────────
    if verbose: print("  Case 2: Unsolvable maze")
    prob_bad_maze, _ = generate_unsolvable_maze(rows=12, cols=12)
    results_c2 = runner_all.run_all(prob_bad_maze)
    for algo, res in results_c2.items():
        all_rows.append({
            "experiment":  "D",
            "case":        "D2_unsolvable_maze",
            "algo":        algo,
            "env_type":    "grid",
            **res.stats.as_dict(),
        })

    # ── Case 3: DFS tree-search on maze ──────────────────────
    # Use a very short timeout to capture the blow-up without hanging
    if verbose: print("  Case 3: DFS tree-search on maze (short timeout)")
    prob_dfs_demo, _ = generate_maze(rows=12, cols=12, wall_density=0.2, seed=7)

    try:
        from ..algorithms import dfs as dfs_module
        from ..algorithms.stats import SearchStats, SearchResult
    except ImportError:
        from search_benchmark.algorithms import dfs as dfs_module
        from search_benchmark.algorithms.stats import SearchStats, SearchResult

    runner_short = BenchmarkRunner(timeout_seconds=2.0, algorithms=["DFS"])

    # Graph-search (safe)
    res_dfs_graph = runner_short.run_single(prob_dfs_demo, "DFS")
    all_rows.append({
        "experiment":  "D",
        "case":        "D3a_dfs_graph_search",
        "algo":        "DFS_graph",
        "env_type":    "grid",
        **res_dfs_graph.stats.as_dict(),
    })

    # Tree-search via direct call (may hit timeout)
    import time, tracemalloc
    stats_tree = SearchStats()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                dfs_module.solve,
                prob_dfs_demo,
                None, 2.0, False,
                False,   # graph_search=False  ← tree search!
                None,
            )
            res_tree = future.result(timeout=2.0)
    except concurrent.futures.TimeoutError:
        res_tree = SearchResult(
            stats=SearchStats(
                solution_found=False,
                failure_reason="timeout",
                runtime_ms=2000.0,
            ),
            algo_name="DFS_tree",
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    all_rows.append({
        "experiment":  "D",
        "case":        "D3b_dfs_tree_search",
        "algo":        "DFS_tree",
        "env_type":    "grid",
        **res_tree.stats.as_dict(),
    })

    # ── Case 4: BFS on dense 30×30 maze ──────────────────────
    if verbose: print("  Case 4: BFS on 30×30 maze (frontier blow-up vs A*)")
    prob_big, _ = generate_maze(rows=30, cols=30, wall_density=0.1, seed=0)
    runner_big  = BenchmarkRunner(timeout_seconds=10.0, algorithms=["BFS", "A*"])
    results_c4  = runner_big.run_all(prob_big)
    for algo, res in results_c4.items():
        all_rows.append({
            "experiment":  "D",
            "case":        "D4_bfs_dense_maze",
            "algo":        algo,
            "env_type":    "grid",
            **res.stats.as_dict(),
        })

    # ── Case 5: A* with inadmissible heuristic ───────────────
    if verbose: print("  Case 5: A* inadmissible heuristic (sub-optimal path)")
    prob_inadm = GridProblem(
        Grid(np.zeros((20, 20), dtype=np.uint8)),
        initial_state=(0, 0),
        goal_state=(19, 19),
        heuristic_scale=3.0,
    )
    prob_admiss = GridProblem(
        Grid(np.zeros((20, 20), dtype=np.uint8)),
        initial_state=(0, 0),
        goal_state=(19, 19),
        heuristic_scale=1.0,
    )
    runner_c5 = BenchmarkRunner(timeout_seconds=10.0, algorithms=["A*", "UCS"])
    for label_inner, prob_inner in [("admissible", prob_admiss), ("inadmissible_3x", prob_inadm)]:
        res_dict = runner_c5.run_all(prob_inner)
        for algo, res in res_dict.items():
            all_rows.append({
                "experiment":  "D",
                "case":        f"D5_astar_{label_inner}",
                "algo":        algo,
                "env_type":    "grid",
                **res.stats.as_dict(),
            })

    df = pd.DataFrame(all_rows)

    # Save
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/exp_d_failure.csv", index=False)
    if verbose:
        print(f"\n  Saved → results/exp_d_failure.csv  ({len(df)} rows)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(
    verbose: bool = True,
    fast_mode: bool = False,
) -> pd.DataFrame:
    """
    Run all four experiments and return one combined DataFrame.

    Parameters
    ----------
    fast_mode : If True, use smaller problem sizes (for quick testing).

    Returns
    -------
    pd.DataFrame saved to results/all_experiments.csv
    """
    if verbose:
        print("╔══════════════════════════════════════════════╗")
        print("║   Search Algorithm Benchmarking Suite        ║")
        print("╚══════════════════════════════════════════════╝")

    dfs_list = []

    dfs_list.append(experiment_a_graph_scaling(verbose=verbose))
    dfs_list.append(experiment_b_grid_scaling(verbose=verbose))
    dfs_list.append(experiment_c_heuristic_stress(verbose=verbose))
    dfs_list.append(experiment_d_failure_cases(verbose=verbose))

    combined = pd.concat(dfs_list, ignore_index=True)
    combined.to_csv("results/all_experiments.csv", index=False)

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  Total rows  : {len(combined)}")
        print(f"  Experiments : A, B, C, D")
        print(f"  Saved       → results/all_experiments.csv")

    return combined
