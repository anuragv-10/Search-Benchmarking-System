"""
tests/test_benchmarking.py
--------------------------
Unit tests for Section 3: BenchmarkRunner and experiments.
Run with:  python tests/test_benchmarking.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from search_benchmark.core.graph import GraphProblem, generate_random_graph
from search_benchmark.core.grid  import Grid, GridProblem, generate_maze, generate_unsolvable_maze
from search_benchmark.benchmarking.runner import BenchmarkRunner
from search_benchmark.benchmarking.experiments import (
    experiment_a_graph_scaling,
    experiment_b_grid_scaling,
    experiment_c_heuristic_stress,
    experiment_d_failure_cases,
)

passed = failed = 0

def ok(name):
    global passed; passed += 1
    print(f"  PASS  {name}")

def fail(name, err=""):
    global failed; failed += 1
    print(f"  FAIL  {name}: {err}")

def check(name, condition, err="assertion failed"):
    if condition: ok(name)
    else: fail(name, err)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def small_graph_problem():
    g, s, e = generate_random_graph(15, seed=1)
    return GraphProblem(g, s, e)

def small_grid_problem():
    p, _ = generate_maze(8, 8, seed=1)
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BenchmarkRunner basics
# ═══════════════════════════════════════════════════════════════════════════════
print("--- BenchmarkRunner: run_single ---")

runner = BenchmarkRunner(timeout_seconds=5.0)
prob   = small_graph_problem()

for algo in ["BFS", "DFS", "UCS", "Greedy", "A*"]:
    res = runner.run_single(prob, algo)
    check(f"run_single({algo}) returns SearchResult", hasattr(res, "stats"))
    check(f"run_single({algo}) stats has runtime_ms", res.stats.runtime_ms > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. run_all returns dict with all 5 algos
# ═══════════════════════════════════════════════════════════════════════════════
print("--- BenchmarkRunner: run_all ---")

results = runner.run_all(prob)
check("run_all returns dict",          isinstance(results, dict))
check("run_all has 5 keys",            len(results) == 5)
check("run_all keys are algo names",   set(results.keys()) == {"BFS","DFS","UCS","Greedy","A*"})
for name, res in results.items():
    check(f"run_all[{name}] has stats", hasattr(res.stats, "nodes_expanded"))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. run_suite returns correct DataFrame shape
# ═══════════════════════════════════════════════════════════════════════════════
print("--- BenchmarkRunner: run_suite DataFrame shape ---")

probs2 = [
    {"problem": small_graph_problem(), "problem_id": "p1", "meta": {"env_type": "graph", "size": 15}},
    {"problem": small_grid_problem(),  "problem_id": "p2", "meta": {"env_type": "grid",  "size": 8}},
]
df = runner.run_suite(probs2, label="test_suite", save_csv=False, verbose=False)

check("run_suite returns DataFrame",            isinstance(df, pd.DataFrame))
check("run_suite rows = 2 problems × 5 algos", len(df) == 10)
check("df has 'algo' column",                  "algo" in df.columns)
check("df has 'problem_id' column",            "problem_id" in df.columns)
check("df has 'nodes_expanded' column",        "nodes_expanded" in df.columns)
check("df has 'runtime_ms' column",            "runtime_ms" in df.columns)
check("df has 'solution_found' column",        "solution_found" in df.columns)
check("df has 'env_type' from meta",           "env_type" in df.columns)
check("df has 'size' from meta",               "size" in df.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Timeout fires correctly
# ═══════════════════════════════════════════════════════════════════════════════
print("--- Timeout enforcement ---")

# 100×100 open grid — definitely cannot solve in 1ms
runner_tiny = BenchmarkRunner(timeout_seconds=0.001)
prob_huge   = GridProblem(
    Grid(np.zeros((100, 100), dtype=np.uint8)), (0,0), (99,99)
)

t0 = time.perf_counter()
res_timeout = runner_tiny.run_single(prob_huge, "BFS")
elapsed = time.perf_counter() - t0

check(
    "Timeout fires within ~1 second",
    elapsed < 1.5,
    f"Took {elapsed:.2f}s",
)
check(
    "Timeout result has solution_found=False",
    not res_timeout.stats.solution_found,
)
check(
    "Timeout result has failure_reason='timeout'",
    res_timeout.stats.failure_reason == "timeout",
    f"Got {res_timeout.stats.failure_reason!r}",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. UCS path_cost ≤ DFS path_cost (across seeds)
# ═══════════════════════════════════════════════════════════════════════════════
print("--- UCS optimal vs DFS ---")

runner5 = BenchmarkRunner(timeout_seconds=5.0)
for seed in range(10):
    g, s, e = generate_random_graph(20, seed=seed)
    p = GraphProblem(g, s, e)
    r_ucs = runner5.run_single(p, "UCS")
    r_dfs = runner5.run_single(p, "DFS")
    if r_ucs.found and r_dfs.found:
        check(
            f"UCS cost ≤ DFS cost (seed={seed})",
            r_ucs.stats.path_cost <= r_dfs.stats.path_cost + 1e-6,
            f"UCS={r_ucs.stats.path_cost:.3f} > DFS={r_dfs.stats.path_cost:.3f}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CSV written and readable
# ═══════════════════════════════════════════════════════════════════════════════
print("--- CSV persistence ---")

import tempfile, shutil
tmpdir = tempfile.mkdtemp()
try:
    df_saved = runner.run_suite(probs2, label="csv_test", save_csv=True, csv_dir=tmpdir, verbose=False)
    csv_path = os.path.join(tmpdir, "csv_test.csv")
    check("CSV file created", os.path.exists(csv_path))
    df_read = pd.read_csv(csv_path)
    check("CSV readable",             isinstance(df_read, pd.DataFrame))
    check("CSV same row count",       len(df_read) == len(df_saved))
    check("CSV has correct columns",  "algo" in df_read.columns)
finally:
    shutil.rmtree(tmpdir)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. summarise() produces aggregated DataFrame
# ═══════════════════════════════════════════════════════════════════════════════
print("--- BenchmarkRunner.summarise ---")

probs3 = [
    {"problem": small_graph_problem(), "problem_id": f"pg{i}", "meta": {"env_type":"graph","size":15}}
    for i in range(3)
]
df3   = runner.run_suite(probs3, label="summ_test", save_csv=False, verbose=False)
summ  = BenchmarkRunner.summarise(df3, group_cols=["algo"])

check("summarise returns DataFrame",               isinstance(summ, pd.DataFrame))
check("summarise has one row per algo",            len(summ) == 5)
check("summarise has nodes_expanded_mean column",  "nodes_expanded_mean" in summ.columns)
check("summarise has runtime_ms_mean column",      "runtime_ms_mean" in summ.columns)
check("summarise has success_rate column",         "success_rate" in summ.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Experiment A — smoke test (small config)
# ═══════════════════════════════════════════════════════════════════════════════
print("--- Experiment A: smoke test ---")

runner_a = BenchmarkRunner(timeout_seconds=5.0)

# Patch experiment_a to run tiny config for test speed
from search_benchmark.benchmarking.experiments import experiment_a_graph_scaling
import unittest.mock as mock

# Build 2 small problems manually and test the runner
mini_problems = [
    {
        "problem":    GraphProblem(*generate_random_graph(10, seed=s)[::-1]),
        "problem_id": f"mini_{s}",
        "meta":       {"env_type":"graph","size":10,"seed":s,"heuristic_name":"euclidean"},
    }
    for s in range(2)
]
df_a = runner_a.run_suite(mini_problems, label="mini_a", save_csv=False, verbose=False)

check("Exp A DataFrame has rows",          len(df_a) > 0)
check("Exp A DataFrame has all 5 algos",   df_a["algo"].nunique() == 5)
check("Exp A DataFrame has env_type col",  "env_type" in df_a.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Experiment C — optimality gap computed
# ═══════════════════════════════════════════════════════════════════════════════
print("--- Experiment C: optimality gap ---")

# Manually construct the stress scenario for one seed
prob_adm = GridProblem(
    Grid(np.zeros((10, 10), dtype=np.uint8)),
    (0,0), (9,9), heuristic_scale=1.0
)
prob_inadm = GridProblem(
    Grid(np.zeros((10, 10), dtype=np.uint8)),
    (0,0), (9,9), heuristic_scale=3.0
)

runner_c = BenchmarkRunner(timeout_seconds=5.0, algorithms=["UCS","A*"])
r_adm   = runner_c.run_single(prob_adm,   "A*")
r_inadm = runner_c.run_single(prob_inadm, "A*")
r_ucs   = runner_c.run_single(prob_adm,   "UCS")

check("Admissible A* finds solution",        r_adm.found)
check("Inadmissible A* finds solution",      r_inadm.found)
check("UCS optimal cost <= A*(3x) cost",
      r_ucs.stats.path_cost <= r_inadm.stats.path_cost + 1e-6)
check("Inadmissible cost >= admissible cost",
      r_inadm.stats.path_cost >= r_adm.stats.path_cost - 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Experiment D — failure cases (lightweight direct tests)
# ═══════════════════════════════════════════════════════════════════════════════
print("--- Experiment D: failure cases (direct) ---")

runner_d = BenchmarkRunner(timeout_seconds=5.0)

# D1: unsolvable graph — all algos return no_path
from search_benchmark.core.graph import WeightedGraph
g_bad = WeightedGraph(n_nodes=10, positions={i:(float(i),0.0) for i in range(10)})
for i in range(4): g_bad.add_edge(i, i+1, 1.0)
for i in range(5, 9): g_bad.add_edge(i, i+1, 1.0)
prob_bad_graph = GraphProblem(g_bad, initial_state=0, goal_state=9)

for algo in ["BFS", "DFS", "UCS", "Greedy", "A*"]:
    res = runner_d.run_single(prob_bad_graph, algo)
    check(f"D1 {algo}: solution_found=False on unsolvable graph", not res.found)

# D2: unsolvable maze
prob_um, _ = generate_unsolvable_maze(10, 10)
for algo in ["BFS", "UCS", "A*"]:
    res = runner_d.run_single(prob_um, algo)
    check(f"D2 {algo}: solution_found=False on unsolvable maze", not res.found)

# D5: inadmissible heuristic cost degradation
prob_adm   = GridProblem(Grid(np.zeros((15,15),dtype=np.uint8)), (0,0),(14,14), heuristic_scale=1.0)
prob_inadm = GridProblem(Grid(np.zeros((15,15),dtype=np.uint8)), (0,0),(14,14), heuristic_scale=3.0)
r_adm   = runner_d.run_single(prob_adm,   "A*")
r_inadm = runner_d.run_single(prob_inadm, "A*")
r_ucs   = runner_d.run_single(prob_adm,   "UCS")
check("D5: admissible A* finds solution",       r_adm.found)
check("D5: inadmissible A* finds solution",     r_inadm.found)
check("D5: UCS cost <= inadmissible A* cost",   r_ucs.stats.path_cost <= r_inadm.stats.path_cost + 1e-6)
check("D5: inadmissible cost >= admissible",    r_inadm.stats.path_cost >= r_adm.stats.path_cost - 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Algorithm subset configuration
# ═══════════════════════════════════════════════════════════════════════════════
print("--- Algorithm subset ---")

runner_sub = BenchmarkRunner(algorithms=["BFS", "A*"])
prob_sub   = small_graph_problem()
results_sub = runner_sub.run_all(prob_sub)
check("Subset runner only runs configured algos", set(results_sub.keys()) == {"BFS", "A*"})
check("Subset runner has exactly 2 results",      len(results_sub) == 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. record_log flows through runner
# ═══════════════════════════════════════════════════════════════════════════════
print("--- record_log flows through runner ---")

runner_log = BenchmarkRunner(timeout_seconds=5.0, record_log=True)
prob_log   = small_graph_problem()
res_log    = runner_log.run_single(prob_log, "BFS")
check(
    "expansion_log populated when record_log=True on runner",
    len(res_log.stats.expansion_log) > 0,
)

runner_nolog = BenchmarkRunner(timeout_seconds=5.0, record_log=False)
res_nolog    = runner_nolog.run_single(prob_log, "BFS")
check(
    "expansion_log empty when record_log=False on runner",
    len(res_nolog.stats.expansion_log) == 0,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")