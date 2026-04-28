"""
tests/test_analysis.py
----------------------
Unit tests for Section 4: charts and heuristic analysis.
Run with:  python tests/test_analysis.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math, numpy as np

from search_benchmark.core.graph import GraphProblem, generate_random_graph
from search_benchmark.core.grid  import Grid, GridProblem, generate_maze
from search_benchmark.core.problem import Node
from search_benchmark.algorithms import astar, ucs, bfs
from search_benchmark.analysis.heuristics_analysis import (
    check_admissibility, check_consistency, compute_accuracy, full_heuristic_report,
)
from search_benchmark.analysis.charts import (
    chart_c1_nodes_expanded, chart_c2_runtime, chart_c3_memory,
    chart_c4_optimality, chart_c5_heuristic_error,
    chart_c6_frontier_growth, chart_c7_success_heatmap,
    generate_all_charts,
)

import pandas as pd
import tempfile, shutil

passed = failed = 0
def ok(n):   global passed; passed+=1; print(f"  PASS  {n}")
def fail(n,e=""): global failed; failed+=1; print(f"  FAIL  {n}: {e}")
def check(n, c, e="assertion failed"):
    if c: ok(n)
    else: fail(n, e)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def open_grid_problem(size=8):
    return GridProblem(
        Grid(np.zeros((size,size),dtype=np.uint8)),
        (0,0),(size-1,size-1),
        heuristic_name="manhattan",
    )

def line_graph_problem(n=8):
    g,s,e = generate_random_graph(n, seed=0)
    return GraphProblem(g,s,e)

def solve_and_get_nodes(prob):
    res = astar.solve(prob)
    if not res.found:
        return res, []
    # Rebuild Node list from path states
    nodes = []
    cost  = 0.0
    for i, state in enumerate(res.path):
        node = Node(state=state, depth=i, path_cost=cost)
        if i > 0:
            prev = res.path[i-1]
            for a in prob.actions(prev):
                ns = prob.result(prev, a)
                if ns == state:
                    cost += prob.step_cost(prev, a, state)
                    node  = Node(state=state, depth=i, path_cost=cost)
                    break
        nodes.append(node)
    return res, nodes


# ═════════════════════════════════════════════════════════════════════════════
# 1. Admissibility checker
# ═════════════════════════════════════════════════════════════════════════════
print("--- Admissibility checker ---")

prob = open_grid_problem(6)
res, nodes = solve_and_get_nodes(prob)
check("A* found solution for admissibility test", res.found)

if nodes:
    # Manhattan on grid is admissible
    rpt = check_admissibility(prob, nodes, prob.h, max_sample=10)
    check("Admissible h: violations == 0",      rpt.violations == 0)
    check("Admissible h: is_admissible == True", rpt.is_admissible)
    check("Admissible h: rate == 1.0",           rpt.admissibility_rate == 1.0)
    check("report has total_nodes > 0",          rpt.total_nodes > 0)
    check("as_dict returns dict",                isinstance(rpt.as_dict(), dict))

    # Inadmissible: 3× scaled heuristic
    prob_inadm = GridProblem(
        Grid(np.zeros((6,6),dtype=np.uint8)), (0,0),(5,5),
        heuristic_scale=3.0
    )
    rpt_inadm = check_admissibility(prob_inadm, nodes, prob_inadm.h, max_sample=10)
    check("Inadmissible h: violations > 0",       rpt_inadm.violations > 0)
    check("Inadmissible h: is_admissible == False", not rpt_inadm.is_admissible)
    check("Inadmissible h: rate < 1.0",            rpt_inadm.admissibility_rate < 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Consistency checker
# ═════════════════════════════════════════════════════════════════════════════
print("--- Consistency checker ---")

prob2 = open_grid_problem(6)
res2, nodes2 = solve_and_get_nodes(prob2)

if nodes2:
    # Manhattan on cardinal grid is consistent
    rpt_con = check_consistency(prob2, nodes2[:15], prob2.h)
    check("Manhattan consistent: violations == 0",  rpt_con.violations == 0)
    check("Manhattan consistent: rate == 1.0",      rpt_con.consistency_rate == 1.0)
    check("Manhattan consistent: is_consistent",    rpt_con.is_consistent)
    check("Consistency report total_edges > 0",     rpt_con.total_edges > 0)
    check("Consistency as_dict returns dict",       isinstance(rpt_con.as_dict(), dict))

    # 3× inadmissible → also inconsistent
    prob_inc = GridProblem(
        Grid(np.zeros((6,6),dtype=np.uint8)), (0,0),(5,5),
        heuristic_scale=3.0
    )
    rpt_inc = check_consistency(prob_inc, nodes2[:15], prob_inc.h)
    check("3× heuristic is inconsistent",           not rpt_inc.is_consistent)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Accuracy score
# ═════════════════════════════════════════════════════════════════════════════
print("--- Accuracy score ---")

prob3 = open_grid_problem(5)
res3, nodes3 = solve_and_get_nodes(prob3)

if nodes3:
    acc_admiss = compute_accuracy(prob3, nodes3, prob3.h, max_sample=10)
    check("Accuracy is in [0,1]",    0.0 <= acc_admiss <= 1.0)
    check("Admissible h accuracy > 0.0", acc_admiss > 0.0)

    # Perfect heuristic (h=0, no error) accuracy should be near 1 in trivial case
    acc_zero = compute_accuracy(prob3, nodes3[:1], lambda n: 0.0, max_sample=5)
    check("Accuracy returns float", isinstance(acc_zero, float))
    check("Accuracy is finite",     math.isfinite(acc_admiss))


# ═════════════════════════════════════════════════════════════════════════════
# 4. full_heuristic_report
# ═════════════════════════════════════════════════════════════════════════════
print("--- full_heuristic_report ---")

prob4 = open_grid_problem(6)
res4  = astar.solve(prob4)
rpt4  = full_heuristic_report(prob4, res4)

check("full_report returns dict",                    isinstance(rpt4, dict))
check("full_report has admissibility key",           "admissibility" in rpt4)
check("full_report has consistency key",             "consistency"   in rpt4)
check("full_report has accuracy key",                "accuracy"      in rpt4)
check("full_report admissibility is dict",           isinstance(rpt4["admissibility"], dict))
check("full_report accuracy in [0,1]",              0.0 <= rpt4["accuracy"] <= 1.0)

# No-solution case
prob_bad = GridProblem(
    Grid(np.array([[0,1],[1,0]],dtype=np.uint8)), (0,0),(1,1)
)
res_bad = astar.solve(prob_bad)
rpt_bad = full_heuristic_report(prob_bad, res_bad)
check("full_report handles no-solution gracefully", "note" in rpt_bad)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Charts render without error and save PNGs
# ═════════════════════════════════════════════════════════════════════════════
print("--- Charts render and save ---")

# Build a minimal DataFrame for chart testing
from search_benchmark.benchmarking.runner import BenchmarkRunner

runner = BenchmarkRunner(timeout_seconds=5.0)
probs_chart = [
    {
        "problem": open_grid_problem(s),
        "problem_id": f"grid_{s}",
        "meta": {
            "env_type": "grid", "size": s,
            "heuristic_name": "manhattan",
            "experiment": "A",
        }
    }
    for s in [6, 8, 10]
]
df_chart = runner.run_suite(probs_chart, label="chart_test", save_csv=False, verbose=False)
df_chart["experiment"] = "A"

tmpdir = tempfile.mkdtemp()
try:
    # C1
    path_c1 = chart_c1_nodes_expanded(df_chart, chart_dir=tmpdir)
    check("C1 PNG created",    os.path.isfile(path_c1))
    check("C1 is .png",        path_c1.endswith(".png"))

    # C2
    path_c2 = chart_c2_runtime(df_chart, chart_dir=tmpdir)
    check("C2 PNG created",    os.path.isfile(path_c2))

    # C3
    path_c3 = chart_c3_memory(df_chart, chart_dir=tmpdir)
    check("C3 PNG created",    os.path.isfile(path_c3))

    # C4
    path_c4 = chart_c4_optimality(df_chart, chart_dir=tmpdir)
    check("C4 PNG created",    os.path.isfile(path_c4))

    # C5 — needs heuristic_error & heuristic_name cols
    df_c5 = df_chart.copy()
    df_c5["heuristic_error"] = 0.1
    df_c5["heuristic_name"]  = "manhattan"
    path_c5 = chart_c5_heuristic_error(df_c5, chart_dir=tmpdir)
    check("C5 PNG created",    os.path.isfile(path_c5))

    # C6 — frontier growth from expansion_log
    prob_log = open_grid_problem(6)
    runner_log = BenchmarkRunner(timeout_seconds=5.0, record_log=True)
    logs = {}
    for algo in ["BFS", "A*"]:
        res_l = runner_log.run_single(prob_log, algo)
        if res_l.stats.expansion_log:
            logs[algo] = [len(e["frontier"]) for e in res_l.stats.expansion_log]
    path_c6 = chart_c6_frontier_growth(logs, chart_dir=tmpdir)
    check("C6 PNG created",    os.path.isfile(path_c6))
    check("C6 has log data",   len(logs) > 0)

    # C7
    path_c7 = chart_c7_success_heatmap(df_chart, chart_dir=tmpdir)
    check("C7 PNG created",    os.path.isfile(path_c7))

    # generate_all_charts convenience wrapper
    saved = generate_all_charts(df_c5, expansion_logs=logs, chart_dir=tmpdir, verbose=False)
    check("generate_all_charts returns dict",   isinstance(saved, dict))
    check("generate_all_charts has C1-C7",      all(k in saved for k in ["C1","C2","C3","C4","C5","C6","C7"]))
    check("All 7 chart files exist",
          all(os.path.isfile(p) for p in saved.values()),
          str({k:os.path.isfile(p) for k,p in saved.items()}))

finally:
    shutil.rmtree(tmpdir)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Chart handles missing columns gracefully
# ═════════════════════════════════════════════════════════════════════════════
print("--- Charts handle missing data gracefully ---")

tmpdir2 = tempfile.mkdtemp()
try:
    df_empty = pd.DataFrame({"algo": ["BFS"], "solution_found": [True]})
    path = chart_c1_nodes_expanded(df_empty, chart_dir=tmpdir2)
    check("C1 with no 'size' col still saves PNG", os.path.isfile(path))
    path7 = chart_c7_success_heatmap(df_empty, chart_dir=tmpdir2)
    check("C7 with no 'experiment' col saves PNG", os.path.isfile(path7))
finally:
    shutil.rmtree(tmpdir2)


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print()
print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")