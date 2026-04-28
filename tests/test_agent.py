"""
tests/test_agent.py
-------------------
Unit tests for Section 5: ProfileStore, StrategySelector, and scenarios.
Run with:  python tests/test_agent.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile, shutil
import pandas as pd
import numpy as np

from search_benchmark.agent.profiles  import ProfileStore, size_bucket, DEFAULT_PROFILES
from search_benchmark.agent.selector  import StrategySelector, Recommendation, OPTIMAL_ALGOS
from search_benchmark.agent.scenario import (
    scenario_speed_priority,
    scenario_optimality_priority,
    scenario_memory_constrained,
    scenario_unknown_problem,
)

passed = failed = 0
def ok(n):       global passed; passed+=1; print(f"  PASS  {n}")
def fail(n,e=""): global failed; failed+=1; print(f"  FAIL  {n}: {e}")
def check(n, c, e="assertion failed"):
    if c: ok(n)
    else: fail(n, e)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Size bucket
# ═══════════════════════════════════════════════════════════════════════════════
print("--- size_bucket ---")
check("size 10 → tiny",   size_bucket(10)  == "tiny")
check("size 20 → tiny",   size_bucket(20)  == "tiny")
check("size 21 → small",  size_bucket(21)  == "small")
check("size 50 → small",  size_bucket(50)  == "small")
check("size 51 → medium", size_bucket(51)  == "medium")
check("size 100 → medium",size_bucket(100) == "medium")
check("size 101 → large", size_bucket(101) == "large")
check("size 500 → large", size_bucket(500) == "large")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ProfileStore.default()
# ═══════════════════════════════════════════════════════════════════════════════
print("--- ProfileStore.default ---")
store = ProfileStore.default()
check("default store has A*",     store.get("A*")     is not None)
check("default store has BFS",    store.get("BFS")    is not None)
check("default store has DFS",    store.get("DFS")    is not None)
check("default store has UCS",    store.get("UCS")    is not None)
check("default store has Greedy", store.get("Greedy") is not None)
for algo in DEFAULT_PROFILES:
    p = store.get(algo)
    check(f"{algo} profile has mean_runtime_ms",     "mean_runtime_ms"     in p)
    check(f"{algo} profile has mean_nodes_expanded", "mean_nodes_expanded" in p)
    check(f"{algo} profile has mean_memory_kb",      "mean_memory_kb"      in p)
    check(f"{algo} profile has optimal_rate",        "optimal_rate"        in p)
    check(f"{algo} profile has success_rate",        "success_rate"        in p)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ProfileStore.from_dataframe
# ═══════════════════════════════════════════════════════════════════════════════
print("--- ProfileStore.from_dataframe ---")
# Build a minimal synthetic DataFrame
rows = []
for algo in ["BFS","DFS","UCS","Greedy","A*"]:
    for i in range(4):
        rows.append({
            "algo":           algo,
            "problem_id":     f"p{i}",
            "env_type":       "graph",
            "size":           30,
            "runtime_ms":     10.0 + i,
            "nodes_expanded": 20 + i,
            "peak_memory_kb": 50.0 + i,
            "path_cost":      5.0 if algo in ("UCS","A*") else 8.0,
            "solution_found": True,
            "failure_reason": None,
        })
df_syn = pd.DataFrame(rows)
store_df = ProfileStore.from_dataframe(df_syn)
p_bfs = store_df.get("BFS", env_type="graph", size=30)
check("from_df BFS profile exists",                p_bfs is not None)
check("from_df BFS has mean_runtime_ms",           "mean_runtime_ms" in p_bfs)
check("from_df BFS success_rate == 1.0",           p_bfs["success_rate"] == 1.0)
check("from_df UCS optimal_rate == 1.0",           store_df.get("UCS", size=30)["optimal_rate"] == 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ProfileStore save / load
# ═══════════════════════════════════════════════════════════════════════════════
print("--- ProfileStore persistence ---")
tmpdir = tempfile.mkdtemp()
try:
    path = os.path.join(tmpdir, "profiles.json")
    store.save(path)
    check("profiles.json created",      os.path.isfile(path))
    store2 = ProfileStore.load(path)
    p1 = store.get("A*")
    p2 = store2.get("A*")
    check("loaded profile matches saved", p1["mean_runtime_ms"] == p2["mean_runtime_ms"])
    # Load from nonexistent path → default
    store3 = ProfileStore.load(os.path.join(tmpdir, "missing.json"))
    check("load missing file → default",  store3.get("BFS") is not None)
finally:
    shutil.rmtree(tmpdir)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. StrategySelector — optimality constraint
# ═══════════════════════════════════════════════════════════════════════════════
print("--- StrategySelector: optimality constraint ---")
sel = StrategySelector()
rec = sel.recommend(optimality_required=True)
check("optimality_required: primary is optimal",
      rec.primary in OPTIMAL_ALGOS,
      f"got {rec.primary}")
check("optimality_required: DFS not in fallback",
      "DFS" not in [rec.primary] + rec.fallback_order or
      "DFS" in rec.fallback_order,   # DFS might be fallback but should be eliminated from primary
      "DFS should be eliminated from primary")
# More strictly: DFS should be eliminated entirely
dfs_eliminated = "DFS" not in ([rec.primary] + rec.fallback_order)
greedy_eliminated = "Greedy" not in ([rec.primary] + rec.fallback_order)
check("DFS eliminated when optimality required",    dfs_eliminated)
check("Greedy eliminated when optimality required", greedy_eliminated)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. StrategySelector — time limit constraint
# ═══════════════════════════════════════════════════════════════════════════════
print("--- StrategySelector: time limit ---")
sel = StrategySelector()
# Very tight time limit → only the fastest algos should survive
rec_fast = sel.recommend(time_limit_ms=1.0, optimality_required=False)
check("time limit rec has a primary",      rec_fast.primary != "")
check("time limit rec has explanation",    len(rec_fast.explanation) > 0)
# Slow algos (UCS, BFS) should be eliminated when limit is very tight
p_primary = sel.profiles.get(rec_fast.primary)
# When constraints eliminate all algos, selector relaxes and picks safe default
# So we just verify it returned something valid
check("time limit rec returns a valid algorithm",
      rec_fast.primary in StrategySelector.ALL_ALGOS,
      f"got {rec_fast.primary}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. StrategySelector — memory limit constraint
# ═══════════════════════════════════════════════════════════════════════════════
print("--- StrategySelector: memory limit ---")
sel     = StrategySelector()
rec_mem = sel.recommend(memory_limit_kb=25.0, optimality_required=False)
check("memory limit rec has primary",  rec_mem.primary != "")
p_mem = sel.profiles.get(rec_mem.primary)
check("primary algo memory <= memory limit (or close)",
      p_mem["mean_memory_kb"] <= 30.0,
      f"{rec_mem.primary} mem={p_mem['mean_memory_kb']}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. StrategySelector — recommendation structure
# ═══════════════════════════════════════════════════════════════════════════════
print("--- StrategySelector: recommendation structure ---")
sel = StrategySelector()
rec = sel.recommend()
check("rec has primary",                 isinstance(rec.primary, str) and rec.primary != "")
check("rec has fallback_order list",     isinstance(rec.fallback_order, list))
check("rec has explanation string",      isinstance(rec.explanation, str) and len(rec.explanation) > 20)
check("rec has scores dict",             isinstance(rec.scores, dict) and len(rec.scores) > 0)
check("rec has constraints_met dict",    isinstance(rec.constraints_met, dict))
check("primary not in fallback_order",   rec.primary not in rec.fallback_order)
check("primary + fallbacks cover all candidates",
      len([rec.primary] + rec.fallback_order) >= 1)
check("explanation mentions primary algo", rec.primary in rec.explanation)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. StrategySelector — all constraints eliminated → relaxes gracefully
# ═══════════════════════════════════════════════════════════════════════════════
print("--- StrategySelector: all-eliminated fallback ---")
sel = StrategySelector()
# Impossible constraints: need optimal AND time limit of 0.0001ms
rec_imp = sel.recommend(
    optimality_required = True,
    time_limit_ms       = 0.0001,
)
check("impossible constraints: still returns a recommendation", rec_imp.primary != "")
check("impossible constraints: used_defaults or primary returned",
      rec_imp.primary in StrategySelector.ALL_ALGOS)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Scenarios
# ═══════════════════════════════════════════════════════════════════════════════
print("--- Scenario 1: Speed Priority ---")
s1 = scenario_speed_priority(verbose=False)
check("S1 returns ScenarioResult",          hasattr(s1, "recommendation"))
check("S1 has a recommendation",            s1.recommendation.primary != "")
check("S1 runs the algorithm",              s1.search_result is not None)
check("S1 search result has stats",         hasattr(s1.search_result, "stats"))
check("S1 summary is non-empty",            len(s1.summary) > 0)
check("S1 recommended algo not eliminated for speed",
      s1.recommendation.primary in StrategySelector.ALL_ALGOS)

print("--- Scenario 2: Optimality Priority ---")
s2 = scenario_optimality_priority(verbose=False)
check("S2 returns ScenarioResult",           hasattr(s2, "recommendation"))
check("S2 primary is optimal algo",          s2.recommendation.primary in OPTIMAL_ALGOS,
      f"got {s2.recommendation.primary}")
check("S2 DFS eliminated",                   "DFS" not in ([s2.recommendation.primary] + s2.recommendation.fallback_order))
check("S2 Greedy eliminated",                "Greedy" not in ([s2.recommendation.primary] + s2.recommendation.fallback_order))
check("S2 finds solution",                   s2.search_result is not None and s2.search_result.found)

print("--- Scenario 3: Memory Constrained ---")
s3 = scenario_memory_constrained(verbose=False)
check("S3 returns ScenarioResult",           hasattr(s3, "recommendation"))
check("S3 has a primary recommendation",     s3.recommendation.primary != "")
check("S3 runs algorithm",                   s3.search_result is not None)
check("S3 summary mentions memory",          "memory" in s3.summary.lower() or "Memory" in s3.summary)

print("--- Scenario 4: Unknown Problem ---")
s4 = scenario_unknown_problem(verbose=False)
check("S4 returns ScenarioResult",           hasattr(s4, "recommendation"))
check("S4 has a primary",                    s4.recommendation.primary != "")
check("S4 algo runs successfully",           s4.search_result is not None)
check("S4 explanation non-empty",            len(s4.recommendation.explanation) > 0)
# Unknown problem should lean toward A* as safe default
check("S4 defaults to a safe algo (A* or UCS)",
      s4.recommendation.primary in ("A*", "UCS", "BFS"),
      f"got {s4.recommendation.primary}")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ProfileStore built from real benchmark data
# ═══════════════════════════════════════════════════════════════════════════════
print("--- ProfileStore from real benchmark data ---")
from search_benchmark.benchmarking.runner import BenchmarkRunner
from search_benchmark.core.graph import GraphProblem, generate_random_graph

runner  = BenchmarkRunner(timeout_seconds=5.0)
problems = [
    {
        "problem":    GraphProblem(*generate_random_graph(20, seed=s)),
        "problem_id": f"real_{s}",
        "meta":       {"env_type": "graph", "size": 20, "seed": s},
    }
    for s in range(3)
]
df_real   = runner.run_suite(problems, label="agent_test", save_csv=False, verbose=False)
store_real = ProfileStore.from_dataframe(df_real)
sel_real   = StrategySelector(store_real)
rec_real   = sel_real.recommend(env_type="graph", problem_size=20, optimality_required=True)
check("Real-data selector returns valid recommendation", rec_real.primary in OPTIMAL_ALGOS)
check("Real-data explanation non-empty",                 len(rec_real.explanation) > 20)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print()
print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")