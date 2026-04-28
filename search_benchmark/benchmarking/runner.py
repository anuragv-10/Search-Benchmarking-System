"""
benchmarking/runner.py
----------------------
BenchmarkRunner
    Unified runner that executes any subset of the five algorithms on
    any Problem instance, collects SearchStats, and returns structured
    results as a pandas DataFrame.

Key design decisions
--------------------
- Every algorithm call is wrapped in a concurrent.futures thread so that
  the timeout is enforced without killing the main process.
- Results are row-normalised: one DataFrame row per (problem_id, algo).
- The runner is stateless; call run_suite() multiple times with different
  problem lists and concatenate the DataFrames yourself.
"""

from __future__ import annotations

import concurrent.futures
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

try:
    from ..algorithms import ALGORITHMS
    from ..algorithms.stats import SearchResult, SearchStats
    from ..core.problem import Problem
except ImportError:
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.algorithms import ALGORITHMS
    from search_benchmark.algorithms.stats import SearchResult, SearchStats
    from search_benchmark.core.problem import Problem


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkRunner
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Orchestrates benchmark experiments.

    Parameters
    ----------
    timeout_seconds : Wall-clock limit per individual algorithm run.
    record_log      : If True, populate expansion_log in every SearchStats.
                      Useful for visualisation; disable for large experiments.
    algorithms      : Subset of algorithm names to run.
                      Defaults to all five: BFS, DFS, UCS, Greedy, A*.
    """

    ALL_ALGOS = ["BFS", "DFS", "UCS", "Greedy", "A*"]

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        record_log:      bool  = False,
        algorithms:      Optional[List[str]] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.record_log      = record_log
        self.algorithms      = algorithms or self.ALL_ALGOS

    # ─────────────────────────────────────────────────────────
    # Single run
    # ─────────────────────────────────────────────────────────

    def run_single(
        self,
        problem:   Problem,
        algo_name: str,
        heuristic: Optional[Callable] = None,
    ) -> SearchResult:
        """
        Run one algorithm on one problem with timeout enforcement.

        Returns a SearchResult.  If the timeout fires, returns a
        SearchResult with solution_found=False and failure_reason='timeout'.
        """
        if algo_name not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algo_name}'. "
                f"Available: {list(ALGORITHMS)}"
            )

        module = ALGORITHMS[algo_name]

        def _run():
            return module.solve(
                problem,
                heuristic   = heuristic,
                timeout     = self.timeout_seconds,
                record_log  = self.record_log,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run)
            try:
                return future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                stats = SearchStats()
                stats.solution_found  = False
                stats.failure_reason  = "timeout"
                stats.runtime_ms      = self.timeout_seconds * 1000
                return SearchResult(stats=stats, algo_name=algo_name)
            except Exception as e:
                stats = SearchStats()
                stats.solution_found = False
                stats.failure_reason = f"error: {e}"
                return SearchResult(stats=stats, algo_name=algo_name)

    # ─────────────────────────────────────────────────────────
    # Run all algorithms on one problem
    # ─────────────────────────────────────────────────────────

    def run_all(
        self,
        problem:    Problem,
        heuristics: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, SearchResult]:
        """
        Run all configured algorithms on *problem*.

        Parameters
        ----------
        problem    : The problem instance.
        heuristics : Optional dict mapping algo_name → heuristic callable.
                     Algorithms not in the dict use problem.h.

        Returns
        -------
        Dict mapping algo_name → SearchResult.
        """
        heuristics = heuristics or {}
        results    = {}
        for name in self.algorithms:
            h = heuristics.get(name)
            results[name] = self.run_single(problem, name, heuristic=h)
        return results

    # ─────────────────────────────────────────────────────────
    # Run a suite of problems
    # ─────────────────────────────────────────────────────────

    def run_suite(
        self,
        problem_list: List[Dict[str, Any]],
        label:        str  = "suite",
        save_csv:     bool = True,
        csv_dir:      str  = "results",
        verbose:      bool = True,
    ) -> pd.DataFrame:
        """
        Run all algorithms on every problem in *problem_list*.

        Parameters
        ----------
        problem_list : List of dicts, each with keys:
                         'problem'    : Problem instance  (required)
                         'problem_id' : str label         (optional, auto-generated)
                         'meta'       : dict of extra columns to attach (optional)
                         'heuristics' : dict algo→callable (optional)
        label        : Used in the saved CSV filename.
        save_csv     : If True, write results/{label}.csv after completion.
        csv_dir      : Directory for CSV output.
        verbose      : Print progress to stdout.

        Returns
        -------
        pd.DataFrame with one row per (problem_id, algo).
        """
        rows = []

        for idx, entry in enumerate(problem_list):
            problem    = entry["problem"]
            problem_id = entry.get("problem_id", f"{label}_{idx:04d}")
            meta       = entry.get("meta", {})
            heuristics = entry.get("heuristics", {})

            if verbose:
                print(f"  [{idx+1:3d}/{len(problem_list)}] {problem_id}", end=" ", flush=True)

            results = self.run_all(problem, heuristics=heuristics)

            for algo_name, result in results.items():
                row = {
                    "problem_id": problem_id,
                    "algo":       algo_name,
                    **meta,
                    **result.stats.as_dict(),
                }
                rows.append(row)

            if verbose:
                found_str = "  ".join(
                    f"{n}={'✓' if r.found else '✗'}"
                    for n, r in results.items()
                )
                print(f"→  {found_str}")

        df = pd.DataFrame(rows)

        if save_csv and len(df) > 0:
            import os
            os.makedirs(csv_dir, exist_ok=True)
            path = os.path.join(csv_dir, f"{label}.csv")
            df.to_csv(path, index=False)
            if verbose:
                print(f"\n  Saved → {path}  ({len(df)} rows)")

        return df

    # ─────────────────────────────────────────────────────────
    # Summary statistics
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def summarise(df: pd.DataFrame, group_cols: List[str] = None) -> pd.DataFrame:
        """
        Aggregate a raw results DataFrame into mean ± std per group.

        Parameters
        ----------
        df         : Raw DataFrame from run_suite().
        group_cols : Columns to group by.
                     Defaults to ['algo'] plus any 'size' or 'env_type' columns present.

        Returns
        -------
        pd.DataFrame with mean and std columns for every numeric metric.
        """
        if group_cols is None:
            candidates = ["algo", "env_type", "size", "heuristic_name", "wall_density"]
            group_cols = [c for c in candidates if c in df.columns]

        numeric_cols = [
            "nodes_expanded", "nodes_generated", "max_frontier_size",
            "runtime_ms", "peak_memory_kb", "path_cost",
            "solution_depth", "heuristic_error", "re_expansions",
        ]
        numeric_cols = [c for c in numeric_cols if c in df.columns]

        agg = df.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).round(4)
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index()

        # Add success rate
        sr = df.groupby(group_cols)["solution_found"].mean().reset_index()
        sr.columns = list(sr.columns[:-1]) + ["success_rate"]
        agg = agg.merge(sr, on=group_cols)

        return agg
