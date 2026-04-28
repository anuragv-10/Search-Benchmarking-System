"""
algorithms/stats.py
-------------------
Dataclasses for recording what happened during a search run.

SearchStats   — raw instrumentation data collected during the search loop
SearchResult  — final outcome returned by every algorithm
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SearchStats:
    """
    Instrumentation data collected *during* a single search run.

    Every counter is updated inside the search loop by the algorithm itself.
    Post-solve fields (heuristic_error) are filled in after the solution
    is found by calling compute_heuristic_error().
    """

    # ── Expansion counters ─────────────────────────────────────
    nodes_expanded:    int   = 0   # nodes popped from the frontier
    nodes_generated:   int   = 0   # child nodes created via expand()
    max_frontier_size: int   = 0   # peak length of the frontier structure

    # ── Resource usage ─────────────────────────────────────────
    runtime_ms:        float = 0.0  # wall-clock time  (perf_counter diff × 1000)
    peak_memory_kb:    float = 0.0  # tracemalloc peak snapshot ÷ 1024

    # ── Solution quality ───────────────────────────────────────
    path_cost:         Optional[float] = None   # g(solution_node); None = no solution
    solution_depth:    Optional[int]   = None   # len(solution()) if found
    solution_found:    bool            = False

    # ── Failure info ───────────────────────────────────────────
    failure_reason:    Optional[str]   = None   # 'timeout' | 'no_path' | 'depth_limit'

    # ── Heuristic quality (filled post-solve) ──────────────────
    heuristic_error:   Optional[float] = None   # mean |h(n) − h*(n)| on solution path

    # ── Step-by-step frontier trace (for visualiser) ───────────
    # Each entry: {"expanded": state, "frontier": [state, ...]}
    expansion_log:     List[dict]      = field(default_factory=list)
    record_log:        bool            = False   # set True to populate expansion_log

    # ── Re-expansion counter (A* with inconsistent h) ──────────
    re_expansions:     int             = 0

    def as_dict(self) -> dict:
        """Serialise to a plain dict (for DataFrame rows and JSON API)."""
        return {
            "nodes_expanded":    self.nodes_expanded,
            "nodes_generated":   self.nodes_generated,
            "max_frontier_size": self.max_frontier_size,
            "runtime_ms":        round(self.runtime_ms, 4),
            "peak_memory_kb":    round(self.peak_memory_kb, 3),
            "path_cost":         self.path_cost,
            "solution_depth":    self.solution_depth,
            "solution_found":    self.solution_found,
            "failure_reason":    self.failure_reason,
            "heuristic_error":   self.heuristic_error,
            "re_expansions":     self.re_expansions,
        }


@dataclass
class SearchResult:
    """
    Final outcome returned by every algorithm's solve() function.

    Algorithms return this; callers inspect .stats for metrics
    and .path for the actual solution sequence.
    """

    stats:      SearchStats
    path:       Optional[List[Any]] = None   # list of states root → goal
    actions:    Optional[List[Any]] = None   # list of actions root → goal
    algo_name:  str                 = ""

    @property
    def found(self) -> bool:
        return self.stats.solution_found

    def summary(self) -> str:
        s = self.stats
        if s.solution_found:
            return (
                f"[{self.algo_name}] FOUND  "
                f"cost={s.path_cost:.3f}  depth={s.solution_depth}  "
                f"expanded={s.nodes_expanded}  time={s.runtime_ms:.2f}ms  "
                f"mem={s.peak_memory_kb:.1f}KB"
            )
        return (
            f"[{self.algo_name}] FAILED ({s.failure_reason})  "
            f"expanded={s.nodes_expanded}  time={s.runtime_ms:.2f}ms"
        )