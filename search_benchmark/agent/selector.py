"""
agent/selector.py
-----------------
StrategySelector
    Given a new problem and user-defined constraints, picks the best
    algorithm using a three-step hybrid:

    Step 1 — Hard Constraint Filter
        Eliminate any algorithm that cannot possibly satisfy the constraints.

    Step 2 — Weighted Scoring
        Score remaining candidates on speed, memory, and optimality.

    Step 3 — Recommend + Explain
        Return the winner with a plain-English explanation and a ranked
        fallback list.  If the primary fails at runtime, auto-retry with
        the first fallback.

Constraint Parameters
---------------------
optimality_required : bool   — must return an optimal (lowest cost) path
time_limit_ms       : float  — estimated max runtime in ms (None = no limit)
memory_limit_kb     : float  — estimated max peak memory in KB (None = no limit)
speed_weight        : float  — weight for speed in scoring  (default 0.4)
memory_weight       : float  — weight for memory in scoring (default 0.2)
quality_weight      : float  — weight for optimality rate   (default 0.4)
env_type            : str    — 'graph' | 'grid'
problem_size        : int    — number of nodes / grid side length
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from .profiles import ProfileStore, DEFAULT_PROFILES
except ImportError:
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from search_benchmark.agent.profiles import ProfileStore, DEFAULT_PROFILES

# Algorithms that guarantee optimal solutions
OPTIMAL_ALGOS = {"UCS", "A*", "BFS"}   # BFS optimal on unit-cost only — flagged in explanation

# Safe default when we have no profile data
SAFE_DEFAULT = "A*"


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    """Output of StrategySelector.recommend()."""
    primary:         str                  # chosen algorithm name
    fallback_order:  List[str]            # remaining candidates, ranked
    explanation:     str                  # human-readable rationale
    scores:          Dict[str, float]     # final score per candidate
    constraints_met: Dict[str, bool]      # which constraints were satisfiable
    used_defaults:   bool = False         # True if no profile data was available


# ─────────────────────────────────────────────────────────────────────────────
# StrategySelector
# ─────────────────────────────────────────────────────────────────────────────

class StrategySelector:
    """
    Rational algorithm selector driven by empirical profiles.

    Parameters
    ----------
    profiles : ProfileStore instance.  If None, uses default profiles.
    """

    ALL_ALGOS = ["BFS", "DFS", "UCS", "Greedy", "A*"]

    def __init__(self, profiles: Optional[ProfileStore] = None) -> None:
        self.profiles = profiles or ProfileStore.default()

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def recommend(
        self,
        env_type:            str   = "graph",
        problem_size:        int   = 50,
        optimality_required: bool  = False,
        time_limit_ms:       Optional[float] = None,
        memory_limit_kb:     Optional[float] = None,
        speed_weight:        float = 0.4,
        memory_weight:       float = 0.2,
        quality_weight:      float = 0.4,
    ) -> Recommendation:
        """
        Recommend the best algorithm for the described problem.

        Returns a Recommendation with primary choice, fallback list,
        and a plain-English explanation.
        """
        # ── Fetch profiles for all algorithms ────────────────
        algo_profiles = {
            algo: self.profiles.get(algo, env_type, problem_size)
            for algo in self.ALL_ALGOS
        }

        # ── Step 1: Hard constraint filter ───────────────────
        candidates, eliminated, constraint_notes = self._filter(
            algo_profiles,
            optimality_required = optimality_required,
            time_limit_ms       = time_limit_ms,
            memory_limit_kb     = memory_limit_kb,
        )

        used_defaults = False
        if not candidates:
            # All eliminated — relax constraints and warn
            candidates    = list(self.ALL_ALGOS)
            used_defaults = True
            constraint_notes["relaxed"] = (
                "All algorithms were eliminated by constraints. "
                "Constraints relaxed — recommending safe default."
            )

        # ── Step 2: Score remaining candidates ───────────────
        scores = self._score(
            candidates,
            algo_profiles,
            speed_weight   = speed_weight,
            memory_weight  = memory_weight,
            quality_weight = quality_weight,
        )

        # Rank by score descending
        ranked = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)

        primary       = ranked[0]
        fallback_list = ranked[1:]

        # ── Step 3: Build explanation ─────────────────────────
        explanation = self._explain(
            primary         = primary,
            ranked          = ranked,
            eliminated      = eliminated,
            constraint_notes= constraint_notes,
            algo_profiles   = algo_profiles,
            optimality_required = optimality_required,
            time_limit_ms   = time_limit_ms,
            memory_limit_kb = memory_limit_kb,
            env_type        = env_type,
            problem_size    = problem_size,
            used_defaults   = used_defaults,
        )

        constraints_met = {
            "optimality": not optimality_required or primary in OPTIMAL_ALGOS,
            "time":       time_limit_ms is None or algo_profiles[primary]["mean_runtime_ms"] <= time_limit_ms,
            "memory":     memory_limit_kb is None or algo_profiles[primary]["mean_memory_kb"] <= memory_limit_kb,
        }

        return Recommendation(
            primary         = primary,
            fallback_order  = fallback_list,
            explanation     = explanation,
            scores          = scores,
            constraints_met = constraints_met,
            used_defaults   = used_defaults,
        )

    # ─────────────────────────────────────────────────────────
    # Step 1 — Hard constraint filter
    # ─────────────────────────────────────────────────────────

    def _filter(
        self,
        algo_profiles:       Dict[str, dict],
        optimality_required: bool,
        time_limit_ms:       Optional[float],
        memory_limit_kb:     Optional[float],
    ):
        candidates   = []
        eliminated   = {}
        constraint_notes = {}

        for algo in self.ALL_ALGOS:
            p      = algo_profiles[algo]
            reason = None

            if optimality_required and algo not in OPTIMAL_ALGOS:
                reason = f"not guaranteed optimal (optimal_rate={p['optimal_rate']:.0%})"

            elif time_limit_ms is not None:
                # Use 80% of the limit as the safety margin
                if p["mean_runtime_ms"] > time_limit_ms * 0.8:
                    reason = (
                        f"mean runtime {p['mean_runtime_ms']:.1f}ms "
                        f"exceeds {time_limit_ms*0.8:.1f}ms limit"
                    )

            elif memory_limit_kb is not None:
                if p["mean_memory_kb"] > memory_limit_kb * 0.8:
                    reason = (
                        f"mean memory {p['mean_memory_kb']:.1f}KB "
                        f"exceeds {memory_limit_kb*0.8:.1f}KB limit"
                    )

            if reason:
                eliminated[algo] = reason
            else:
                candidates.append(algo)

        return candidates, eliminated, constraint_notes

    # ─────────────────────────────────────────────────────────
    # Step 2 — Weighted scoring
    # ─────────────────────────────────────────────────────────

    def _score(
        self,
        candidates:    List[str],
        algo_profiles: Dict[str, dict],
        speed_weight:  float,
        memory_weight: float,
        quality_weight:float,
    ) -> Dict[str, float]:
        if not candidates:
            return {}

        # Gather raw values
        runtimes  = {a: max(algo_profiles[a]["mean_runtime_ms"],  0.001) for a in candidates}
        memories  = {a: max(algo_profiles[a]["mean_memory_kb"],   0.001) for a in candidates}
        qualities = {a: algo_profiles[a]["optimal_rate"]                  for a in candidates}

        # Normalise: lower runtime/memory = higher score
        max_rt  = max(runtimes.values())
        max_mem = max(memories.values())

        scores = {}
        for a in candidates:
            speed_score   = 1.0 - (runtimes[a]  / max_rt)
            memory_score  = 1.0 - (memories[a]  / max_mem)
            quality_score = qualities[a]

            # Bonus for high success rate
            sr_bonus = algo_profiles[a].get("success_rate", 1.0) * 0.05

            scores[a] = (
                speed_weight   * speed_score   +
                memory_weight  * memory_score  +
                quality_weight * quality_score +
                sr_bonus
            )

        return scores

    # ─────────────────────────────────────────────────────────
    # Step 3 — Explanation builder
    # ─────────────────────────────────────────────────────────

    def _explain(
        self,
        primary:             str,
        ranked:              List[str],
        eliminated:          Dict[str, str],
        constraint_notes:    Dict[str, str],
        algo_profiles:       Dict[str, dict],
        optimality_required: bool,
        time_limit_ms:       Optional[float],
        memory_limit_kb:     Optional[float],
        env_type:            str,
        problem_size:        int,
        used_defaults:       bool,
    ) -> str:
        p    = algo_profiles[primary]
        lines = []

        # Header
        lines.append(f"Recommended: {primary}")
        lines.append(f"Problem: {env_type} environment, size ≈ {problem_size}")
        lines.append("")

        # Why this algorithm
        reasons = []
        if primary == "A*":
            reasons.append("A* balances optimality and efficiency using a heuristic to guide the search.")
        elif primary == "UCS":
            reasons.append("UCS guarantees the lowest-cost path by exploring in order of cumulative cost.")
        elif primary == "BFS":
            reasons.append("BFS finds the shallowest solution and is optimal for unit-cost problems.")
        elif primary == "Greedy":
            reasons.append("Greedy Best-First is the fastest option — it expands very few nodes by always heading toward the goal estimate.")
        elif primary == "DFS":
            reasons.append("DFS uses the least memory of all algorithms — ideal when RAM is the bottleneck.")

        lines.append("Why: " + " ".join(reasons))
        lines.append("")

        # Profile numbers
        lines.append(f"Expected performance (from benchmarks):")
        lines.append(f"  Runtime      ≈ {p['mean_runtime_ms']:.1f} ms")
        lines.append(f"  Memory       ≈ {p['mean_memory_kb']:.1f} KB")
        lines.append(f"  Nodes        ≈ {p['mean_nodes_expanded']:.0f} expanded")
        lines.append(f"  Optimal rate = {p['optimal_rate']:.0%}")
        lines.append(f"  Success rate = {p['success_rate']:.0%}")
        lines.append("")

        # Constraints
        if optimality_required:
            lines.append("Constraint — Optimality required: ✓  " +
                          ("(guaranteed)" if primary in OPTIMAL_ALGOS else "(WARNING: not guaranteed)"))
        if time_limit_ms is not None:
            ok = p["mean_runtime_ms"] <= time_limit_ms
            lines.append(f"Constraint — Time limit {time_limit_ms:.0f}ms: {'✓' if ok else '✗'}")
        if memory_limit_kb is not None:
            ok = p["mean_memory_kb"] <= memory_limit_kb
            lines.append(f"Constraint — Memory limit {memory_limit_kb:.0f}KB: {'✓' if ok else '✗'}")

        # Eliminated algorithms
        if eliminated:
            lines.append("")
            lines.append("Eliminated:")
            for algo, reason in eliminated.items():
                lines.append(f"  {algo}: {reason}")

        # Fallback
        if ranked[1:]:
            lines.append("")
            lines.append(f"Fallback order: {' → '.join(ranked[1:])}")

        # Defaults warning
        if used_defaults:
            lines.append("")
            lines.append(
                "Note: No matching benchmark profile found. "
                f"Using default profiles. Run the benchmark suite first for "
                f"more accurate recommendations on {env_type} problems of size {problem_size}."
            )

        if constraint_notes.get("relaxed"):
            lines.append("")
            lines.append(f"Warning: {constraint_notes['relaxed']}")

        return "\n".join(lines)
