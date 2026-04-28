"""
agent/profiles.py
-----------------
Builds and loads empirical strategy profiles for each algorithm.

A profile captures how an algorithm *actually* behaves across different
problem types and sizes, based on the benchmarking results from Section 3.

A profile is keyed by:  (algo_name, env_type, size_bucket)

Size buckets
------------
  "tiny"   :  size  ≤ 20
  "small"  :  size  ≤ 50
  "medium" :  size  ≤ 100
  "large"  :  size  > 100

Each profile entry contains:
  mean_runtime_ms, mean_nodes_expanded, mean_memory_kb,
  optimal_rate, success_rate, failure_modes
"""

from __future__ import annotations
import json
import os
from typing import Dict, Optional

import pandas as pd

# ── Size bucket mapping ───────────────────────────────────────────────────────

def size_bucket(size: int) -> str:
    if   size <= 20:  return "tiny"
    elif size <= 50:  return "small"
    elif size <= 100: return "medium"
    else:             return "large"


# ── Default fallback profiles ─────────────────────────────────────────────────
# Used when no benchmark data exists yet (first run, or data was deleted).
# Values derived from typical behaviour on moderate-size problems.

DEFAULT_PROFILES: Dict[str, dict] = {
    "BFS":    {"mean_runtime_ms": 8.0,  "mean_nodes_expanded": 60,  "mean_memory_kb": 120.0,
               "optimal_rate": 0.85, "success_rate": 0.95, "failure_modes": ["no_path"]},
    "DFS":    {"mean_runtime_ms": 4.0,  "mean_nodes_expanded": 55,  "mean_memory_kb": 30.0,
               "optimal_rate": 0.10, "success_rate": 0.90, "failure_modes": ["no_path", "depth_limit"]},
    "UCS":    {"mean_runtime_ms": 10.0, "mean_nodes_expanded": 80,  "mean_memory_kb": 130.0,
               "optimal_rate": 1.00, "success_rate": 0.95, "failure_modes": ["no_path"]},
    "Greedy": {"mean_runtime_ms": 2.0,  "mean_nodes_expanded": 8,   "mean_memory_kb": 20.0,
               "optimal_rate": 0.20, "success_rate": 0.85, "failure_modes": ["no_path"]},
    "A*":     {"mean_runtime_ms": 6.0,  "mean_nodes_expanded": 30,  "mean_memory_kb": 60.0,
               "optimal_rate": 1.00, "success_rate": 0.95, "failure_modes": ["no_path"]},
}


# ─────────────────────────────────────────────────────────────────────────────
# ProfileStore
# ─────────────────────────────────────────────────────────────────────────────

class ProfileStore:
    """
    Holds empirical profiles for all algorithms.

    Usage
    -----
    store = ProfileStore.from_dataframe(df)   # built from benchmark results
    store = ProfileStore.load("results/profiles.json")  # loaded from disk
    profile = store.get("A*", env_type="graph", size=80)
    store.save("results/profiles.json")
    """

    def __init__(self, profiles: Dict[str, dict]) -> None:
        # profiles: flat dict keyed by "{algo}__{env_type}__{bucket}"
        # plus a fallback keyed by just "{algo}"
        self._profiles = profiles

    # ─────────────────────────────────────────────────────────
    # Build from benchmark DataFrame
    # ─────────────────────────────────────────────────────────

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ProfileStore":
        """
        Build profiles from a run_suite() DataFrame.
        Groups by (algo, env_type, size_bucket) and aggregates metrics.
        """
        profiles: Dict[str, dict] = {}

        # Add size_bucket column
        working = df.copy()
        if "size" in working.columns:
            working["size_bucket"] = working["size"].apply(size_bucket)
        else:
            working["size_bucket"] = "medium"

        if "env_type" not in working.columns:
            working["env_type"] = "graph"

        numeric_cols = ["runtime_ms", "nodes_expanded", "peak_memory_kb", "path_cost"]
        group_cols   = ["algo", "env_type", "size_bucket"]

        for keys, grp in working.groupby(group_cols):
            algo, env_type, bucket = keys
            key = f"{algo}__{env_type}__{bucket}"

            # Optimal rate: fraction where path_cost == UCS path_cost on same problem_id
            ucs_costs = (
                working[
                    (working["algo"] == "UCS") &
                    (working["env_type"] == env_type) &
                    (working["size_bucket"] == bucket)
                ]
                .set_index("problem_id")["path_cost"]
                .to_dict()
                if "problem_id" in working.columns else {}
            )

            optimal_count = 0
            total_solved  = 0
            if "problem_id" in grp.columns:
                for _, row in grp[grp["solution_found"] == True].iterrows():
                    pid       = row.get("problem_id")
                    ucs_cost  = ucs_costs.get(pid)
                    algo_cost = row.get("path_cost")
                    if ucs_cost is not None and algo_cost is not None and ucs_cost > 0:
                        if abs(algo_cost - ucs_cost) / ucs_cost < 0.01:
                            optimal_count += 1
                    total_solved += 1

            failure_modes = (
                grp[grp["failure_reason"].notna()]["failure_reason"]
                .unique().tolist()
            ) if "failure_reason" in grp.columns else []

            profiles[key] = {
                "mean_runtime_ms":      round(grp["runtime_ms"].mean(),      4) if "runtime_ms"      in grp else 0.0,
                "mean_nodes_expanded":  round(grp["nodes_expanded"].mean(),  2) if "nodes_expanded"  in grp else 0.0,
                "mean_memory_kb":       round(grp["peak_memory_kb"].mean(),  3) if "peak_memory_kb"  in grp else 0.0,
                "optimal_rate":         round(optimal_count / max(total_solved, 1), 3),
                "success_rate":         round(grp["solution_found"].mean(),  3) if "solution_found" in grp else 1.0,
                "failure_modes":        failure_modes,
            }

            # Also store a plain algo-level fallback (average across all buckets)
            algo_key = f"{algo}__any__any"
            if algo_key not in profiles:
                profiles[algo_key] = dict(profiles[key])

        # Add DEFAULT entries for algos with no data
        for algo, default in DEFAULT_PROFILES.items():
            fallback_key = f"{algo}__any__any"
            if fallback_key not in profiles:
                profiles[fallback_key] = dict(default)

        return cls(profiles)

    @classmethod
    def default(cls) -> "ProfileStore":
        """Return a ProfileStore built from DEFAULT_PROFILES only."""
        profiles = {f"{a}__any__any": dict(p) for a, p in DEFAULT_PROFILES.items()}
        return cls(profiles)

    # ─────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────

    def get(
        self,
        algo:     str,
        env_type: str = "graph",
        size:     int = 50,
    ) -> dict:
        """
        Return the best matching profile for the given context.
        Falls back progressively: specific → env-specific → algo-level → default.
        """
        bucket = size_bucket(size)
        candidates = [
            f"{algo}__{env_type}__{bucket}",
            f"{algo}__any__{bucket}",
            f"{algo}__{env_type}__any",
            f"{algo}__any__any",
        ]
        for key in candidates:
            if key in self._profiles:
                return dict(self._profiles[key])

        return dict(DEFAULT_PROFILES.get(algo, DEFAULT_PROFILES["BFS"]))

    def all_algos(self) -> list:
        return list(DEFAULT_PROFILES.keys())

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def save(self, path: str = "results/profiles.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._profiles, f, indent=2)

    @classmethod
    def load(cls, path: str = "results/profiles.json") -> "ProfileStore":
        if not os.path.exists(path):
            return cls.default()
        with open(path) as f:
            return cls(json.load(f))
