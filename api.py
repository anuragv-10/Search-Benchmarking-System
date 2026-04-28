"""
api.py
------
FastAPI backend for the Search Algorithm Benchmarking & Analysis Tool.

Run with:
    python api.py

Then open http://localhost:8000 in your browser.

Dependencies:
    pip install fastapi uvicorn numpy pandas matplotlib
"""

from __future__ import annotations
import base64
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional

# ── ensure project root is on path ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from search_benchmark.core.graph     import GraphProblem, WeightedGraph, generate_random_graph
from search_benchmark.core.grid      import Grid, GridProblem, generate_maze, generate_unsolvable_maze
from search_benchmark.core.heuristics import get_heuristic, HEURISTIC_REGISTRY
from search_benchmark.algorithms     import ALGORITHMS
from search_benchmark.benchmarking.runner import BenchmarkRunner
from search_benchmark.analysis.charts    import generate_all_charts, chart_c6_frontier_growth
from search_benchmark.analysis.heuristics_analysis import full_heuristic_report
from search_benchmark.agent.profiles  import ProfileStore
from search_benchmark.agent.selector  import StrategySelector

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Search Algorithm Benchmarking Tool",
    description = "DSE 3241 — Principles of Artificial Intelligence",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve index.html at root
@app.get("/", include_in_schema=False)
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Search Benchmark API — open /docs for API reference"}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class GraphConfig(BaseModel):
    n_nodes:      int   = Field(20, ge=5,  le=300)
    edge_density: float = Field(0.2, ge=0.05, le=0.8)
    weight_range: List[float] = Field([1.0, 10.0])
    seed:         int   = Field(42, ge=0)

class GridConfig(BaseModel):
    rows:         int   = Field(10, ge=5, le=60)
    cols:         int   = Field(10, ge=5, le=60)
    wall_density: float = Field(0.3, ge=0.0, le=0.6)
    seed:         int   = Field(42, ge=0)
    heuristic:    str   = Field("manhattan")

class SolveRequest(BaseModel):
    env_type:       str            = "graph"
    problem_config: Dict[str, Any] = {}
    algo:           str            = "A*"
    heuristic_name: Optional[str]  = None
    heuristic_scale:float          = 1.0
    timeout:        float          = Field(10.0, ge=0.5, le=30.0)
    record_log:     bool           = False

class BenchmarkRequest(BaseModel):
    env_type:        str            = "graph"
    problem_config:  Dict[str, Any] = {}
    heuristic_name:  Optional[str]  = "manhattan"
    heuristic_scale: float          = 1.0
    timeout:         float          = Field(10.0, ge=0.5, le=30.0)
    algorithms:      Optional[List[str]] = None

class HeuristicRequest(BaseModel):
    env_type:        str            = "grid"
    problem_config:  Dict[str, Any] = {}
    heuristic_name:  str            = "manhattan"

class AgentRequest(BaseModel):
    env_type:            str   = "graph"
    problem_size:        int   = Field(50, ge=5, le=300)
    optimality_required: bool  = False
    time_limit_ms:       Optional[float] = None
    memory_limit_kb:     Optional[float] = None
    speed_weight:        float = 0.4
    memory_weight:       float = 0.2
    quality_weight:      float = 0.4
    # Optional: also run the recommended algorithm
    problem_config:      Optional[Dict[str, Any]] = None
    timeout:             float = 10.0

class ExperimentRequest(BaseModel):
    experiment: str  = Field("A", pattern="^[ABCD]$")
    verbose:    bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph_problem(cfg: dict, heuristic_scale: float = 1.0) -> GraphProblem:
    n        = cfg.get("n_nodes", 20)
    density  = cfg.get("edge_density", 0.2)
    w_range  = tuple(cfg.get("weight_range", [1.0, 10.0]))
    seed     = cfg.get("seed", 42)
    g, s, e  = generate_random_graph(n, density, w_range, seed)
    return GraphProblem(g, s, e, heuristic_scale=heuristic_scale)


def _build_grid_problem(cfg: dict, heuristic_scale: float = 1.0) -> GridProblem:
    rows    = cfg.get("rows", 10)
    cols    = cfg.get("cols", 10)
    density = cfg.get("wall_density", 0.3)
    seed    = cfg.get("seed", 42)
    h_name  = cfg.get("heuristic", "manhattan")
    prob, _ = generate_maze(rows, cols, density, seed, heuristic=h_name)
    prob.heuristic_scale = heuristic_scale
    return prob


def _build_problem(env_type: str, cfg: dict, scale: float = 1.0):
    if env_type == "graph":
        return _build_graph_problem(cfg, scale)
    elif env_type == "grid":
        return _build_grid_problem(cfg, scale)
    else:
        raise HTTPException(400, f"Unknown env_type '{env_type}'. Use 'graph' or 'grid'.")


def _result_to_dict(result, algo_name: str = "") -> dict:
    s = result.stats
    return {
        "algo":            algo_name or result.algo_name,
        "solution_found":  s.solution_found,
        "path_cost":       s.path_cost,
        "solution_depth":  s.solution_depth,
        "nodes_expanded":  s.nodes_expanded,
        "nodes_generated": s.nodes_generated,
        "max_frontier":    s.max_frontier_size,
        "runtime_ms":      round(s.runtime_ms, 4),
        "peak_memory_kb":  round(s.peak_memory_kb, 3),
        "heuristic_error": s.heuristic_error,
        "re_expansions":   s.re_expansions,
        "failure_reason":  s.failure_reason,
        "path":            result.path,
        "actions":         result.actions,
        "expansion_log":   s.expansion_log if s.record_log else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Problem generation
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/generate/graph", summary="Generate a random weighted graph")
def generate_graph(cfg: GraphConfig):
    """
    Generate a reproducible weighted graph.
    Returns node positions, edges, start/goal IDs.
    """
    g, start, goal = generate_random_graph(
        n_nodes      = cfg.n_nodes,
        edge_density = cfg.edge_density,
        weight_range = tuple(cfg.weight_range),
        seed         = cfg.seed,
    )
    return {
        "type":  "graph",
        "start": start,
        "goal":  goal,
        **g.to_dict(),
    }


@app.post("/generate/grid", summary="Generate a random grid maze")
def generate_grid(cfg: GridConfig):
    """
    Generate a reproducible grid maze.
    Returns a 2-D array (0=free, 1=wall), start/goal coords.
    """
    prob, grid = generate_maze(
        rows         = cfg.rows,
        cols         = cfg.cols,
        wall_density = cfg.wall_density,
        seed         = cfg.seed,
        heuristic    = cfg.heuristic,
    )
    return {
        "type":  "grid",
        "start": list(prob.initial_state),
        "goal":  list(prob.goal_state),
        **grid.to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Single solve
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/solve", summary="Run one algorithm on a problem")
def solve(req: SolveRequest):
    """
    Run a single algorithm on a generated problem.
    Set record_log=true to get step-by-step expansion data for animation.
    """
    if req.algo not in ALGORITHMS:
        raise HTTPException(400, f"Unknown algo '{req.algo}'. Valid: {list(ALGORITHMS)}")

    problem = _build_problem(req.env_type, req.problem_config, req.heuristic_scale)
    runner  = BenchmarkRunner(timeout_seconds=req.timeout, record_log=req.record_log)
    result  = runner.run_single(problem, req.algo)

    return _result_to_dict(result, req.algo)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Benchmark all algorithms
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/benchmark", summary="Run all algorithms on a problem and compare")
def benchmark(req: BenchmarkRequest):
    """
    Run all (or a subset of) algorithms on the same problem instance.
    Returns a list of result dicts — one per algorithm.
    """
    algos   = req.algorithms or list(ALGORITHMS.keys())
    invalid = [a for a in algos if a not in ALGORITHMS]
    if invalid:
        raise HTTPException(400, f"Unknown algorithms: {invalid}")

    problem = _build_problem(req.env_type, req.problem_config, req.heuristic_scale)
    runner  = BenchmarkRunner(timeout_seconds=req.timeout, algorithms=algos)
    results = runner.run_all(problem)

    return {
        "results":     [_result_to_dict(r, name) for name, r in results.items()],
        "env_type":    req.env_type,
        "heuristic":   req.heuristic_name,
        "scale":       req.heuristic_scale,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Heuristic analysis
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze/heuristic", summary="Admissibility & consistency report")
def analyze_heuristic(req: HeuristicRequest):
    """
    Run A* on a problem and check heuristic admissibility + consistency.
    Returns admissibility_rate, consistency_rate, accuracy score.
    """
    if req.env_type != "grid":
        raise HTTPException(400, "Heuristic analysis currently supports grid problems only.")

    problem = _build_grid_problem(req.problem_config)
    from search_benchmark.algorithms import astar
    result  = astar.solve(problem)
    report  = full_heuristic_report(problem, result)
    return {"env_type": req.env_type, **report}


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Agent
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/agent/recommend", summary="Get algorithm recommendation from agent")
def agent_recommend(req: AgentRequest):
    """
    Ask the agent to recommend the best algorithm given constraints.
    Optionally runs the recommended algorithm if problem_config is provided.
    """
    # Load profiles from disk if available
    profiles = ProfileStore.load("results/profiles.json")
    selector = StrategySelector(profiles)

    rec = selector.recommend(
        env_type            = req.env_type,
        problem_size        = req.problem_size,
        optimality_required = req.optimality_required,
        time_limit_ms       = req.time_limit_ms,
        memory_limit_kb     = req.memory_limit_kb,
        speed_weight        = req.speed_weight,
        memory_weight       = req.memory_weight,
        quality_weight      = req.quality_weight,
    )

    response = {
        "recommended_algo":  rec.primary,
        "fallback_order":    rec.fallback_order,
        "explanation":       rec.explanation,
        "scores":            rec.scores,
        "constraints_met":   rec.constraints_met,
        "used_defaults":     rec.used_defaults,
        "search_result":     None,
    }

    # Optionally run the recommended algorithm
    if req.problem_config:
        problem = _build_problem(req.env_type, req.problem_config)
        runner  = BenchmarkRunner(timeout_seconds=req.timeout)
        result  = runner.run_single(problem, rec.primary)

        # If primary fails, try fallback
        if not result.found and rec.fallback_order:
            for fallback in rec.fallback_order:
                result = runner.run_single(problem, fallback)
                if result.found:
                    response["recommended_algo"] = fallback
                    response["explanation"] += f"\n\n[Auto-retry] Primary ({rec.primary}) failed. Used fallback: {fallback}"
                    break

        response["search_result"] = _result_to_dict(result, response["recommended_algo"])

    return response


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Results / Charts
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/results/summary", summary="Get benchmark summary as JSON")
def results_summary():
    """Return the summary CSV (if it exists) as a JSON array."""
    path = "results/summary.csv"
    if not os.path.exists(path):
        # Try all_experiments.csv
        path = "results/all_experiments.csv"
    if not os.path.exists(path):
        raise HTTPException(404, "No results found. Run the benchmark suite first.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.get("/results/charts/{chart_id}", summary="Get a chart PNG as base64")
def get_chart(chart_id: str):
    """
    Return a chart PNG as a base64-encoded string.
    chart_id: c1_nodes_expanded | c2_runtime | c3_memory |
              c4_optimality | c5_heuristic_error | c6_frontier_growth | c7_success_heatmap
    """
    path = f"results/charts/{chart_id}.png"
    if not os.path.exists(path):
        raise HTTPException(404, f"Chart '{chart_id}' not found. Run /run/experiment first.")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"chart_id": chart_id, "data": data, "format": "png/base64"}


@app.post("/run/experiment", summary="Run benchmark experiments and generate charts")
def run_experiment(req: ExperimentRequest):
    """
    Run one of the four benchmark experiments (A/B/C/D)
    and generate all 7 charts from the results.
    Returns a summary of results.
    """
    from search_benchmark.benchmarking.experiments import (
        experiment_a_graph_scaling,
        experiment_b_grid_scaling,
        experiment_c_heuristic_stress,
        experiment_d_failure_cases,
    )

    exp_map = {
        "A": experiment_a_graph_scaling,
        "B": experiment_b_grid_scaling,
        "C": experiment_c_heuristic_stress,
        "D": experiment_d_failure_cases,
    }

    runner  = BenchmarkRunner(timeout_seconds=12.0)
    fn      = exp_map[req.experiment]
    df      = fn(runner=runner if req.experiment in ("A","B","C") else None,
                 verbose=req.verbose)
    df["experiment"] = req.experiment

    # Save profiles
    os.makedirs("results", exist_ok=True)
    store = ProfileStore.from_dataframe(df)
    store.save("results/profiles.json")

    # Generate charts
    generate_all_charts(df, chart_dir="results/charts", verbose=False)

    summary = BenchmarkRunner.summarise(df)
    return {
        "experiment": req.experiment,
        "rows":       len(df),
        "charts":     list(range(1, 8)),
        "summary":    summary.to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Utility
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/heuristics", summary="List available heuristics")
def list_heuristics():
    return {"heuristics": list(HEURISTIC_REGISTRY.keys())}


@app.get("/algorithms", summary="List available algorithms")
def list_algorithms():
    return {"algorithms": list(ALGORITHMS.keys())}


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "version": "1.0.0"}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("╔══════════════════════════════════════════════╗")
    print("║  Search Algorithm Benchmarking Tool          ║")
    print("║  http://localhost:8000                       ║")
    print("║  API docs: http://localhost:8000/docs        ║")
    print("╚══════════════════════════════════════════════╝")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)