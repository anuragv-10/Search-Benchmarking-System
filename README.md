# Search Algorithm Benchmarking Tool

Welcome to the Search Algorithm Benchmarking Tool.

This isn't just a toy implementation of A*. It's a full-stack, highly instrumented testbed for classical search algorithms. It lets you run head-to-head performance benchmarks, evaluate the mathematical properties of custom heuristics, and even uses a meta-agent to dynamically select the optimal search strategy based on real-world constraints like memory limits and required optimality.

## What's Under the Hood?

### 1. The Core Environments
The project supports two distinct, dynamically generated problem spaces:
- **Weighted Graphs:** We generate random graphs with configurable node counts and edge densities, ensuring connectivity and realistic weight distributions. 
- **Grid Mazes:** Built on top of NumPy arrays for performance, the grid environment supports adjustable wall densities, 4-way (cardinal) or 8-way (diagonal) movement, and guaranteed solvability checks during generation.

### 2. The Search Algorithms
We've implemented the five pillars of classical search:
- **Breadth-First Search (BFS)**
- **Depth-First Search (DFS)**
- **Uniform-Cost Search (UCS)**
- **Greedy Best-First Search (GBFS)**
- **A* Search**

*Technical note on the implementations:* The algorithms are heavily instrumented. The frontier is managed using Python's `heapq` (keyed on `f(n) = g(n) + h(n)` for A*). To handle the priority queue efficiently, we use a lazy-deletion pattern. More importantly, the stats engine tracks deep metrics: nodes generated vs. expanded, peak memory usage, runtime, and **re-expansions** (which only occur when testing inconsistent heuristics). 

We even do post-solve heuristic error calculation by running a mini-UCS backward from nodes on the solution path to compute the true `h*(n)` and report the mean absolute error `|h(n) - h*(n)|`.

### 3. Strategy Selector Agent
One of the coolest features is the `StrategySelector` agent in `agent/selector.py`. Instead of guessing which algorithm to use, the agent uses a three-step hybrid pipeline:
1. **Hard Constraint Filter:** Immediately drops algorithms that violate user constraints (e.g., filtering out GBFS if optimality is strictly required, or dropping A* if its mean profile memory exceeds a strict KB limit).
2. **Weighted Scoring:** Ranks the survivors based on normalized, user-weighted preferences for speed, memory, and path quality. It uses empirical profile data generated from past benchmark runs to make informed decisions.
3. **Recommendation & Fallback:** Returns a primary recommendation, a plain-English explanation of *why* it was chosen, and a ranked list of fallbacks just in case the primary fails at runtime.

### 4. Benchmarking & Heuristic Analysis
The tool allows you to run massive experiment suites across thousands of problem instances. It generates raw data summaries (CSV) and a suite of 7 detailed Matplotlib charts covering:
- Runtime vs. Problem Size
- Peak Memory Scaling
- Path Cost Optimality
- Frontier Growth Dynamics

There's also a dedicated heuristic analysis pipeline that explicitly checks custom heuristics for **admissibility** (never overestimates) and **consistency** (satisfies the triangle inequality), giving you an exact percentage rate of compliance over the state space.

## Technology Stack

- **Backend Logic & API:** Python 3.10+, FastAPI, Uvicorn
- **Data & Math:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Frontend:** Pure HTML/Vanilla JS (keeping it fast and dependency-free, interacting with the backend via REST)

## Getting Started

To spin this up locally, you'll need Python installed.

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Saimanish123/search_benchmark.git
   cd search_benchmark
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install fastapi uvicorn numpy pandas matplotlib
   ```

4. **Fire up the backend:**
   ```bash
   python api.py
   ```
   *The FastAPI server will boot up on `http://localhost:8000`.*

5. **Access the UI & Docs:**
   - Head over to `http://localhost:8000` to see the web interface.
   - For the interactive API documentation (Swagger UI), go to `http://localhost:8000/docs`.

## Project Structure

A quick map of the repository to help you navigate the source:

- `/api.py` — The FastAPI application binding everything together.
- `/index.html` — The frontend interface.
- `/search_benchmark/algorithms/` — The actual search implementations (`astar.py`, `bfs.py`, etc.) and the stats tracking utilities.
- `/search_benchmark/core/` — Environment definitions (`graph.py`, `grid.py`, `problem.py`) and the heuristic functions.
- `/search_benchmark/agent/` — The empirical `ProfileStore` and the `StrategySelector` logic.
- `/search_benchmark/benchmarking/` — The `BenchmarkRunner` and experiment definitions.
- `/search_benchmark/analysis/` — Reporting tools and Matplotlib chart generators.
- `/tests/` — Pytest suite covering all the core logic.

Feel free to poke around, run the experiment suite, and try writing a custom heuristic to see if you can beat Manhattan distance on the grids!
