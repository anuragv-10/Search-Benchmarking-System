"""
tests/test_core.py
------------------
Unit tests for Section 1: core infrastructure.
Run with:  python -m pytest tests/test_core.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import numpy as np

from search_benchmark.core.problem import Node, Problem
from search_benchmark.core.heuristics import (
    euclidean, manhattan, chebyshev, zero,
    scaled_manhattan, get_heuristic, HEURISTIC_REGISTRY,
)
from search_benchmark.core.graph import (
    WeightedGraph, GraphProblem, generate_random_graph, _bfs_reachable,
)
from search_benchmark.core.grid import (
    Grid, GridProblem, generate_maze, generate_unsolvable_maze,
    _flood_reachable,
)


class TestNode:

    def test_root_node_defaults(self):
        n = Node(state="A")
        assert n.parent is None
        assert n.action is None
        assert n.path_cost == 0.0
        assert n.depth == 0

    def test_path_single_node(self):
        n = Node(state="A")
        assert n.path() == [n]

    def test_path_chain(self):
        a = Node(state="A")
        b = Node(state="B", parent=a, action="go_B", path_cost=1.0, depth=1)
        c = Node(state="C", parent=b, action="go_C", path_cost=3.0, depth=2)
        p = c.path()
        assert len(p) == 3
        assert p[0].state == "A"
        assert p[1].state == "B"
        assert p[2].state == "C"

    def test_solution_actions(self):
        a = Node(state="A")
        b = Node(state="B", parent=a, action="right", path_cost=1, depth=1)
        c = Node(state="C", parent=b, action="down", path_cost=2, depth=2)
        assert c.solution() == ["right", "down"]

    def test_solution_root_is_empty(self):
        a = Node(state="A")
        assert a.solution() == []

    def test_lt_by_path_cost(self):
        cheap = Node(state="X", path_cost=1.0)
        dear = Node(state="Y", path_cost=5.0)
        assert cheap < dear
        assert not dear < cheap

    def test_lt_tie_break_by_depth(self):
        n1 = Node(state="X", path_cost=3.0, depth=1)
        n2 = Node(state="Y", path_cost=3.0, depth=4)
        assert n1 < n2

    def test_equality_on_state(self):
        n1 = Node(state=5, path_cost=1.0)
        n2 = Node(state=5, path_cost=99.0)
        assert n1 == n2

    def test_hash_usable_in_set(self):
        nodes = {Node(state=i) for i in range(10)}
        assert len(nodes) == 10

    def test_repr_contains_state(self):
        n = Node(state=42, path_cost=3.14)
        assert "42" in repr(n)


class _SimpleProblem(Problem):
    def actions(self, s): return [s + 1, s - 1]
    def result(self, s, a): return a
    def step_cost(self, s, a, ns): return 1.0
    def h(self, node): return abs(node.state - self.goal_state)


class TestProblem:

    def _make(self):
        return _SimpleProblem(initial_state=0, goal_state=5)

    def test_instantiation(self):
        p = self._make()
        assert p.initial_state == 0
        assert p.goal_state == 5

    def test_goal_test_true(self):
        p = self._make()
        assert p.goal_test(5)

    def test_goal_test_false(self):
        p = self._make()
        assert not p.goal_test(3)

    def test_root_node(self):
        p = self._make()
        root = p.root_node()
        assert root.state == 0
        assert root.path_cost == 0.0
        assert root.depth == 0

    def test_expand_creates_children(self):
        p = self._make()
        root = p.root_node()
        children = p.expand(root)
        states = {c.state for c in children}
        assert 1 in states
        assert -1 in states

    def test_expand_sets_cost(self):
        p = self._make()
        root = p.root_node()
        children = p.expand(root)
        for c in children:
            assert c.path_cost == 1.0

    def test_expand_sets_depth(self):
        p = self._make()
        root = p.root_node()
        children = p.expand(root)
        for c in children:
            assert c.depth == 1
            assert c.parent is root


class TestHeuristics:

    def test_euclidean_same_point(self):
        assert euclidean((0, 0), (0, 0)) == 0.0

    def test_euclidean_known(self):
        assert abs(euclidean((0, 0), (3, 4)) - 5.0) < 1e-9

    def test_manhattan_same_point(self):
        assert manhattan((2, 3), (2, 3)) == 0.0

    def test_manhattan_known(self):
        assert manhattan((0, 0), (3, 4)) == 7.0

    def test_chebyshev_known(self):
        assert chebyshev((0, 0), (3, 4)) == 4.0

    def test_zero_always_zero(self):
        for a, b in [((0, 0), (5, 5)), ((1, 2), (9, 8)), ((0, 0), (0, 0))]:
            assert zero(a, b) == 0.0

    def test_scaled_manhattan_doubles(self):
        h2 = scaled_manhattan(2.0)
        assert h2((0, 0), (3, 4)) == pytest.approx(14.0)

    def test_get_heuristic_valid(self):
        fn = get_heuristic("manhattan")
        assert fn((0, 0), (3, 4)) == 7.0

    def test_get_heuristic_invalid(self):
        with pytest.raises(KeyError):
            get_heuristic("does_not_exist")

    def test_all_registry_keys_callable(self):
        for name, fn in HEURISTIC_REGISTRY.items():
            result = fn((0, 0), (1, 1))
            assert isinstance(result, float), f"{name} did not return float"

    def test_admissibility_euclidean_vs_manhattan(self):
        for a in [(0, 0), (1, 1), (5, 3)]:
            for b in [(3, 4), (2, 7), (9, 1)]:
                assert euclidean(a, b) <= manhattan(a, b) + 1e-9


class TestWeightedGraph:

    def _five_node_graph(self) -> WeightedGraph:
        positions = {0: (0, 1), 1: (1, 1), 2: (0.5, 0.5), 3: (0, 0), 4: (1, 0)}
        g = WeightedGraph(n_nodes=5, positions=positions)
        g.add_edge(0, 1, 2.0)
        g.add_edge(0, 3, 3.0)
        g.add_edge(1, 4, 4.0)
        g.add_edge(3, 4, 5.0)
        return g

    def test_neighbour_count(self):
        g = self._five_node_graph()
        assert len(g.neighbours(0)) == 2
        assert len(g.neighbours(2)) == 0

    def test_neighbour_weights(self):
        g = self._five_node_graph()
        weights = {v: w for v, w in g.neighbours(0)}
        assert weights[1] == 2.0
        assert weights[3] == 3.0

    def test_undirected(self):
        g = self._five_node_graph()
        assert any(v == 0 for v, _ in g.neighbours(1))

    def test_positions_stored(self):
        g = self._five_node_graph()
        assert g.position(0) == (0, 1)

    def test_euclidean_distance(self):
        g = self._five_node_graph()
        d = g.distance(0, 1)
        assert d == pytest.approx(euclidean((0, 1), (1, 1)))

    def test_to_from_dict_roundtrip(self):
        g = self._five_node_graph()
        d = g.to_dict()
        g2 = WeightedGraph.from_dict(d)
        assert g2.n_nodes == g.n_nodes
        assert set(g2.neighbours(0)) == set(g.neighbours(0))


class TestGraphProblem:

    def _problem(self) -> GraphProblem:
        positions = {i: (float(i), 0.0) for i in range(5)}
        g = WeightedGraph(n_nodes=5, positions=positions)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(2, 3, 1.0)
        g.add_edge(3, 4, 1.0)
        return GraphProblem(g, initial_state=0, goal_state=4)

    def test_goal_test(self):
        p = self._problem()
        assert p.goal_test(4)
        assert not p.goal_test(2)

    def test_actions_returns_neighbours(self):
        p = self._problem()
        actions = p.actions(0)
        nbr_ids = [a[0] for a in actions]
        assert 1 in nbr_ids

    def test_result(self):
        p = self._problem()
        action = (1, 1.0)
        assert p.result(0, action) == 1

    def test_step_cost_from_action(self):
        p = self._problem()
        action = (1, 2.5)
        assert p.step_cost(0, action, 1) == 2.5

    def test_h_euclidean(self):
        p = self._problem()
        node = Node(state=0)
        assert p.h(node) == pytest.approx(4.0)

    def test_h_scaled_inadmissible(self):
        p = self._problem()
        p.heuristic_scale = 2.0
        node = Node(state=0)
        assert p.h(node) == pytest.approx(8.0)

    def test_serialisation_roundtrip(self):
        p = self._problem()
        d = p.to_dict()
        p2 = GraphProblem.from_dict(d)
        assert p2.initial_state == p.initial_state
        assert p2.goal_state == p.goal_state


class TestGenerateRandomGraph:

    def test_deterministic(self):
        g1, s1, e1 = generate_random_graph(20, seed=42)
        g2, s2, e2 = generate_random_graph(20, seed=42)
        assert s1 == s2 == 0
        assert e1 == e2 == 19
        assert g1.to_dict()["edges"] == g2.to_dict()["edges"]

    def test_different_seeds_differ(self):
        g1, _, _ = generate_random_graph(20, seed=1)
        g2, _, _ = generate_random_graph(20, seed=2)
        assert g1.to_dict()["edges"] != g2.to_dict()["edges"]

    def test_correct_node_count(self):
        g, _, _ = generate_random_graph(50, seed=7)
        assert g.n_nodes == 50

    def test_path_guaranteed(self):
        for seed in range(10):
            g, start, goal = generate_random_graph(30, edge_density=0.1, seed=seed)
            assert _bfs_reachable(g, start, goal), (
                f"Seed {seed}: no path from {start} to {goal}"
            )

    def test_start_is_0_goal_is_last(self):
        _, s, g = generate_random_graph(15, seed=99)
        assert s == 0
        assert g == 14


class TestGrid:

    def _small_grid(self) -> Grid:
        data = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.uint8)
        return Grid(data)

    def test_shape(self):
        g = self._small_grid()
        assert g.rows == 3
        assert g.cols == 3

    def test_is_free_open_cell(self):
        g = self._small_grid()
        assert g.is_free(0, 0)

    def test_is_free_wall(self):
        g = self._small_grid()
        assert not g.is_free(1, 1)

    def test_is_free_out_of_bounds(self):
        g = self._small_grid()
        assert not g.is_free(-1, 0)
        assert not g.is_free(0, 10)

    def test_in_bounds(self):
        g = self._small_grid()
        assert g.in_bounds(2, 2)
        assert not g.in_bounds(3, 0)

    def test_to_from_dict(self):
        g = self._small_grid()
        d = g.to_dict()
        g2 = Grid.from_dict(d)
        assert np.array_equal(g.data, g2.data)


class TestGridProblem:

    def _open_grid_problem(self) -> GridProblem:
        data = np.zeros((5, 5), dtype=np.uint8)
        return GridProblem(Grid(data), initial_state=(0, 0), goal_state=(4, 4))

    def test_goal_test(self):
        p = self._open_grid_problem()
        assert p.goal_test((4, 4))
        assert not p.goal_test((0, 0))

    def test_actions_cardinal_center(self):
        p = self._open_grid_problem()
        aa = p.actions((2, 2))
        assert len(aa) == 4

    def test_actions_corner(self):
        p = self._open_grid_problem()
        aa = p.actions((0, 0))
        assert len(aa) == 2

    def test_result_move_down(self):
        p = self._open_grid_problem()
        ns = p.result((0, 0), (1, 0))
        assert ns == (1, 0)

    def test_step_cost_cardinal(self):
        p = self._open_grid_problem()
        assert p.step_cost((0, 0), (1, 0), (1, 0)) == 1.0

    def test_step_cost_diagonal(self):
        p = GridProblem(
            Grid(np.zeros((5, 5), dtype=np.uint8)),
            (0, 0), (4, 4),
            eight_directional=True,
        )
        assert p.step_cost((0, 0), (1, 1), (1, 1)) == pytest.approx(1.414)

    def test_h_manhattan(self):
        p = self._open_grid_problem()
        node = Node(state=(0, 0))
        assert p.h(node) == pytest.approx(8.0)

    def test_h_euclidean(self):
        p = GridProblem(
            Grid(np.zeros((5, 5), dtype=np.uint8)),
            (0, 0), (4, 4),
            heuristic_name="euclidean",
        )
        node = Node(state=(0, 0))
        assert p.h(node) == pytest.approx(math.sqrt(32))

    def test_h_chebyshev(self):
        p = GridProblem(
            Grid(np.zeros((5, 5), dtype=np.uint8)),
            (0, 0), (4, 4),
            heuristic_name="chebyshev",
        )
        node = Node(state=(0, 0))
        assert p.h(node) == pytest.approx(4.0)

    def test_h_scaled_inadmissible(self):
        p = GridProblem(
            Grid(np.zeros((5, 5), dtype=np.uint8)),
            (0, 0), (4, 4),
            heuristic_scale=2.0,
        )
        node = Node(state=(0, 0))
        assert p.h(node) == pytest.approx(16.0)

    def test_invalid_heuristic_raises(self):
        with pytest.raises(ValueError):
            GridProblem(
                Grid(np.zeros((5, 5), dtype=np.uint8)),
                (0, 0), (4, 4),
                heuristic_name="bogus",
            )

    def test_serialisation_roundtrip(self):
        p = self._open_grid_problem()
        d = p.to_dict()
        p2 = GridProblem.from_dict(d)
        assert p2.initial_state == p.initial_state
        assert p2.goal_state == p.goal_state
        assert p2.heuristic_name == p.heuristic_name


class TestGenerateMaze:

    def test_deterministic(self):
        _, g1 = generate_maze(10, 10, seed=42)
        _, g2 = generate_maze(10, 10, seed=42)
        assert np.array_equal(g1.data, g2.data)

    def test_different_seeds_differ(self):
        _, g1 = generate_maze(10, 10, seed=1)
        _, g2 = generate_maze(10, 10, seed=2)
        assert not np.array_equal(g1.data, g2.data)

    def test_correct_shape(self):
        _, g = generate_maze(15, 20, seed=7)
        assert g.rows == 15
        assert g.cols == 20

    def test_start_goal_are_free(self):
        _, g = generate_maze(10, 10, seed=42)
        assert g.is_free(0, 0)
        assert g.is_free(9, 9)

    def test_solvable_across_seeds(self):
        for seed in range(15):
            _, g = generate_maze(10, 10, wall_density=0.3, seed=seed)
            assert _flood_reachable(g.data, (0, 0), (9, 9), 10, 10), (
                f"Seed {seed}: maze is unsolvable"
            )

    def test_unsolvable_maze_blocks_goal(self):
        _, g = generate_unsolvable_maze(rows=10, cols=10)
        assert not _flood_reachable(g.data, (0, 0), (9, 9), 10, 10)

    @pytest.mark.parametrize("density", [0.1, 0.3, 0.5])
    def test_various_densities_solvable(self, density):
        for seed in range(5):
            _, g = generate_maze(12, 12, wall_density=density, seed=seed)
            assert _flood_reachable(g.data, (0, 0), (11, 11), 12, 12)


if __name__ == "__main__":
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )

