"""
core/problem.py
---------------
Foundational abstractions shared by all search algorithms.

Classes
-------
Node        — a search-tree node (state wrapper with bookkeeping)
Problem     — abstract base class every concrete problem must subclass
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    """
    A single node in the search tree.

    Attributes
    ----------
    state       : The problem state this node represents.
    parent      : The Node that generated this one (None for the root).
    action      : The action applied to parent.state to reach this state.
    path_cost   : g(n) — cumulative cost from the root to this node.
    depth       : Number of edges from the root to this node.
    """

    state: Any
    parent: Optional["Node"] = field(default=None, repr=False)
    action: Any = field(default=None)
    path_cost: float = field(default=0.0)
    depth: int = field(default=0)

    # ------------------------------------------------------------------
    # Comparison — needed so nodes can sit in a heapq.
    # Primary key is path_cost; tie-break by depth (prefer shallower).
    # ------------------------------------------------------------------
    def __lt__(self, other: "Node") -> bool:
        if self.path_cost != other.path_cost:
            return self.path_cost < other.path_cost
        return self.depth < other.depth

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.state == other.state

    def __hash__(self) -> int:
        # Hash on state so Node can be stored in sets/dicts keyed by state.
        return hash(self.state)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def path(self) -> List["Node"]:
        """Return the list of nodes on the path from the root to this node."""
        node, path = self, []
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def solution(self) -> List[Any]:
        """Return the list of actions from the root to this node (excludes root action)."""
        return [node.action for node in self.path()[1:]]

    def __repr__(self) -> str:
        return (
            f"Node(state={self.state!r}, "
            f"g={self.path_cost:.3f}, "
            f"depth={self.depth})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Problem
# ─────────────────────────────────────────────────────────────────────────────

class Problem(ABC):
    """
    Abstract base class for a search problem.

    All concrete problem types (GraphProblem, GridProblem, …) must subclass
    this and implement every @abstractmethod.  Algorithms receive only a
    Problem instance — they never touch raw environment data directly.

    Parameters
    ----------
    initial_state : The starting state.
    goal_state    : The target state (single goal; override goal_test for multi-goal).
    """

    def __init__(self, initial_state: Any, goal_state: Any) -> None:
        self.initial_state = initial_state
        self.goal_state = goal_state

    # ------------------------------------------------------------------
    # Must-override interface
    # ------------------------------------------------------------------

    @abstractmethod
    def actions(self, state: Any) -> List[Any]:
        """Return the list of actions available in *state*."""

    @abstractmethod
    def result(self, state: Any, action: Any) -> Any:
        """Return the state reached by applying *action* in *state*."""

    @abstractmethod
    def step_cost(self, state: Any, action: Any, next_state: Any) -> float:
        """Return the cost of taking *action* from *state* to *next_state*."""

    @abstractmethod
    def h(self, node: Node) -> float:
        """
        Heuristic estimate of the cost from *node* to the goal.
        Must be non-negative.  Should be admissible (h(n) ≤ h*(n)) for A*
        optimality guarantees.
        """

    # ------------------------------------------------------------------
    # May-override helpers (defaults provided)
    # ------------------------------------------------------------------

    def goal_test(self, state: Any) -> bool:
        """Return True if *state* is a goal state."""
        return state == self.goal_state

    def value(self, node: Node) -> float:
        """
        Objective value for local-search variants.
        Default: negative path cost (higher = better).
        """
        return -node.path_cost

    # ------------------------------------------------------------------
    # Node factory — used by algorithms to expand a node
    # ------------------------------------------------------------------

    def expand(self, node: Node) -> List[Node]:
        """
        Generate all child nodes reachable from *node*.

        Returns a list of Node objects, one per available action.
        Algorithms call this method; they never call actions/result/step_cost
        directly.
        """
        children = []
        for action in self.actions(node.state):
            next_state = self.result(node.state, action)
            cost = node.path_cost + self.step_cost(node.state, action, next_state)
            child = Node(
                state=next_state,
                parent=node,
                action=action,
                path_cost=cost,
                depth=node.depth + 1,
            )
            children.append(child)
        return children

    def root_node(self) -> Node:
        """Convenience: create the root Node from initial_state."""
        return Node(state=self.initial_state)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"initial={self.initial_state!r}, "
            f"goal={self.goal_state!r})"
        )

