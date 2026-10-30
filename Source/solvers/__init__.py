"""Search algorithms package for Bloxorz."""

from .base import BaseSolver, SolveResult

from .bfs import BFSSolver
from .dfs import DFSSolver
from .ucs import UCSSolver
from .astar import AStarSolver

__all__ = [
    "BaseSolver",
    "SolveResult",
    "BFSSolver",
    "DFSSolver",
    "UCSSolver",
    "AStarSolver",
]