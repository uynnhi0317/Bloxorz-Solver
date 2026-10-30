"""Khởi chạy chương trình: python main.py (đứng trong thư mục Source/)."""
import sys
import threading

from gui.app import App
from solvers.bfs import BFSSolver
from solvers.dfs import DFSSolver
from solvers.ucs import UCSSolver
from solvers.astar import AStarSolver


class StartedBoard:
    """Board giống hệt board gốc, chỉ khác initial_state = state người chơi đang đứng."""

    def __init__(self, board, start):
        self._board = board
        self.initial_state = start

    def __getattr__(self, name):  # chỉ được gọi khi không tìm thấy thuộc tính trên proxy
        return getattr(self._board, name)


def main() -> None:
    app = App()
    for solver in (BFSSolver(), DFSSolver(), UCSSolver(), AStarSolver()):
        app.register_solver(
            solver.name,  # phải khớp key nút: "BFS", "DFS", "UCS", "A*"
            lambda b, s, sv=solver: sv.solve(StartedBoard(b, s)),
        )
    app.run()


if __name__ == "__main__":
    main()