import heapq
from itertools import count
from typing import Dict, Optional, Sequence, Tuple

from game.board import Board
from game.rules import Action, get_successors, is_goal
from game.state import State
from solvers.base import BaseSolver, CameFrom
from solvers.heuristics import HeuristicFactory, point_object_heuristic
from solvers.ucs import move_cost


class AStarSolver(BaseSolver):
    """A*: mở node có f = g + h nhỏ nhất, g tính theo `move_cost` của UCS."""

    name = "A*"

    def __init__(
        self, heuristic: HeuristicFactory = point_object_heuristic
    ) -> None:
        self.heuristic = heuristic

    def _search(
        self, board: Board
    ) -> Tuple[Optional[Sequence[Action]], int]:
        h = self.heuristic(board)  # Tiền xử lý một lần cho mỗi board
        start = board.initial_state
        came_from: CameFrom = {start: (None, None)}
        best_cost: Dict[State, int] = {start: 0}

        # Cùng f thì ưu tiên h nhỏ
        tie = count()
        h_start = h(start)
        frontier = [(h_start, h_start, next(tie), 0, start)]
        expanded = 0

        while frontier:
            _, _, _, cost, state = heapq.heappop(frontier)
            if cost > best_cost[state]:
                continue 

            if is_goal(board, state):
                return self.reconstruct_path(came_from, state), expanded
            expanded += 1

            for action, child in get_successors(board, state):
                new_cost = cost + move_cost(board, state, action, child)
                if child not in best_cost or new_cost < best_cost[child]:
                    best_cost[child] = new_cost
                    came_from[child] = (state, action)
                    h_child = h(child)
                    heapq.heappush(frontier, (new_cost + h_child, h_child,
                                              next(tie), new_cost, child))

        return None, expanded