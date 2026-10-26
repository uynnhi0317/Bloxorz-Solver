from collections import deque
from typing import Optional, Sequence, Tuple

from game.board import Board
from game.rules import Action, get_successors, is_goal
from solvers.base import BaseSolver, CameFrom

class BFSSolver(BaseSolver):
    name = "BFS"

    def _search(self, board: Board) -> Tuple[Optional[Sequence[Action]], int]:
        start = board.initial_state
        came_from: CameFrom = {start: (None, None)}  # Đồng thời là visited set
        if is_goal(board, start):
            return (), 0

        frontier = deque([start])
        expanded = 0

        while frontier:
            state = frontier.popleft()
            expanded += 1

            for action, child in get_successors(board, state):
                if child in came_from:
                    continue
                came_from[child] = (state, action)

                if is_goal(board, child):
                    return self.reconstruct_path(came_from, child), expanded
                frontier.append(child)

        return None, expanded