from collections import deque
from typing import Dict, Optional, Sequence, Tuple

from game.board import Board
from game.rules import Action, get_successors, is_goal
from game.state import State
from solvers.base import BaseSolver, CameFrom


class BFSSolver(BaseSolver):
    name = "BFS"

    def _search(self, board: Board) -> Tuple[Optional[Sequence[Action]], int]:
        start = board.initial_state
        came_from: CameFrom = {start: (None, None)}
        rolls: Dict[State, int] = {start: 0}  # Số cú lăn ít nhất tới mỗi state

        frontier = deque([(0, start)])
        expanded = 0

        while frontier:
            depth, state = frontier.popleft()
            if depth > rolls[state]:
                continue  # Bản ghi cũ, state đã được tới bằng ít cú lăn hơn

            if is_goal(board, state):
                return self.reconstruct_path(came_from, state), expanded
            expanded += 1

            for action, child in get_successors(board, state):
                # Đổi cube không phải cú lăn: ở lại cùng tầng (đầu hàng đợi)
                step = 0 if action == Action.SWITCH_CONTROL else 1
                new_depth = depth + step
                if child in rolls and rolls[child] <= new_depth:
                    continue
                rolls[child] = new_depth
                came_from[child] = (state, action)
                if step == 0:
                    frontier.appendleft((new_depth, child))
                else:
                    frontier.append((new_depth, child))

        return None, expanded
