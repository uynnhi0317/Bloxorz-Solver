from typing import Iterator, List, Optional, Sequence, Set, Tuple

from game import Board, Move, State, get_successors, is_goal
from solvers.base import BaseSolver

Successors = Iterator[Tuple[Move, State]]


class DFSSolver(BaseSolver):
    name = "DFS"

    def _search(self, board: Board) -> Tuple[Optional[Sequence[Move]], int]:
        start = board.initial_state
        visited: Set[State] = {start}
        if is_goal(board, start):
            return (), 0

        expanded = 1
        stack: List[Successors] = [iter(get_successors(board, start))]
        moves: List[Move] = []

        while stack:
            for move, next_state in stack[-1]:
                if next_state in visited:
                    continue
                visited.add(next_state)
                moves.append(move)
                if is_goal(board, next_state):
                    return tuple(moves), expanded
                expanded += 1
                stack.append(iter(get_successors(board, next_state)))
                break
            else:
                stack.pop()
                if moves:
                    moves.pop()

        return None, expanded