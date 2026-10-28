from typing import List, Optional, Sequence, Set, Tuple

from game import *
from solvers.base import BaseSolver


class DFSSolver(BaseSolver):
    name = "DFS"

    def _search(self, board: Board) -> Tuple[Optional[Sequence[Action]], int]:
        start = board.initial_state
        visited: Set[State] = set()
        moves: List[Action] = []

        found, expanded = self.visit(board, start, visited, moves)
        if found:
            return tuple(moves), expanded
        return None, expanded

    def visit(self, board: Board, state: State, visited: Set[State], moves: List[Action],) -> Tuple[bool, int]:
        """
        Tìm kiếm theo chiều sâu DFS từ một trạng thái

        Tham số:
            board: Bàn chơi cần giải
            state: Trạng thái hiện tại đang được duyệt
            visited: Tập các trạng thái đã được thăm
            moves: Danh sách các hành động trên đường đi hiện tại

        Trả về:
            Tuple gồm cờ cho biết đã tìm thấy đích hay chưa và
            số trạng thái đã mở rộng trong nhánh tìm kiếm này
        """
        visited.add(state)

        if is_goal(board, state):
            return True, 0

        expanded = 1

        # Duyệt qua các successor để đệ quy
        for action, next_state in get_successors(board, state):
            if next_state in visited:
                continue

            moves.append(action)
            found, child_expanded = self.visit(board, next_state, visited, moves)
            expanded += child_expanded

            if found:
                return True, expanded

            # Nhánh này không tới đích, quay lui bằng cách bỏ nước đi vừa thêm
            moves.pop()

        # Không tìm thấy đáp án từ trạng thái này
        return False, expanded