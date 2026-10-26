from typing import Dict, List, Optional, Sequence, Tuple

from game.board import Board
from game.rules import Action, get_successors, is_goal
from game.state import State
from .base import BaseSolver

class IDSSolver(BaseSolver):
    """
        Giải màn chơi bằng thuật toán IDS (Iterative Deepening Search)
        Về độ sâu thì không giới hạn, độ sâu được tăng dần cho tới khi
        tìm được lời giải hoặc màn chơi không giải được
    """

    name = "IDS"

    def _search(self, board: Board) -> Tuple[Optional[Sequence[Action]], int]:
        start = board.initial_state
        expanded_nodes = 0
        depth_limit = 0

        # Biến lưu số trạng thái ở depth limit trước
        prev_size = 0 

        while True:
            # Độ sâu nhỏ nhất đã duyệt tới mỗi trạng thái, reset mỗi vòng IDS
            visited_depth: Dict[State, int] = {start: 0}
            moves: List[Action] = []

            found, expanded = self._depth_limited_search(board, start, 0, depth_limit, visited_depth, moves)
            expanded_nodes += expanded
            if found:
                return tuple(moves), expanded_nodes

            # Trường hợp đi vào sâu hơn nhưng không có trạng thái mới, vô nghiệm
            if len(visited_depth) == prev_size:
                return None, expanded_nodes

            prev_size = len(visited_depth)

            # Tăng độ sâu cho IDS nếu vẫn còn tiềm năng tìm thấy đáp án
            depth_limit += 1

    def _depth_limited_search(
        self,
        board: Board,
        state: State,
        depth: int,
        depth_limit: int,
        visited_depth: Dict[State, int],
        moves: List[Action],
    ) -> Tuple[bool, int]:
        """
        Tham số:
            + state: trạng thái hiện tại
            + depth: độ sâu tại trạng thái hiện tại
            + depth_limit: giới hạn độ sâu của lượt tìm kiếm này
            + visited_depth: độ sâu nhỏ nhất đã duyệt tới mỗi trạng thái
            + moves: dãy hành động trên đường đi hiện tại (được sửa trực tiếp)

        Trả về:
            + found: True nếu tìm thấy lời giải, khi đó moves là lời giải
            + expanded: số nút đã mở rộng trong nhánh này
        """
        if is_goal(board, state):
            return True, 0

        if depth >= depth_limit:
            return False, 0

        expanded = 1

        for action, next_state in get_successors(board, state):
            # Đã duyệt tới trạng thái này ở độ sâu nhỏ hơn hoặc bằng độ sâu hiện tại thì bỏ qua
            if visited_depth.get(next_state, depth + 2) <= depth + 1:
                continue

            visited_depth[next_state] = depth + 1
            moves.append(action)

            found, branch_expanded = self._depth_limited_search(board, next_state, depth + 1, depth_limit, visited_depth, moves)
            expanded += branch_expanded
            if found:
                return True, expanded
            moves.pop()

        return False, expanded