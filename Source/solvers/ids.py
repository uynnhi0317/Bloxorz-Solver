from typing import Dict, Optional, Set, Tuple, Sequence

from game.board import Board
from game.rules import Action, get_successors, is_goal
from game.state import State
from .base import BaseSolver

# came_from[state] = (state cha, hướng đi từ cha); state đầu ánh xạ về (None, None)
CameFrom = Dict[State, Tuple[Optional[State], Optional[Action]]]

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

        while True:
            came_from: CameFrom = { start: (None, None) }
            path: Set[State] = {start}

            def depth_limited_search(state: State, depth: int) -> Tuple[bool, bool]:
                """
                    Hàm thực thi tìm kiếm theo chiều sâu bị giới hạn
                    Tham số:
                        + state: trạng thái hiện tại
                        + depth: độ sâu tại trạng thái hiện tại

                    Trả về:
                        + found: True nếu trạng thái state là lời giải
                        + cutoff: True nếu ở trạng thái hiện tại có thể đi sâu thêm được nữa để tìm đáp án
                """
                nonlocal expanded_nodes
                if is_goal(board, state):
                    return True, False

                expanded_nodes += 1
                successors = get_successors(board, state)
                if depth >= depth_limit:
                    # Kiểm tra nhánh này còn có thể đi sâu hơn được không
                    cutoff = any(next_state not in path for _, next_state in successors)
                    return False, cutoff

                cutoff = False
                for action, next_state in successors:
                    # Bỏ qua trạng thái đã tồn tại trong đường đi
                    if next_state in path:
                        continue
                    came_from[next_state] = (state, action)
                    path.add(next_state)

                    # Thực hiện tìm kết quả ở độ sâu tiếp theo
                    found, branch_cutoff = depth_limited_search(next_state, depth + 1)
                    if found:
                        return True, False
                    
                    path.remove(next_state)
                    came_from.pop(next_state, None)
                    cutoff = cutoff or branch_cutoff

                return False, cutoff

            found, cutoff = depth_limited_search(start, 0)
            if found:
                goal_state = next(state for state in path if is_goal(board, state))
                return self.reconstruct_path(came_from, goal_state), expanded_nodes
            if not cutoff:
                return None, expanded_nodes

            # Tăng độ sâu cho IDS nếu vẫn còn tiềm năng tìm thấy đáp án
            depth_limit += 1