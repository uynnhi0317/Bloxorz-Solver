from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
import time
import tracemalloc
from typing import Dict, Optional, Sequence, Tuple
from game.board import Board
from game.rules import Action, get_next_state, is_goal
from game.state import State

CameFrom = Dict[State, Tuple[Optional[State], Optional[Action]]]

@dataclass(frozen = True, slots = True)
class SolveResult:
    """
        Lưu trữ kết quả giải và các chỉ số hiệu năng của thuật toán:
        + moves: chuỗi hướng di chuyển từ start đến goal (None nếu không tìm ra lời giải)
        + time: thời gian thực thi thuật toán (tính bằng giây)
        + peak_memory: đỉnh bộ nhớ tiêu thụ thêm trong quá trình giải (tính bằng bytes)
        + expanded_nodes: tổng số trạng thái đã được mở rộng
    """
    moves: Optional[Tuple[Action, ...]]
    time: float
    peak_memory: int
    expanded_nodes: int

    def __post_init__(self) -> None:
        """Chuẩn hóa và kiểm tra dữ liệu đầu vào"""
        if self.moves is not None:
            object.__setattr__(self, "moves", tuple(Action(m) for m in self.moves))
        
        if self.time < 0:
            raise ValueError("SolveResult: Thời gian chạy không được âm")
        if self.peak_memory < 0:
            raise ValueError("SolveResult: Bộ nhớ tiêu thụ không được âm")
        if self.expanded_nodes < 0:
            raise ValueError("SolveResult: Số node mở rộng không được âm")

    @property
    def is_solved(self) -> bool:
        """Kiểm tra thuật toán có tìm được lời giải hay không"""
        return self.moves is not None

    @property
    def solution_length(self) -> int:
        """
            Độ dài lời giải (số bước di chuyển).
            Trả về 0 nếu không tìm được lời giải hoặc trạng thái đầu đã là đích.
        """
        return len(self.moves) if self.moves is not None else 0


class BaseSolver(ABC):
    """Lớp cơ sở trừu tượng (Interface) cho mọi thuật toán giải Bloxorz (BFS, DFS, UCS, A*)."""
    name: str = "BaseSolver"

    @abstractmethod
    def _search(self, board: Board) -> Tuple[Optional[Sequence[Action]], int]:
        """
            Phương thức trừu tượng thực thi thuật toán tìm kiếm cốt lõi.
            Các lớp con kế thừa bắt buộc phải cài đặt phương thức này.

            Trả về:
            + Tuple[0]: Chuỗi các hướng di chuyển dẫn tới goal (hoặc None nếu vô nghiệm)
            + Tuple[1]: Số lượng trạng thái đã mở rộng (expanded_nodes)
        """
        raise NotImplementedError

    def solve(self, board: Board) -> SolveResult:
        """
            Hàm giao tiếp chính: Chạy thuật toán _search(board) và tự động đo
            thời gian chạy (time) cùng đỉnh bộ nhớ (peak_memory).
        """
        # Dọn bộ nhớ rác trước khi đo để tránh nhiễu giữa các lần chạy liên tiếp
        gc.collect()

        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            moves, expanded_nodes = self._search(board)
            _, peak_memory = tracemalloc.get_traced_memory()
        finally:
            elapsed_time = time.perf_counter() - start_time
            tracemalloc.stop()

        normalized_moves = tuple(moves) if moves is not None else None
        return SolveResult(
            moves = normalized_moves,
            time = elapsed_time,
            peak_memory = peak_memory,
            expanded_nodes = expanded_nodes,
        )

    @staticmethod
    def reconstruct_path(
        came_from: CameFrom,
        goal_state: State,
    ) -> Tuple[Action, ...]:
        """
            Hàm tiện ích: Truy vết lại chuỗi hướng đi từ trạng thái đích về trạng thái đầu.
            + came_from: dict ánh xạ state_con -> (state_cha, hướng_đi_tạo_ra_state_con)
            + goal_state: trạng thái đích khi kết thúc tìm kiếm
        """
        moves = []
        current: Optional[State] = goal_state

        while current is not None:
            parent_info = came_from.get(current)
            if parent_info is None:
                break
            parent_state, direction = parent_info
            if direction is not None:
                moves.append(direction)
            current = parent_state

        moves.reverse()
        return tuple(moves)

    @staticmethod
    def verify_solution(board: Board, moves: Sequence[Action]) -> bool:
        """
            Hàm tiện ích: Kiểm tra một chuỗi bước đi có hợp lệ và dẫn tới đúng ô đích không.
            Hữu ích cho cả unit test (test_solvers.py) và kiểm tra chéo khi chạy benchmark.
        """
        state = board.initial_state
        for direction in moves:
            next_state = get_next_state(board, state, direction)
            if next_state is None:
                return False
            state = next_state
        return is_goal(board, state)