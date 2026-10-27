import heapq
from itertools import count
from typing import Dict, Optional, Sequence, Tuple

from game.board import Board
from game.rules import Action, get_next_state, get_successors, is_goal
from game.state import State
from game.tile import TileType
from solvers.base import BaseSolver, CameFrom

NORMAL_COST = 2    # Lăn thông thường
SWITCH_COST = 1    # Kích switch làm mở ít nhất một cầu
FRAGILE_COST = 4   # Block đè lên ô fragile
CONTROL_COST = 0   # Đổi cube không phải cú lăn nên không tính chi phí
MIN_ROLL_COST = min(NORMAL_COST, SWITCH_COST, FRAGILE_COST)


def move_cost(board: Board, parent: State, action: Action, child: State) -> int:
    """Chi phí một action, không âm; ưu tiên đổi cube > fragile > mở cầu > thường."""
    if action == Action.SWITCH_CONTROL:
        return CONTROL_COST
    # Cube đơn không làm vỡ fragile nên chỉ tính rủi ro cho block nguyên vẹn
    if child.split_positions is None and any(
        board.tile_at(cell).kind == TileType.FRAGILE for cell in child.occupied_cells
    ):
        return FRAGILE_COST
    # Bit bật = cầu đóng
    if parent.bridge_mask & ~child.bridge_mask:
        return SWITCH_COST
    return NORMAL_COST


def solution_cost(board: Board, actions: Sequence[Action]) -> int:
    """Tổng chi phí của một lời giải hợp lệ"""
    total, state = 0, board.initial_state
    for action in actions:
        child = get_next_state(board, state, action)
        if child is None:
            raise ValueError("solution_cost: chuỗi action không hợp lệ")
        total += move_cost(board, state, action, child)
        state = child
    return total


class UCSSolver(BaseSolver):
    """UCS: tìm lời giải có tổng chi phí nhỏ nhất theo `move_cost`."""

    name = "UCS"

    def _search(self, board: Board) -> Tuple[Optional[Sequence[Action]], int]:
        start = board.initial_state
        came_from: CameFrom = {start: (None, None)}
        best_cost: Dict[State, int] = {start: 0}

        tie = count()  # Phá hòa khi cùng chi phí, tránh so sánh State trong heap
        frontier = [(0, next(tie), start)]
        expanded = 0

        while frontier:
            cost, _, state = heapq.heappop(frontier)
            if cost > best_cost[state]:
                continue  # Đã có đường rẻ hơn

            # Goal test lúc lấy node: bắt buộc để UCS tối ưu khi chi phí không đều
            if is_goal(board, state):
                return self.reconstruct_path(came_from, state), expanded
            expanded += 1

            for action, child in get_successors(board, state):
                new_cost = cost + move_cost(board, state, action, child)
                if child not in best_cost or new_cost < best_cost[child]:
                    best_cost[child] = new_cost
                    came_from[child] = (state, action)
                    heapq.heappush(frontier, (new_cost, next(tie), child))

        return None, expanded
