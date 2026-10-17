from enum import Enum
from typing import Optional, Tuple
from .board import Board
from .state import Orientation, State
from .tile import TileType

# Các hướng di chuyển của khối block
class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

DIRECTIONS = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)

"""
    - Từ điển lưu trạng thái block khi di chuyển 4 phía
    - Lưu trữ trạng thái sau khi di chuyển bao gồm (hướng đặt block, tọa độ hàng điểm neo, tọa độ cột điểm neo)
    dựa trên hướng đặt block trước đó và hướng di chuyển
"""
TRANSITIONS = {
    Orientation.STANDING: {
        Direction.UP: (Orientation.VERTICAL, -2, 0),
        Direction.DOWN: (Orientation.VERTICAL, 1, 0),
        Direction.LEFT: (Orientation.HORIZONTAL, 0, -2),
        Direction.RIGHT: (Orientation.HORIZONTAL, 0, 1),
    },
    Orientation.HORIZONTAL: {
        Direction.UP: (Orientation.HORIZONTAL, -1, 0),
        Direction.DOWN: (Orientation.HORIZONTAL, 1, 0),
        Direction.LEFT: (Orientation.STANDING, 0, -1),
        Direction.RIGHT: (Orientation.STANDING, 0, 2),
    },
    Orientation.VERTICAL: {
        Direction.UP: (Orientation.STANDING, -1, 0),
        Direction.DOWN: (Orientation.STANDING, 2, 0),
        Direction.LEFT: (Orientation.VERTICAL, 0, -1),
        Direction.RIGHT: (Orientation.VERTICAL, 0, 1),
    },
}

def roll(state: State, direction: Direction) -> State:
    """Trả về trạng thái sau khi di chuyển, chưa xét tính hợp lệ trên board"""
    orientation, row_delta, column_delta = TRANSITIONS[state.orientation][direction]
    return State(state.row + row_delta, state.column + column_delta, orientation, bridge_mask = state.bridge_mask)

def is_supported(board: Board, state: State) -> bool:
    """Kiểm tra tính hợp lệ cơ bản: block có trong board và có nằm trên ô trống không"""
    for cell in state.occupied_cells:
        if not board.in_bounds(cell):
            return False
        if board.tile_at(cell).kind == TileType.VOID:
            return False
    return True

def get_next_state(board: Board, state: State, direction: Direction) -> Optional[State]:
    """Trả về state hợp lệ sau khi di chuyển"""
    if state.is_split:
        return None

    next_state = roll(state, direction)
    if not is_supported(board, next_state):
        return None

    return next_state

def get_successors(board: Board, state: State) -> Tuple[Tuple[Direction, State], ...]:
    """
        Liệt kê các nước đi hợp lệ theo 4 hướng di chuyển
        Trả về: Tuple gồm các cặp thuộc tính tương ứng cho nước đi hợp lệ
        + Hướng lăn để đạt được đến trạng thái hợp lệ
        + Trạng thái hợp lệ
    """
    successors = []
    for direction in DIRECTIONS:
        next_state = get_next_state(board, state, direction)
        if next_state is not None:
            successors.append((direction, next_state))
    return tuple(successors)

def is_goal(board: Board, state: State) -> bool:
    """Kiểm tra block đứng thẳng trên ô goal"""
    return not state.is_split and state.orientation == Orientation.STANDING and state.position == board.goal
