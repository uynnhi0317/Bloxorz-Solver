from enum import Enum
from typing import Optional, Tuple
from .board import Board
from .state import Orientation, State
from .tile import SwitchEffect, SwitchKind, Tile, TileType

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
    return State(state.row + row_delta, state.column + column_delta, orientation, bridge_mask = state.bridge_mask, split_positions = state.split_positions)

def is_bridge_closed(board: Board, state: State, bridge_id: Optional[str]) -> bool:
    """Kiểm tra xem cây cầu có mã bridge_id đang đóng (True) hay mở (False)"""
    if bridge_id is None:
        return False
    idx = board.bridge_index(bridge_id)
    return bool(state.bridge_mask & (1 << idx))

def is_supported(board: Board, state: State) -> bool:
    """
    Kiểm tra tính hợp lệ cơ bản:
    - Không rơi vào ô VOID
    - Không đè lên closed BRIDGE
    - Không đứng thẳng trên ô FRAGILE
    """
    for cell in state.occupied_cells:
        if board.tile_at(cell).kind == TileType.VOID:
            return False
        if board.tile_at(cell).kind == TileType.BRIDGE and is_bridge_closed(board, state, board.tile_at(cell).tile_id):
            return False

    if not state.is_split and state.orientation == Orientation.STANDING:
        standing_tile = board.tile_at(state.position)
        if standing_tile.kind == TileType.FRAGILE or not standing_tile.allows_standing:
            return False
    return True

def _is_switch_triggered(state: State, tile: Tile) -> bool:
    """Kiểm tra công tắc tại ô tile có được kích hoạt bởi block không"""
    if tile.kind != TileType.SWITCH:
        return False
    if tile.switch_kind == SwitchKind.SOFT:
        return True
    if tile.switch_kind == SwitchKind.HEAVY:
        return not state.is_split and state.orientation == Orientation.STANDING
    return False

def _should_permanent_close(tile: Tile) -> bool:
    """Kiểm tra xem công tắc PERMANENT này dùng để đóng cầu hay mở cầu."""
    meta = dict(tile.metadata)
    target_state = meta.get("state", "open").lower()
    return target_state == "closed"

def _apply_switch_effect(board: Board, mask: int, tile: Tile) -> int:
    """Cập nhật dãy bit bridge_mask theo hiệu ứng TOGGLE hoặc PERMANENT của switch"""
    for bridge_id in tile.targets:
        idx = board.bridge_index(bridge_id)
        bit = 1 << idx

        if tile.effect == SwitchEffect.TOGGLE:
            mask = mask ^ bit

        elif tile.effect == SwitchEffect.PERMANENT:
            if _should_permanent_close(tile):
                mask = mask | bit # Đóng cầu cố định
            else:
                mask = mask & (~bit) # Mở cầu cố định

    return mask

def apply_switches(board: Board, state: State) -> State:
    """Kiểm tra tất cả các ô block đang chiếm để kích hoạt switch nếu có"""
    new_mask = state.bridge_mask
    for cell in state.occupied_cells:
        tile = board.tile_at(cell)
        if _is_switch_triggered(state, tile):
            new_mask = _apply_switch_effect(board, new_mask, tile)

    if new_mask == state.bridge_mask:
        return state

    return State(
        state.row,
        state.column,
        state.orientation,
        bridge_mask=new_mask,
        split_positions=state.split_positions,
    )

def get_next_state(board: Board, state: State, direction: Direction) -> Optional[State]:
    """Trả về state hợp lệ sau khi di chuyển và kích hoạt switch nếu có"""
    if state.is_split:
        return None

    next_state = roll(state, direction)
    if not is_supported(board, next_state):
        return None

    next_state = apply_switches(board, next_state)
    # Kiểm tra lại phòng trường hợp block nằm vừa đè lên switch vừa đè lên cầu đang mở mà sau đó bị switch đó đóng lại
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
