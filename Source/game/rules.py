from enum import Enum
from typing import Optional, Tuple
from .board import Board
from .state import Orientation, State
from .tile import SwitchKind, Tile, TileType

class Action(str, Enum):
    """Hành động trong game: lăn theo hướng hoặc đổi cube được điều khiển."""
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    SWITCH_CONTROL = "switch_control"

MOVE_ACTIONS = (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT)
ACTIONS = (
    *MOVE_ACTIONS,
    Action.SWITCH_CONTROL,
)
DIRECTION_DELTAS = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}

"""
    - Từ điển lưu trạng thái block khi di chuyển 4 phía
    - Lưu trữ trạng thái sau khi di chuyển bao gồm (hướng đặt block, tọa độ hàng điểm neo, tọa độ cột điểm neo)
    dựa trên hướng đặt block trước đó và hướng di chuyển
"""
TRANSITIONS = {
    Orientation.STANDING: {
        Action.MOVE_UP: (Orientation.VERTICAL, -2, 0),
        Action.MOVE_DOWN: (Orientation.VERTICAL, 1, 0),
        Action.MOVE_LEFT: (Orientation.HORIZONTAL, 0, -2),
        Action.MOVE_RIGHT: (Orientation.HORIZONTAL, 0, 1),
    },
    Orientation.HORIZONTAL: {
        Action.MOVE_UP: (Orientation.HORIZONTAL, -1, 0),
        Action.MOVE_DOWN: (Orientation.HORIZONTAL, 1, 0),
        Action.MOVE_LEFT: (Orientation.STANDING, 0, -1),
        Action.MOVE_RIGHT: (Orientation.STANDING, 0, 2),
    },
    Orientation.VERTICAL: {
        Action.MOVE_UP: (Orientation.STANDING, -1, 0),
        Action.MOVE_DOWN: (Orientation.STANDING, 2, 0),
        Action.MOVE_LEFT: (Orientation.VERTICAL, 0, -1),
        Action.MOVE_RIGHT: (Orientation.VERTICAL, 0, 1),
    },
}

def roll(state: State, action: Action) -> State:
    """Trả về trạng thái sau khi di chuyển, chưa xét tính hợp lệ trên board"""
    orientation, row_delta, column_delta = TRANSITIONS[state.orientation][action]
    return State(state.row + row_delta, state.column + column_delta, orientation, bridge_mask = state.bridge_mask, split_positions = state.split_positions, selected_cube=state.selected_cube)

def is_bridge_closed(board: Board, state: State, bridge_id: Optional[str]) -> bool:
    """Kiểm tra xem cây cầu có mã bridge_id đang đóng (True) hay mở (False)"""
    if bridge_id is None:
        return False
    return bool(state.bridge_mask & board.bridge_bit[bridge_id])

def is_supported(board: Board, state: State) -> bool:
    """
    Kiểm tra tính hợp lệ cơ bản:
    - Không rơi vào ô VOID
    - Không thể đi qua cầu đang đóng
    - Không đứng thẳng trên ô FRAGILE
    """
    for cell in state.occupied_cells:
        tile = board.tile_at(cell)
        if tile.kind == TileType.VOID:
            return False
        if tile.kind == TileType.BRIDGE and is_bridge_closed(board, state, tile.tile_id):
            return False

    # Cube đơn lẻ sau khi tách không làm vỡ fragile tile.
    # Chỉ block nguyên vẹn khi đứng thẳng mới rơi trên fragile tile.
    if state.split_positions is None and state.orientation == Orientation.STANDING and board.tile_at(state.position).kind == TileType.FRAGILE:
        return False
    return True

def _is_switch_triggered(state: State, tile: Tile) -> bool:
    """Kiểm tra công tắc tại ô tile có được kích hoạt bởi block không"""
    if tile.kind != TileType.SWITCH:
        return False
    if tile.switch_kind == SwitchKind.SOFT:
        return True
    if tile.switch_kind == SwitchKind.HEAVY:
        return state.split_positions is None and state.orientation == Orientation.STANDING
    return False

def _split_at_switch(board: Board, state: State, previous_state: State) -> Optional[State]:
    """Tách block tại split switch, đặt hai cube ở các ô đích của switch."""
    if state.split_positions is not None:
        return None
    newly_occupied = set(state.occupied_cells) - set(previous_state.occupied_cells)
    for cell in newly_occupied:
        tile = board.tile_at(cell)
        if tile.kind == TileType.SWITCH and tile.switch_kind == SwitchKind.SPLIT:
            return State.split(*tile.destinations, bridge_mask=state.bridge_mask)
    return None

def _merge_if_adjacent(state: State) -> State:
    """Ghép hai cube thành block 1x1x2 khi chúng đứng cạnh nhau"""
    if state.split_positions is None:
        return state
    (r1, c1), (r2, c2) = state.split_positions
    if r1 == r2 and abs(c1 - c2) == 1:
        return State(r1, min(c1, c2), Orientation.HORIZONTAL, state.bridge_mask)
    if c1 == c2 and abs(r1 - r2) == 1:
        return State(min(r1, r2), c1, Orientation.VERTICAL, state.bridge_mask)
    return state

def _move_split_cube(state: State, action: Action) -> State:
    """Di chuyển riêng cube đang được điều khiển một ô theo hướng đã chọn"""
    if state.split_positions:
        dr, dc = DIRECTION_DELTAS[action]
        active = state.selected_cube if state.selected_cube is not None else state.split_positions[0]
        target = (active[0] + dr, active[1] + dc)
        other = next(cell for cell in state.split_positions if cell != active)
        if target == other:
            return state
        return State(state.row, state.column, bridge_mask=state.bridge_mask, split_positions=(target, other), selected_cube=target)
    return state

def apply_switches(board: Board, state: State, previous_state: State) -> State:
    """
        Kiểm tra các ô mới được block chiếm để kích hoạt switch nếu có

        Tại một switch, chỉ một trong toggle_mask, close_mask, open_mask khác 0.
        Bit bật biểu thị cầu đóng; bit tắt biểu thị cầu mở.
    """
    new_mask = state.bridge_mask
    newly_occupied = set(state.occupied_cells) - set(previous_state.occupied_cells)
    for cell in newly_occupied:
        entry = board.switch_masks.get(cell)
        if entry is None:
            continue
        if not _is_switch_triggered(state, board.tile_at(cell)):
            continue
        toggle_mask, close_mask, open_mask = entry
        # Toggle đảo bit, permanent đặt bit theo trạng thái đích.
        new_mask = ((new_mask ^ toggle_mask) | close_mask) & ~open_mask

    if new_mask == state.bridge_mask:
        return state

    return State(
        state.row,
        state.column,
        state.orientation,
        bridge_mask=new_mask,
        split_positions=state.split_positions,
        selected_cube=state.selected_cube,
    )

def get_next_state(board: Board, state: State, action: Action) -> Optional[State]:
    """Tính trạng thái kế tiếp, bao gồm cấu hình và quyền điều khiển."""
    if action == Action.SWITCH_CONTROL:
        return state.switch_control() if state.split_positions is not None else None

    # Trường hợp block đang bị phân chia, di chuyển hướng của block con
    if state.split_positions is not None:
        next_state = _move_split_cube(state, action)
        if next_state == state or not is_supported(board, next_state):
            return None
        next_state = _merge_if_adjacent(next_state)
        next_state = apply_switches(board, next_state, state)
        return next_state if is_supported(board, next_state) else None

    # Trường hợp block không bị chia
    next_state = roll(state, action)
    if not is_supported(board, next_state):
        return None

    # Trường hợp block dẫm lên split switch
    split_state = _split_at_switch(board, next_state, state)
    if split_state is not None:
        if not is_supported(board, split_state):
            return None
        split_state = apply_switches(board, split_state, next_state)
        return split_state if is_supported(board, split_state) else None

    # Trường hợp block dẫm lên soft/heavy switch
    next_state = apply_switches(board, next_state, state)

    # Kiểm tra lại phòng trường hợp block nằm vừa đè lên switch vừa đè lên cầu đang mở mà sau đó bị switch đó đóng lại
    if not is_supported(board, next_state):
        return None

    return next_state

def get_successors(board: Board, state: State) -> Tuple[Tuple[Action, State], ...]:
    """
        Liệt kê các nước di chuyển hợp lệ và chuyển quyền điều khiển khi block đã tách.
        Trả về: Tuple gồm các cặp thuộc tính tương ứng cho nước đi hợp lệ
        + Hướng lăn để đạt được đến trạng thái hợp lệ
        + Trạng thái hợp lệ
    """
    successors = []
    for action in ACTIONS:
        next_state = get_next_state(board, state, action)
        if next_state is not None:
            successors.append((action, next_state))
    return tuple(successors)

def is_goal(board: Board, state: State) -> bool:
    """Kiểm tra block đứng thẳng trên ô goal"""
    return state.split_positions is None and state.orientation == Orientation.STANDING and state.position == board.goal
