from enum import Enum
from typing import Optional, Tuple
from .board import Board
from .state import Orientation, State
from .context import GameContext
from .tile import SwitchKind, Tile, TileType

# Các hướng di chuyển của khối block
class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

DIRECTIONS = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)
DIRECTION_DELTAS = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}

class Action(str, Enum):
    """Hành động trong game: lăn theo hướng hoặc đổi cube được điều khiển."""
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    SWITCH_CONTROL = "switch_control"

ACTIONS = (
    Action.MOVE_UP,
    Action.MOVE_DOWN,
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
    Action.SWITCH_CONTROL,
)
ACTION_DIRECTIONS = {
    Action.MOVE_UP: Direction.UP,
    Action.MOVE_DOWN: Direction.DOWN,
    Action.MOVE_LEFT: Direction.LEFT,
    Action.MOVE_RIGHT: Direction.RIGHT,
}

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

    if state.split_positions is not None:
        if any(board.tile_at(cell).kind == TileType.FRAGILE for cell in state.occupied_cells):
            return False
    elif state.orientation == Orientation.STANDING and board.tile_at(state.position).kind == TileType.FRAGILE:
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

def _move_split_cube(state: State, direction: Direction, selected_cube: Optional[Tuple[int, int]]) -> State:
    """Di chuyển riêng cube đang được điều khiển một ô theo hướng đã chọn"""
    if state.split_positions:
        dr, dc = DIRECTION_DELTAS[direction]
        active = selected_cube if selected_cube is not None else state.split_positions[0]
        target = (active[0] + dr, active[1] + dc)
        other = next(cell for cell in state.split_positions if cell != active)
        if target == other:
            return state
        return State.split(first = target, second = other, bridge_mask = state.bridge_mask)
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
    )

def get_next_state(board: Board, state: State, direction: Direction, selected_cube: Optional[Tuple[int, int]] = None) -> Optional[State]:
    """Tính cấu hình vật lý tiếp theo; quyền điều khiển được truyền riêng"""
    # Trường hợp block đang bị phân chia, di chuyển hướng của block con
    if state.split_positions is not None:
        next_state = _move_split_cube(state, direction, selected_cube)
        if next_state == state or not is_supported(board, next_state):
            return None
        next_state = _merge_if_adjacent(next_state)
        next_state = apply_switches(board, next_state, state)
        return next_state if is_supported(board, next_state) else None

    # Trường hợp block không bị chia
    next_state = roll(state, direction)
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

def get_next_context(board: Board, context: GameContext, action: Action | Direction) -> Optional[GameContext]:
    """Tính bước chơi tiếp theo, giữ quyền điều khiển ngoài State vật lý"""
    # Nếu đổi điều khiển qua cube khác
    if action == Action.SWITCH_CONTROL:
        return context.switch_control() if context.state.split_positions is not None else None

    # Nếu di chuyển 4 phía
    direction = ACTION_DIRECTIONS[action] if isinstance(action, Action) else action
    next_state = get_next_state(board, context.state, direction, context.selected_cube)
    if next_state is None:
        return None

    if next_state.split_positions is None:
        # Nếu trạng thái kế tiếp vẫn là block 1x1x2
        selected_cube = None
    elif context.state.split_positions is not None and context.selected_cube is not None:
        # Nếu trạng thái kế tiếp và ngay trước đó là 2 cube riêng lẻ 
        dr, dc = DIRECTION_DELTAS[direction]
        selected_cube = (context.selected_cube[0] + dr, context.selected_cube[1] + dc)
    else:
        # Nếu tới trạng thái kế tiếp block mới split
        selected_cube = next_state.split_positions[0]

    return GameContext(next_state, selected_cube)

def get_successors(board: Board, context: GameContext) -> Tuple[Tuple[Action, GameContext], ...]:
    """
        Liệt kê các nước di chuyển hợp lệ và chuyển quyền điều khiển khi block đã tách
        Trả về: Tuple gồm các cặp thuộc tính tương ứng cho nước đi hợp lệ
        + Hướng lăn để đạt được đến trạng thái hợp lệ
        + Trạng thái hợp lệ
    """
    successors = []
    for action in ACTIONS:
        next_context = get_next_context(board, context, action)
        if next_context is not None:
            successors.append((action, next_context))
    return tuple(successors)

def is_goal(board: Board, state: State | GameContext) -> bool:
    """Kiểm tra block đứng thẳng trên ô goal"""
    if isinstance(state, GameContext):
        state = state.state
    return state.split_positions is None and state.orientation == Orientation.STANDING and state.position == board.goal
