import pytest

from game import (Board, Direction, Orientation, State, Tile, TileType,
                  get_next_state, get_successors, is_goal)
from game.rules import DIRECTIONS, is_supported, roll

SYMBOLS = {".": TileType.FLOOR, "#": TileType.VOID, "G": TileType.GOAL}

# Tạo board từ chuỗi ký hiệu
def _make_board(rows, start):
    return Board(tiles=[[Tile(SYMBOLS[ch]) for ch in row] for row in rows], start=start)

@pytest.fixture
def open_board():
    return _make_board(["G....", ".....", ".....", ".....", "....."], start=(2, 2))

@pytest.fixture
def holey_board():
    return _make_board(["G..", ".#.", "..."], start=(0, 2))

# Block xuất phát với neo (2, 2)
TRANSITION_CASES = [
    # Đứng chiếm (2, 2)
    (Orientation.STANDING, Direction.UP, Orientation.VERTICAL, (0, 2), ((0, 2), (1, 2))),
    (Orientation.STANDING, Direction.DOWN, Orientation.VERTICAL, (3, 2), ((3, 2), (4, 2))),
    (Orientation.STANDING, Direction.LEFT, Orientation.HORIZONTAL, (2, 0), ((2, 0), (2, 1))),
    (Orientation.STANDING, Direction.RIGHT, Orientation.HORIZONTAL, (2, 3), ((2, 3), (2, 4))),
    # Nằm ngang chiếm (2, 2), (2, 3)
    (Orientation.HORIZONTAL, Direction.UP, Orientation.HORIZONTAL, (1, 2), ((1, 2), (1, 3))),
    (Orientation.HORIZONTAL, Direction.DOWN, Orientation.HORIZONTAL, (3, 2), ((3, 2), (3, 3))),
    (Orientation.HORIZONTAL, Direction.LEFT, Orientation.STANDING, (2, 1), ((2, 1),)),
    (Orientation.HORIZONTAL, Direction.RIGHT, Orientation.STANDING, (2, 4), ((2, 4),)),
    # Nằm dọc chiếm (2, 2), (3, 2)
    (Orientation.VERTICAL, Direction.UP, Orientation.STANDING, (1, 2), ((1, 2),)),
    (Orientation.VERTICAL, Direction.DOWN, Orientation.STANDING, (4, 2), ((4, 2),)),
    (Orientation.VERTICAL, Direction.LEFT, Orientation.VERTICAL, (2, 1), ((2, 1), (3, 1))),
    (Orientation.VERTICAL, Direction.RIGHT, Orientation.VERTICAL, (2, 3), ((2, 3), (3, 3))),
]
TRANSITION_IDS = [f"{o.value}-{d.value}" for o, d, *_ in TRANSITION_CASES]

# Bảng chuyển orientation khi lăn
class TestRoll:
    def test_cases_cover_all_pairs_exactly_once(self):
        pairs = [(o, d) for o, d, *_ in TRANSITION_CASES]
        assert len(pairs) == len(set(pairs))
        assert set(pairs) == {(o, d) for o in Orientation for d in Direction}

    @pytest.mark.parametrize("orientation, direction, new_orientation, anchor, cells",
                             TRANSITION_CASES, ids=TRANSITION_IDS)
    def test_roll(self, orientation, direction, new_orientation, anchor, cells):
        next_state = roll(State(2, 2, orientation), direction)
        assert next_state.orientation == new_orientation
        assert next_state.position == anchor
        assert next_state.occupied_cells == cells

    def test_roll_keeps_bridge_mask(self):
        assert roll(State(2, 2, bridge_mask=0b101), Direction.UP).bridge_mask == 0b101

# Block phải nằm hoàn toàn trên sàn
class TestIsSupported:
    @pytest.mark.parametrize("state", [
        State(2, 2),
        State(0, 0, Orientation.HORIZONTAL),
        State(3, 4, Orientation.VERTICAL),
    ], ids=["standing", "horizontal", "vertical"])
    def test_on_floor(self, open_board, state):
        assert is_supported(open_board, state)

    @pytest.mark.parametrize("state", [
        State(-1, 2),
        State(5, 2),
        State(2, -1),
        State(2, 5),
        State(2, 4, Orientation.HORIZONTAL),
        State(2, -1, Orientation.HORIZONTAL),
        State(4, 2, Orientation.VERTICAL),
        State(-1, 2, Orientation.VERTICAL),
    ], ids=["standing_top", "standing_bottom", "standing_left", "standing_right",
            "horizontal_right_half", "horizontal_left_half",
            "vertical_bottom_half", "vertical_top_half"])
    def test_out_of_bounds(self, open_board, state):
        assert not is_supported(open_board, state)

    @pytest.mark.parametrize("state", [
        State(1, 1),
        State(1, 0, Orientation.HORIZONTAL),
        State(1, 1, Orientation.HORIZONTAL),
        State(0, 1, Orientation.VERTICAL),
        State(1, 1, Orientation.VERTICAL),
    ], ids=["standing", "horizontal_right_half", "horizontal_left_half",
            "vertical_bottom_half", "vertical_top_half"])
    def test_on_void(self, holey_board, state):
        assert not is_supported(holey_board, state)

# Lăn block trên board
class TestGetNextState:
    @pytest.mark.parametrize("orientation", list(Orientation))
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_valid_move_equals_roll(self, open_board, orientation, direction):
        state = State(2, 2, orientation)
        assert get_next_state(open_board, state, direction) == roll(state, direction)

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_off_board_returns_none(self, direction):
        board = _make_board(["G"], start=(0, 0))
        assert get_next_state(board, State(0, 0), direction) is None

    @pytest.mark.parametrize("state, direction", [
        (State(1, 2), Direction.LEFT),
        (State(0, 1), Direction.DOWN),
    ], ids=["left_into_void", "down_into_void"])
    def test_into_void_returns_none(self, holey_board, state, direction):
        assert get_next_state(holey_board, state, direction) is None

# Liệt kê nước đi hợp lệ
class TestGetSuccessors:
    def test_open_board_has_all_directions(self, open_board):
        successors = get_successors(open_board, State(2, 2))
        assert [direction for direction, _ in successors] == list(DIRECTIONS)

    def test_small_board_filters_invalid_moves(self):
        # Board 2x2: đứng thì mọi hướng đều cần 2 ô nên không có nước đi
        board = _make_board(["G.", ".."], start=(0, 0))
        assert get_successors(board, State(0, 0)) == ()
        assert get_successors(board, State(0, 0, Orientation.HORIZONTAL)) == (
            (Direction.DOWN, State(1, 0, Orientation.HORIZONTAL)),
        )

# Điều kiện thắng: chỉ thắng khi block nguyên vẹn đứng thẳng trên goal.
class TestIsGoal:
    @pytest.mark.parametrize("state", [
        State(0, 0),
        State(0, 0, bridge_mask=0b11),
    ], ids=["standing", "ignores_bridge_mask"])
    def test_true(self, open_board, state):
        assert is_goal(open_board, state)

    @pytest.mark.parametrize("state", [
        State(0, 0, Orientation.HORIZONTAL),
        State(0, 0, Orientation.VERTICAL),
        State(1, 1),
        State.split((0, 0), (2, 2)),
    ], ids=["horizontal_on_goal", "vertical_on_goal", "standing_elsewhere", "split"])
    def test_false(self, open_board, state):
        assert not is_goal(open_board, state)

    def test_false_when_lying_with_second_half_on_goal(self):
        board = _make_board([".G", ".."], start=(1, 0))
        assert not is_goal(board, State(0, 0, Orientation.HORIZONTAL))

# bridge_mask, split_positions và tính phân biệt của State.
class TestState:
    def test_toggle_flips_only_target_bit(self):
        s = State(0, 0, bridge_mask=0b101)
        assert s.toggle_bridge(1).bridge_mask == 0b111
        assert s.toggle_bridge(0).bridge_mask == 0b100

    @pytest.mark.parametrize("state", [
        State(2, 3, Orientation.HORIZONTAL),
        State.split((1, 1), (3, 3)),
    ], ids=["whole_block", "split_block"])
    def test_toggle_keeps_block_placement(self, state):
        toggled = state.toggle_bridge(0)
        assert toggled.bridge_mask == 0b1
        assert toggled.position == state.position
        assert toggled.orientation == state.orientation
        assert toggled.split_positions == state.split_positions

    def test_toggle_negative_index_raises(self):
        with pytest.raises(ValueError, match="không âm"):
            State(0, 0).toggle_bridge(-1)

    def test_negative_bridge_mask_raises(self):
        with pytest.raises(ValueError, match="bridge_mask"):
            State(0, 0, bridge_mask=-1)

    def test_same_position_different_mask_are_distinct(self):
        # Solver dùng State làm key của visited, nên khác mask phải là 2 state riêng
        a = State(1, 1, bridge_mask=0)
        b = State(1, 1, bridge_mask=1)
        assert a != b
        assert len({a, b}) == 2

    @pytest.mark.parametrize("first, second", [
        ((1, 1), (3, 3)),
        ((3, 3), (1, 1)),
    ], ids=["ordered", "reversed"])
    def test_split_positions_are_normalized(self, first, second):
        state = State.split(first, second)
        assert state.split_positions == ((1, 1), (3, 3))
        assert state.occupied_cells == ((1, 1), (3, 3))

    def test_split_state_equal_regardless_of_order(self):
        a, b = State.split((1, 1), (3, 3)), State.split((3, 3), (1, 1))
        assert a == b
        assert hash(a) == hash(b)

    def test_split_overrides_anchor_and_orientation(self):
        state = State(9, 9, Orientation.VERTICAL, split_positions=((3, 3), (1, 1)))
        assert state.position == (1, 1)
        assert state.orientation == Orientation.STANDING

    def test_split_same_cell_raises(self):
        with pytest.raises(ValueError, match="cùng 1 ô"):
            State.split((2, 2), (2, 2))