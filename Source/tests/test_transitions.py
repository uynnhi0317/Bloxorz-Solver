import pytest

from game import (Action, Board, Move, Orientation, State, Tile, TileType,
                  get_next_state, get_successors, is_goal)
from game.rules import MOVE_ACTIONS, is_supported, roll

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
    (Orientation.STANDING, Action.MOVE_UP, Orientation.VERTICAL, (0, 2), ((0, 2), (1, 2))),
    (Orientation.STANDING, Action.MOVE_DOWN, Orientation.VERTICAL, (3, 2), ((3, 2), (4, 2))),
    (Orientation.STANDING, Action.MOVE_LEFT, Orientation.HORIZONTAL, (2, 0), ((2, 0), (2, 1))),
    (Orientation.STANDING, Action.MOVE_RIGHT, Orientation.HORIZONTAL, (2, 3), ((2, 3), (2, 4))),
    # Nằm ngang chiếm (2, 2), (2, 3)
    (Orientation.HORIZONTAL, Action.MOVE_UP, Orientation.HORIZONTAL, (1, 2), ((1, 2), (1, 3))),
    (Orientation.HORIZONTAL, Action.MOVE_DOWN, Orientation.HORIZONTAL, (3, 2), ((3, 2), (3, 3))),
    (Orientation.HORIZONTAL, Action.MOVE_LEFT, Orientation.STANDING, (2, 1), ((2, 1),)),
    (Orientation.HORIZONTAL, Action.MOVE_RIGHT, Orientation.STANDING, (2, 4), ((2, 4),)),
    # Nằm dọc chiếm (2, 2), (3, 2)
    (Orientation.VERTICAL, Action.MOVE_UP, Orientation.STANDING, (1, 2), ((1, 2),)),
    (Orientation.VERTICAL, Action.MOVE_DOWN, Orientation.STANDING, (4, 2), ((4, 2),)),
    (Orientation.VERTICAL, Action.MOVE_LEFT, Orientation.VERTICAL, (2, 1), ((2, 1), (3, 1))),
    (Orientation.VERTICAL, Action.MOVE_RIGHT, Orientation.VERTICAL, (2, 3), ((2, 3), (3, 3))),
]
TRANSITION_IDS = [f"{o.value}-{d.value}" for o, d, *_ in TRANSITION_CASES]

# Bảng chuyển orientation khi lăn
class TestRoll:
    def test_cases_cover_all_pairs_exactly_once(self):
        pairs = [(o, d) for o, d, *_ in TRANSITION_CASES]
        assert len(pairs) == len(set(pairs))
        assert set(pairs) == {(o, d) for o in Orientation for d in MOVE_ACTIONS}

    @pytest.mark.parametrize("orientation, action, new_orientation, anchor, cells",
                             TRANSITION_CASES, ids=TRANSITION_IDS)
    def test_roll(self, orientation, action, new_orientation, anchor, cells):
        next_state = roll(State(2, 2, orientation), action)
        assert next_state.orientation == new_orientation
        assert next_state.position == anchor
        assert next_state.occupied_cells == cells

    def test_roll_keeps_bridge_mask(self):
        assert roll(State(2, 2, bridge_mask=0b101), Action.MOVE_UP).bridge_mask == 0b101

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
        State(4, 2, Orientation.VERTICAL),
    ], ids=["standing_top", "standing_bottom", "standing_left", "standing_right",
            "horizontal_right_half", "vertical_bottom_half"])
    def test_out_of_bounds(self, open_board, state):
        assert not is_supported(open_board, state)

    @pytest.mark.parametrize("state", [
        State(1, 1),
        State(1, 0, Orientation.HORIZONTAL),
        State(0, 1, Orientation.VERTICAL),
    ], ids=["standing", "horizontal_right_half", "vertical_bottom_half"])
    def test_on_void(self, holey_board, state):
        assert not is_supported(holey_board, state)

# Lăn block trên board
class TestGetNextState:
    def test_valid_move_returns_rolled_state(self, open_board):
        state = State(2, 2)
        assert get_next_state(open_board, state, Action.MOVE_RIGHT) == roll(state, Action.MOVE_RIGHT)

    def test_off_board_returns_none(self, open_board):
        assert get_next_state(open_board, State(0, 2), Action.MOVE_UP) is None

    def test_into_void_returns_none(self, holey_board):
        assert get_next_state(holey_board, State(1, 2), Action.MOVE_LEFT) is None

# Liệt kê nước đi hợp lệ
class TestGetSuccessors:
    def test_open_board_has_all_move_actions(self, open_board):
        successors = get_successors(open_board, State(2, 2))
        assert [move for move, _ in successors] == [Move(a) for a in MOVE_ACTIONS]

    def test_small_board_filters_invalid_moves(self):
        # Board 2x2: đứng thì mọi hướng đều cần 2 ô nên không có nước đi
        board = _make_board(["G.", ".."], start=(0, 0))
        assert get_successors(board, State(0, 0)) == ()
        assert get_successors(board, State(0, 0, Orientation.HORIZONTAL)) == (
            (Move(Action.MOVE_DOWN), State(1, 0, Orientation.HORIZONTAL)),
        )

# Điều kiện thắng: chỉ thắng khi block nguyên vẹn đứng thẳng trên goal.
class TestIsGoal:
    def test_standing_on_goal(self, open_board):
        assert is_goal(open_board, State(0, 0))

    @pytest.mark.parametrize("state", [
        State(0, 0, Orientation.HORIZONTAL),
        State(1, 1),
        State.split((0, 0), (2, 2)),
    ], ids=["lying_on_goal", "standing_elsewhere", "single_cube_on_goal"])
    def test_false(self, open_board, state):
        assert not is_goal(open_board, state)

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

    @pytest.mark.parametrize("a, b", [
        (State(1, 1, bridge_mask=0), State(1, 1, bridge_mask=1)),
        (State.split((1, 1), (3, 3)), State.split((1, 1), (3, 4))),
    ], ids=["bridge_mask", "split_positions"])
    def test_states_differing_in_one_field_are_distinct(self, a, b):
        # Solver dùng State làm key của visited
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

    def test_split_overrides_anchor_and_orientation(self):
        state = State(9, 9, Orientation.VERTICAL, split_positions=((3, 3), (1, 1)))
        assert state.position == (1, 1)
        assert state.orientation == Orientation.STANDING

    def test_split_same_cell_raises(self):
        with pytest.raises(ValueError, match="cùng 1 ô"):
            State.split((2, 2), (2, 2))

# Block đã tách: di chuyển từng cube, đổi điều khiển, tự ghép khi kề nhau
class TestSplitBlock:
    def test_move_of_cube_not_in_state_returns_none(self, open_board):
        state = State.split((1, 1), (3, 3))
        assert get_next_state(open_board, state, Move(Action.MOVE_LEFT, (2, 2))) is None

    def test_bare_action_on_split_state_returns_none(self, open_board):
        # Block đã tách bắt buộc phải nói rõ cube nào di chuyển
        assert get_next_state(open_board, State.split((1, 1), (3, 3)), Action.MOVE_LEFT) is None

    def test_successors_try_both_cubes(self, open_board):
        state = State.split((1, 1), (3, 3))
        successors = get_successors(open_board, state)
        assert [move for move, _ in successors] == [
            Move(action, cube) for cube in state.split_positions for action in MOVE_ACTIONS]

    @pytest.mark.parametrize("action, target", [
        (Action.MOVE_UP, (0, 1)),
        (Action.MOVE_DOWN, (2, 1)),
        (Action.MOVE_LEFT, (1, 0)),
        (Action.MOVE_RIGHT, (1, 2)),
    ], ids=["up", "down", "left", "right"])
    def test_moves_only_given_cube_one_cell(self, open_board, action, target):
        # Cube (1,1) đi 1 ô, cube (3,3) đứng yên
        next_state = get_next_state(open_board, State.split((1, 1), (3, 3)), Move(action, (1, 1)))
        assert next_state == State.split(target, (3, 3))

    def test_moves_second_cube(self, open_board):
        next_state = get_next_state(open_board, State.split((1, 1), (3, 3)), Move(Action.MOVE_LEFT, (3, 3)))
        assert next_state.split_positions == ((1, 1), (3, 2))

    def test_cube_off_board_returns_none(self, open_board):
        state = State.split((0, 4), (4, 0))
        assert get_next_state(open_board, state, Move(Action.MOVE_UP, (0, 4))) is None

    def test_cube_into_other_cube_returns_none(self, open_board):
        # Trạng thái dựng tay: 2 cube kề nhau nhưng chưa ghép
        state = State.split((1, 1), (1, 2))
        assert get_next_state(open_board, state, Move(Action.MOVE_RIGHT, (1, 1))) is None

    @pytest.mark.parametrize("state, move, merged", [
        (State.split((1, 1), (1, 3), bridge_mask=0b1), Move(Action.MOVE_RIGHT, (1, 1)),
         State(1, 2, Orientation.HORIZONTAL, bridge_mask=0b1)),
        (State.split((1, 0), (1, 2)), Move(Action.MOVE_LEFT, (1, 2)),
         State(1, 0, Orientation.HORIZONTAL)),
        (State.split((1, 1), (3, 1)), Move(Action.MOVE_DOWN, (1, 1)),
         State(2, 1, Orientation.VERTICAL)),
    ], ids=["horizontal", "horizontal_from_right", "vertical"])
    def test_adjacent_cubes_merge(self, open_board, state, move, merged):
        assert get_next_state(open_board, state, move) == merged
