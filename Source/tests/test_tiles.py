import json
from pathlib import Path

import pytest

from game import (Action, Board, Move, Orientation, State, SwitchEffect, SwitchKind,
                  Tile, TileType, get_next_state, is_goal, load_stage)
from game.rules import apply_switches, is_supported
from game.tile import PermanentState

MOCK_PATH = Path(__file__).parent / "data" / "mock_data.json"

# Board 1 hàng: floor + middle_tiles + goal, start ở (0, 0)
def _simple_board(middle_tiles, **kwargs):
    row = [{"type": "floor"}] + middle_tiles + [{"type": "goal"}]
    return Board.from_dict({"start": [0, 0], "tiles": [row], **kwargs})

# Board 1 hàng: floor | switch | floor | bridge 'b' | goal.
def _switch_board(switch, effect="toggle", closed=True, set_to=None):
    spec = {"type": "switch", "switch": switch, "effect": effect, "targets": ["b"]}
    if set_to:
        spec["set_to"] = set_to
    return _simple_board([spec, {"type": "floor"}, {"type": "bridge", "id": "b"}],
                         initial_closed_bridges=["b"] if closed else [])

# Trạng thái trước đó cho apply_switches: đứng ở goal (0, 4), không chiếm switch
_BEFORE = State(0, 4)

# Board 3x5 cho split, cầu 'b' đóng sẵn:
#   F  F  Fr F  S      S: soft -> b, Fr: fragile
#   P  F  F  G  B      P: split -> (0,3), (2,3); G: goal; B: bridge b
#   F  F  F  F  H      H: heavy -> b
def _split_board():
    floor = {"type": "floor"}
    return Board.from_dict({"start": [1, 2], "initial_closed_bridges": ["b"], "tiles": [
        [floor, floor, {"type": "fragile"}, floor, {"type": "switch", "targets": ["b"]}],
        [{"type": "switch", "switch": "split", "destinations": [[0, 3], [2, 3]]},
         floor, floor, {"type": "goal"}, {"type": "bridge", "id": "b"}],
        [floor, floor, floor, floor, {"type": "switch", "switch": "heavy", "targets": ["b"]}],
    ]})

# Board 1 hàng, cầu 'b' đóng: floor | split -> (0,0), (0,3) | floor | dest_tile | (extra) | goal
def _split_line_board(dest_tile, extra=()):
    split = {"type": "switch", "switch": "split", "destinations": [[0, 0], [0, 3]]}
    return _simple_board([split, {"type": "floor"}, dest_tile, *extra],
                         initial_closed_bridges=["b"])

def _assert_mock_board(board):
    assert board.name == "Mẫu đầy đủ: các loại ô, công tắc và cầu"
    assert (board.height, board.width) == (5, 9)
    assert board.start == (2, 0)
    assert board.goal == (2, 6)
    assert board.bridge_ids == ("bridge_closed_1", "bridge_closed_2", "bridge_toggle")
    assert board.initial_closed_bridges == ("bridge_closed_1", "bridge_closed_2")
    assert board.initial_bridge_mask == 0b011
    assert board.initial_state == State(2, 0, Orientation.STANDING, bridge_mask=0b011)

@pytest.fixture
def mock_data():
    return json.loads(MOCK_PATH.read_text(encoding="utf-8"))

@pytest.fixture
def mock_board():
    return load_stage(MOCK_PATH)

# Tạo tile
class TestNonSwitchTiles:
    @pytest.mark.parametrize("spec, kind", [
        ({"type": "floor"}, TileType.FLOOR),
        ({"type": "void"}, TileType.VOID),
        ({"type": "goal"}, TileType.GOAL),
        ({"type": "fragile"}, TileType.FRAGILE),
        ({"type": "bridge", "id": "b1"}, TileType.BRIDGE),
    ], ids=["floor", "void", "goal", "fragile", "bridge"])
    def test_kind_and_empty_switch_fields(self, spec, kind):
        tile = Tile.from_spec(spec)
        assert tile.kind == kind
        assert tile.switch_kind is None
        assert tile.effect is None
        assert tile.permanent_state is None
        assert tile.targets == ()
        assert tile.destinations == ()

# tile_id, ràng buộc id của bridge và chuẩn hoá metadata
class TestTileIdAndMetadata:
    @pytest.mark.parametrize("spec, expected", [
        ({"type": "bridge", "id": "bridge_a"}, "bridge_a"),
        ({"type": "switch", "targets": ["b1"]}, None),
    ], ids=["bridge", "switch_without_id"])
    def test_tile_id(self, spec, expected):
        assert Tile.from_spec(spec).tile_id == expected

    @pytest.mark.parametrize("spec", [
        {"type": "bridge"},
        {"type": "bridge", "id": ""},
    ], ids=["missing", "empty_string"])
    def test_bridge_without_valid_id_raises(self, spec):
        with pytest.raises(ValueError, match="mã định danh"):
            Tile.from_spec(spec)

    def test_metadata_sorted_and_stringified(self):
        tile = Tile.from_spec({"type": "floor", "metadata": {"z": 1, 2: "x"}})
        assert tile.metadata == (("2", "x"), ("z", "1"))

# Switch SOFT/HEAVY: tổ hợp hợp lệ, mặc định và ràng buộc targets.
class TestSoftHeavySwitch:
    @pytest.mark.parametrize("switch_name, switch_kind", [
        ("soft", SwitchKind.SOFT),
        ("heavy", SwitchKind.HEAVY),
    ], ids=["soft", "heavy"])
    @pytest.mark.parametrize("effect_name, effect", [
        ("toggle", SwitchEffect.TOGGLE),
        ("permanent", SwitchEffect.PERMANENT),
    ], ids=["toggle", "permanent"])
    def test_valid_combinations(self, switch_name, switch_kind, effect_name, effect):
        tile = Tile.from_spec({"type": "switch", "id": "s1", "switch": switch_name,
                               "effect": effect_name, "targets": ["b1", "b2"]})
        assert tile.kind == TileType.SWITCH
        assert tile.tile_id == "s1"
        assert tile.switch_kind == switch_kind
        assert tile.effect == effect
        assert tile.targets == ("b1", "b2")
        assert tile.destinations == ()

    def test_defaults_to_soft_toggle(self):
        tile = Tile.from_spec({"type": "switch", "targets": ["b1"]})
        assert tile.switch_kind == SwitchKind.SOFT
        assert tile.effect == SwitchEffect.TOGGLE

    def test_missing_targets_raises(self):
        with pytest.raises(ValueError, match="cầu kết nối"):
            Tile.from_spec({"type": "switch"})

# Switch PERMANENT: set_to quyết định mở hay đóng cầu, mặc định open
class TestPermanentSwitch:
    @pytest.mark.parametrize("extra, expected", [
        ({}, PermanentState.OPEN),
        ({"set_to": "open"}, PermanentState.OPEN),
        ({"set_to": "close"}, PermanentState.CLOSE),
    ], ids=["default", "open", "close"])
    def test_set_to(self, extra, expected):
        tile = Tile.from_spec({"type": "switch", "effect": "permanent", "targets": ["b1"], **extra})
        assert tile.permanent_state == expected

    def test_invalid_set_to_raises(self):
        with pytest.raises(ValueError, match="set_to"):
            Tile.from_spec({"type": "switch", "effect": "permanent",
                            "set_to": "half", "targets": ["b1"]})

    def test_toggle_ignores_set_to(self):
        tile = Tile.from_spec({"type": "switch", "set_to": "close", "targets": ["b1"]})
        assert tile.permanent_state is None

# Switch SPLIT: đúng 2 đích, không có targets/effect
class TestSplitSwitch:
    def test_valid_split_switch(self):
        tile = Tile.from_spec({"type": "switch", "switch": "split",
                               "destinations": [[1, 2], [3, 4]]})
        assert tile.switch_kind == SwitchKind.SPLIT
        assert tile.effect is None
        assert tile.targets == ()
        assert tile.destinations == ((1, 2), (3, 4))

    def test_destinations_cast_to_int(self):
        tile = Tile.from_spec({"type": "switch", "switch": "split",
                               "destinations": [["1", "2"], [3.0, 4]]})
        assert tile.destinations == ((1, 2), (3, 4))
        assert all(isinstance(v, int) for dest in tile.destinations for v in dest)

    @pytest.mark.parametrize("destinations", [
        [[1, 2]],
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [1, 2]],
    ], ids=["one", "three", "same_cell"])
    def test_wrong_destination_count_raises(self, destinations):
        with pytest.raises(ValueError, match="đúng 2 tọa độ"):
            Tile.from_spec({"type": "switch", "switch": "split", "destinations": destinations})

    def test_malformed_destination_raises(self):
        with pytest.raises(ValueError, match="unpack"):
            Tile.from_spec({"type": "switch", "switch": "split", "destinations": [[1, 2, 3], [4, 5]]})

    def test_split_with_targets_raises(self):
        with pytest.raises(ValueError, match="không điều khiển cầu"):
            Tile.from_spec({"type": "switch", "switch": "split",
                            "targets": ["b1"], "destinations": [[1, 2], [3, 4]]})

    def test_split_with_effect_raises(self):
        with pytest.raises(ValueError, match="không có effect"):
            Tile.from_spec({"type": "switch", "switch": "split",
                            "effect": "toggle", "destinations": [[1, 2], [3, 4]]})

class TestTileConstructorValidation:
    @pytest.mark.parametrize("extra", [
        {"switch_kind": SwitchKind.SOFT},
        {"effect": SwitchEffect.TOGGLE},
        {"targets": ("b1",)},
        {"permanent_state": PermanentState.OPEN},
    ], ids=["switch_kind", "effect", "targets", "permanent_state"])
    def test_non_switch_with_switch_fields_raises(self, extra):
        with pytest.raises(ValueError, match="switch không hợp lệ"):
            Tile(TileType.FLOOR, **extra)

    def test_switch_without_kind_raises(self):
        with pytest.raises(ValueError, match="thông tin về loại"):
            Tile(TileType.SWITCH, effect=SwitchEffect.TOGGLE, targets=("b1",))

    def test_switch_without_effect_raises(self):
        with pytest.raises(ValueError, match="effect"):
            Tile(TileType.SWITCH, switch_kind=SwitchKind.SOFT, targets=("b1",))

    def test_permanent_without_state_raises(self):
        with pytest.raises(ValueError, match="phải có permanent_state"):
            Tile(TileType.SWITCH, switch_kind=SwitchKind.SOFT,
                 effect=SwitchEffect.PERMANENT, targets=("b1",))

    def test_toggle_with_permanent_state_raises(self):
        with pytest.raises(ValueError, match="chỉ hợp lệ với switch PERMANENT"):
            Tile(TileType.SWITCH, switch_kind=SwitchKind.SOFT, effect=SwitchEffect.TOGGLE,
                 permanent_state=PermanentState.OPEN, targets=("b1",))

# from_spec với đầu vào sai schema.
class TestTileSchemaErrors:
    @pytest.mark.parametrize("spec", [
        {"type": "lava"},
        {"type": "switch", "switch": "medium", "targets": ["b1"]},
        {"type": "switch", "switch": "soft", "effect": "explode", "targets": ["b1"]},
    ], ids=["bad_type", "bad_switch_kind", "bad_effect"])
    def test_unknown_enum_raises(self, spec):
        with pytest.raises(ValueError, match="is not a valid"):
            Tile.from_spec(spec)

    def test_missing_type_raises_key_error(self):
        # from_spec đọc spec["type"] trực tiếp
        with pytest.raises(KeyError):
            Tile.from_spec({"id": "x"})

    def test_non_mapping_raises_type_error(self):
        with pytest.raises(TypeError):
            Tile.from_spec("floor")

    def test_metadata_non_dict_raises(self):
        # from_spec gọi metadata.items() mà không kiểm tra kiểu
        with pytest.raises(AttributeError):
            Tile.from_spec({"type": "floor", "metadata": "not-a-dict"})

# Board tổng hợp bridge/switch và kiểm tra tính hợp lệ của sàn chơi.
class TestBoardValidation:
    def test_bridge_cells_with_same_id_form_one_group(self):
        board = _simple_board([{"type": "bridge", "id": "a"}, {"type": "bridge", "id": "a"}])
        assert board.bridge_ids == ("a",)

    def test_bridge_ids_sorted_and_mask_bit_follows_order(self):
        # Sắp xếp alpha để vị trí bit ổn định: a = bit 0, b = bit 1
        board = _simple_board([{"type": "bridge", "id": "b"}, {"type": "bridge", "id": "a"}],
                              initial_closed_bridges=["b"])
        assert board.bridge_ids == ("a", "b")
        assert board.bridge_bit == {"a": 0b01, "b": 0b10}
        assert board.initial_bridge_mask == 0b10

    @pytest.mark.parametrize("closed, match", [
        (["ghost"], "không tồn tại"),
        (["a", "a"], "ID trùng"),
    ], ids=["unknown_id", "duplicate_id"])
    def test_invalid_initial_closed_bridges_raises(self, closed, match):
        with pytest.raises(ValueError, match=match):
            _simple_board([{"type": "bridge", "id": "a"}], initial_closed_bridges=closed)

    def test_switch_target_nonexistent_bridge_raises(self):
        with pytest.raises(ValueError, match="không có trong sàn chơi"):
            _simple_board([{"type": "switch", "targets": ["ghost"]}])

    @pytest.mark.parametrize("dest_tile, destinations, match", [
        ({"type": "floor"}, [[0, 99], [0, 0]], "nằm ngoài sàn chơi"),
        ({"type": "void"}, [[0, 1], [0, 0]], "nằm trên ô trống"),
        ({"type": "fragile"}, [[0, 1], [0, 0]], "dễ vỡ"),
    ], ids=["out_of_bounds", "void", "fragile"])
    def test_invalid_split_destination_raises(self, dest_tile, destinations, match):
        with pytest.raises(ValueError, match=match):
            _simple_board([dest_tile, {"type": "switch", "switch": "split",
                                       "destinations": destinations}])

# Stage loader.
class TestStageLoader:
    @pytest.mark.parametrize("make_source", [
        lambda: MOCK_PATH,
        lambda: str(MOCK_PATH),
        lambda: MOCK_PATH.read_text(encoding="utf-8"),
        lambda: json.loads(MOCK_PATH.read_text(encoding="utf-8")),
    ], ids=["path", "str_path", "json_text", "mapping"])
    def test_load_from_sources(self, make_source):
        _assert_mock_board(load_stage(make_source()))

    def test_parses_tiles(self, mock_board):
        assert mock_board.tile_at((0, 0)).kind == TileType.VOID
        assert mock_board.tile_at((2, 3)).kind == TileType.FRAGILE
        assert mock_board.tile_at((2, 4)).tile_id == "bridge_toggle"
        assert mock_board.tile_at((2, 1)).targets == ("bridge_toggle",)
        assert mock_board.tile_at((1, 4)).permanent_state == PermanentState.OPEN
        heavy = mock_board.tile_at((3, 2))
        assert heavy.switch_kind == SwitchKind.HEAVY
        assert heavy.permanent_state == PermanentState.OPEN
        assert mock_board.tile_at((3, 6)).destinations == ((1, 2), (3, 5))

    def test_defaults(self, mock_data):
        del mock_data["name"]
        del mock_data["initial_closed_bridges"]
        board = load_stage(mock_data)
        assert board.name == "Untitled"
        assert board.initial_bridge_mask == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_stage(tmp_path / "missing.json")

    def test_invalid_json_text_raises(self):
        with pytest.raises(ValueError, match="JSON hợp lệ"):
            load_stage("{not valid json")

    def test_top_level_not_object_raises(self, tmp_path):
        path = tmp_path / "stage.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="object JSON"):
            load_stage(path)

    def test_unsupported_source_type_raises(self):
        with pytest.raises(TypeError):
            load_stage(b"{}")

    @pytest.mark.parametrize("changes, match", [
        ({"tiles": []}, r"tiles\[\]"),
        ({"tiles": "not a list"}, r"tiles\[\]"),
        ({"tiles": [[]]}, "ít nhất 1 ô"),
        ({"start": [2]}, r"\[row, column\]"),
        ({"start": [9, 9]}, "nằm ngoài sàn chơi"),
        ({"start": [0, 0]}, "ô trống"),
        ({"initial_closed_bridges": "bridge_a"}, "danh sách bridge ID"),
    ], ids=["empty_tiles", "tiles_not_list", "empty_row", "start_wrong_length",
            "start_out_of_bounds", "start_on_void", "closed_bridges_not_list"])
    def test_invalid_fields_raise(self, mock_data, changes, match):
        with pytest.raises(ValueError, match=match):
            load_stage({**mock_data, **changes})

    def test_missing_start_raises_key_error(self, mock_data):
        # Board.from_dict đọc data["start"] trực tiếp
        del mock_data["start"]
        with pytest.raises(KeyError):
            load_stage(mock_data)

    def test_ragged_rows_raise(self, mock_data):
        mock_data["tiles"][1] = mock_data["tiles"][1][:-1]
        with pytest.raises(ValueError, match="cùng số ô"):
            load_stage(mock_data)

    @pytest.mark.parametrize("cell, tile", [
        ((2, 6), {"type": "floor"}),
        ((1, 1), {"type": "goal"}),
    ], ids=["no_goal", "two_goals"])
    def test_goal_count_must_be_one(self, mock_data, cell, tile):
        mock_data["tiles"][cell[0]][cell[1]] = tile
        with pytest.raises(ValueError, match="đúng 1 ô goal"):
            load_stage(mock_data)

# Hành vi trên tile
class TestFragileTile:
    def test_is_supported(self):
        board = _simple_board([{"type": "fragile"}])
        assert not is_supported(board, State(0, 1))
        assert is_supported(board, State(0, 0, Orientation.HORIZONTAL))

    def test_rolling_to_stand_on_fragile_returns_none(self, mock_board):
        # Nằm ngang (2,1)-(2,2) lăn RIGHT -> đứng tại (2,3) fragile
        assert get_next_state(mock_board, State(2, 1, Orientation.HORIZONTAL), Action.MOVE_RIGHT) is None

class TestBridgeTile:
    @pytest.mark.parametrize("orientation", [Orientation.STANDING, Orientation.HORIZONTAL],
                             ids=["standing", "lying_half"])
    @pytest.mark.parametrize("closed", [True, False], ids=["closed", "open"])
    def test_is_supported(self, mock_board, orientation, closed):
        # Nằm ngang tại (2,4) chiếm bridge_toggle (2,4) + floor (2,5)
        mask = mock_board.bridge_bit["bridge_toggle"] if closed else 0
        assert is_supported(mock_board, State(2, 4, orientation, bridge_mask=mask)) is not closed

    def test_rolling_onto_closed_bridge_returns_none(self, mock_board):
        # Nằm ngang (1,4)-(1,5) lăn DOWN -> (2,4) bridge_toggle đang đóng
        state = State(1, 4, Orientation.HORIZONTAL, bridge_mask=mock_board.bridge_bit["bridge_toggle"])
        assert get_next_state(mock_board, state, Action.MOVE_DOWN) is None

class TestSwitchTile:
    @pytest.mark.parametrize("state", [
        State(0, 1, bridge_mask=0b1),
        State(0, 0, Orientation.HORIZONTAL, bridge_mask=0b1),
        State(0, 1, Orientation.HORIZONTAL, bridge_mask=0b1),
    ], ids=["standing", "lying_right_half", "lying_left_half"])
    def test_soft_triggered_by_any_orientation(self, state):
        assert apply_switches(_switch_board("soft"), state, _BEFORE).bridge_mask == 0b0

    @pytest.mark.parametrize("state, expected", [
        (State(0, 1, bridge_mask=0b1), 0b0),
        (State(0, 0, Orientation.HORIZONTAL, bridge_mask=0b1), 0b1),
        (State(0, 1, Orientation.HORIZONTAL, bridge_mask=0b1), 0b1),
    ], ids=["standing", "lying_right_half", "lying_left_half"])
    def test_heavy_triggered_only_when_standing(self, state, expected):
        assert apply_switches(_switch_board("heavy"), state, _BEFORE).bridge_mask == expected

    def test_only_toggles_its_target_bridge(self):
        # Bridge 'a' là bit 0, 'b' là bit 1; switch chỉ nhắm 'b'
        board = _simple_board([{"type": "switch", "targets": ["b"]},
                               {"type": "bridge", "id": "a"}, {"type": "bridge", "id": "b"}])
        assert apply_switches(board, State(0, 1, bridge_mask=0b11), _BEFORE).bridge_mask == 0b01

    def test_toggle_closes_open_bridge(self):
        # Chiều mở cầu đóng đã có ở các test kích switch phía trên
        board = _switch_board("soft", closed=False)
        assert apply_switches(board, State(0, 1), _BEFORE).bridge_mask == 0b1

    @pytest.mark.parametrize("mask", [0b1, 0b0], ids=["closed", "already_open"])
    def test_permanent_open_always_opens(self, mask):
        board = _switch_board("soft", effect="permanent", set_to="open")
        assert apply_switches(board, State(0, 1, bridge_mask=mask), _BEFORE).bridge_mask == 0b0

    @pytest.mark.parametrize("mask", [0b0, 0b1], ids=["open", "already_closed"])
    def test_permanent_close_always_closes(self, mask):
        board = _switch_board("soft", effect="permanent", closed=False, set_to="close")
        assert apply_switches(board, State(0, 1, bridge_mask=mask), _BEFORE).bridge_mask == 0b1

    def test_switch_closing_bridge_under_block_returns_none(self):
        # Nằm lên switch (0,1) + bridge mở (0,2); switch toggle đóng cầu ngay dưới block
        board = _simple_board([{"type": "switch", "targets": ["b"]}, {"type": "bridge", "id": "b"}])
        assert get_next_state(board, State(0, 0), Action.MOVE_RIGHT) is None

class TestSplitSwitchTile:
    def test_standing_on_split_switch_splits_block(self):
        # Nằm ngang (1,1)-(1,2) lăn LEFT -> đứng trên split (1,0) -> 2 cube tại (0,3), (2,3)
        board = _split_board()
        state = State(1, 1, Orientation.HORIZONTAL, bridge_mask=0b1)
        assert get_next_state(board, state, Action.MOVE_LEFT) == State.split(
            (0, 3), (2, 3), bridge_mask=0b1)

    def test_lying_on_split_switch_does_not_split(self):
        # Đứng (1,2) lăn LEFT -> nằm (1,0)-(1,1), nửa trái trên split
        board = _split_board()
        assert get_next_state(board, State(1, 2, bridge_mask=0b1), Action.MOVE_LEFT) == State(
            1, 0, Orientation.HORIZONTAL, bridge_mask=0b1)

    def test_split_destination_on_closed_bridge_returns_none(self):
        # Đích (0,3) là bridge 'b' đang đóng -> cube rơi
        board = _split_line_board({"type": "bridge", "id": "b"})
        state = State(0, 2, Orientation.HORIZONTAL, bridge_mask=0b1)
        assert get_next_state(board, state, Action.MOVE_LEFT) is None

    def test_split_destination_on_soft_switch_triggers_it(self):
        board = _split_line_board({"type": "switch", "targets": ["b"]},
                                  extra=[{"type": "bridge", "id": "b"}])
        state = State(0, 2, Orientation.HORIZONTAL, bridge_mask=0b1)
        assert get_next_state(board, state, Action.MOVE_LEFT) == State.split((0, 0), (0, 3))

    def test_single_cube_does_not_break_fragile(self):
        state = State.split((0, 3), (2, 3), bridge_mask=0b1)
        next_state = get_next_state(_split_board(), state, Move(Action.MOVE_LEFT, (0, 3)))
        assert next_state is not None
        assert next_state.split_positions == ((0, 2), (2, 3))

    @pytest.mark.parametrize("positions, selected, expected_mask", [
        (((0, 3), (2, 0)), (0, 3), 0b0),
        (((0, 0), (2, 3)), (2, 3), 0b1),
    ], ids=["soft_triggered", "heavy_not_triggered"])
    def test_single_cube_on_switch(self, positions, selected, expected_mask):
        # Cube đang điều khiển lăn RIGHT lên soft (0,4) hoặc heavy (2,4)
        state = State(0, 0, bridge_mask=0b1, split_positions=positions)
        assert get_next_state(_split_board(), state, Move(Action.MOVE_RIGHT, selected)).bridge_mask == expected_mask

    def test_resting_cube_on_soft_switch_is_not_triggered_again(self):
        # Cube (0,4) đứng sẵn trên soft, cube (2,0) di chuyển -> cầu giữ nguyên
        state = State(0, 0, bridge_mask=0b1, split_positions=((0, 4), (2, 0)))
        assert get_next_state(_split_board(), state, Move(Action.MOVE_RIGHT, (2, 0))).bridge_mask == 0b1

class TestMockStage:
    """Tích hợp fragile + switch + bridge trên màn mock."""

    def test_solution_reaches_goal(self, mock_board):
        # Switch toggle đổi trạng thái bridge_toggle hai lần; switch_open mở
        # bridge_closed_1. Cầu bridge_closed_2 vẫn đóng khi block đứng trên goal.
        moves = [Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_UP,
                 Action.MOVE_RIGHT, Action.MOVE_RIGHT, Action.MOVE_DOWN, Action.MOVE_RIGHT]
        state = mock_board.initial_state
        for action in moves:
            state = get_next_state(mock_board, state, action)
            assert state is not None
        assert state == State(2, 6, bridge_mask=0b010)
        assert is_goal(mock_board, state)
