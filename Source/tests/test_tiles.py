import json
from pathlib import Path

import pytest

from game import (Board, Direction, Orientation, State, SwitchEffect, SwitchKind,
                  Tile, TileType, get_next_state, get_successors, is_goal, load_stage)
from game.rules import apply_switches, is_supported

MOCK_PATH = Path(__file__).parent / "data" / "mock_data.json"

# Board 1 hàng: floor + middle_tiles + goal, start ở (0, 0)
def _simple_board(middle_tiles, **kwargs):
    row = [{"type": "floor"}] + middle_tiles + [{"type": "goal"}]
    return Board.from_dict({"start": [0, 0], "tiles": [row], **kwargs})

# Board 1 hàng: floor | switch | floor | bridge 'b' | goal.
def _switch_board(switch, effect="toggle", closed=True, metadata=None):
    spec = {"type": "switch", "switch": switch, "effect": effect, "targets": ["b"]}
    if metadata:
        spec["metadata"] = metadata
    return _simple_board([spec, {"type": "floor"}, {"type": "bridge", "id": "b"}],
                         initial_closed_bridges=["b"] if closed else [])

def _assert_mock_board(board):
    assert board.name == "Mẫu màn chơi: Cầu và công tắc"
    assert (board.height, board.width) == (5, 7)
    assert board.start == (2, 0)
    assert board.goal == (2, 6)
    assert board.bridge_ids == ("bridge_a",)
    assert board.switch_ids == ("switch_a",)
    assert board.initial_closed_bridges == ("bridge_a",)
    assert board.initial_bridge_mask == 0b1
    assert board.initial_state == State(2, 0, Orientation.STANDING, bridge_mask=0b1)

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
        assert tile.targets == ()
        assert tile.destinations == ()

# Giá trị mặc định và override của allows_standing
class TestAllowsStanding:
    @pytest.mark.parametrize("spec, expected", [
        ({"type": "floor"}, True),
        ({"type": "goal"}, True),
        ({"type": "fragile"}, False),
        ({"type": "floor", "allows_standing": False}, False),
    ], ids=["floor_default", "goal_default", "fragile_default", "floor_override"])
    def test_allows_standing(self, spec, expected):
        assert Tile.from_spec(spec).allows_standing is expected

# tile_id, ràng buộc id của bridge và chuẩn hoá metadata
class TestTileIdAndMetadata:
    @pytest.mark.parametrize("spec, expected", [
        ({"type": "bridge", "id": "bridge_a"}, "bridge_a"),
        ({"type": "floor", "id": "f1"}, "f1"),
        ({"type": "switch", "targets": ["b1"]}, None),
    ], ids=["bridge", "floor", "switch_without_id"])
    def test_tile_id(self, spec, expected):
        assert Tile.from_spec(spec).tile_id == expected

    @pytest.mark.parametrize("spec", [
        {"type": "bridge"},
        {"type": "bridge", "id": ""},
        {"type": "bridge", "id": None},
    ], ids=["missing", "empty_string", "none"])
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

    @pytest.mark.parametrize("spec, switch_kind", [
        ({"type": "switch", "targets": ["b1"]}, SwitchKind.SOFT),
        ({"type": "switch", "switch": "heavy", "targets": ["b1"]}, SwitchKind.HEAVY),
    ], ids=["no_switch_field", "heavy"])
    def test_defaults(self, spec, switch_kind):
        tile = Tile.from_spec(spec)
        assert tile.switch_kind == switch_kind
        assert tile.effect == SwitchEffect.TOGGLE

    @pytest.mark.parametrize("switch_name", ["soft", "heavy"])
    @pytest.mark.parametrize("extra", [{}, {"targets": []}], ids=["no_targets", "empty_targets"])
    def test_missing_targets_raises(self, switch_name, extra):
        with pytest.raises(ValueError, match="cầu kết nối"):
            Tile.from_spec({"type": "switch", "switch": switch_name, **extra})

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
        [],
        [[1, 2]],
        [[1, 2], [3, 4], [5, 6]],
    ], ids=["zero", "one", "three"])
    def test_wrong_destination_count_raises(self, destinations):
        with pytest.raises(ValueError, match="đúng 2 tọa độ"):
            Tile.from_spec({"type": "switch", "switch": "split", "destinations": destinations})

    @pytest.mark.parametrize("destinations", [
        [[1, 2, 3], [4, 5]],
        [[1], [4, 5]],
    ], ids=["too_many_coords", "too_few_coords"])
    def test_malformed_destination_raises(self, destinations):
        with pytest.raises(ValueError, match="unpack"):
            Tile.from_spec({"type": "switch", "switch": "split", "destinations": destinations})

    def test_split_with_targets_raises(self):
        with pytest.raises(ValueError, match="không điều khiển cầu"):
            Tile.from_spec({"type": "switch", "switch": "split",
                            "targets": ["b1"], "destinations": [[1, 2], [3, 4]]})

    @pytest.mark.parametrize("effect_name", ["toggle", "permanent"])
    def test_split_with_effect_raises(self, effect_name):
        with pytest.raises(ValueError, match="không có effect"):
            Tile.from_spec({"type": "switch", "switch": "split",
                            "effect": effect_name, "destinations": [[1, 2], [3, 4]]})

class TestTileConstructorValidation:
    @pytest.mark.parametrize("kind", [
        TileType.FLOOR, TileType.VOID, TileType.GOAL, TileType.FRAGILE, TileType.BRIDGE,
    ], ids=["floor", "void", "goal", "fragile", "bridge"])
    @pytest.mark.parametrize("extra", [
        {"switch_kind": SwitchKind.SOFT},
        {"effect": SwitchEffect.TOGGLE},
        {"targets": ("b1",)},
    ], ids=["switch_kind", "effect", "targets"])
    def test_non_switch_with_switch_fields_raises(self, kind, extra):
        with pytest.raises(ValueError, match="switch không hợp lệ"):
            Tile(kind, tile_id="x", **extra)

    def test_switch_without_kind_raises(self):
        with pytest.raises(ValueError, match="thông tin về loại"):
            Tile(TileType.SWITCH, effect=SwitchEffect.TOGGLE, targets=("b1",))

    @pytest.mark.parametrize("switch_kind", [SwitchKind.SOFT, SwitchKind.HEAVY], ids=["soft", "heavy"])
    def test_switch_without_effect_raises(self, switch_kind):
        with pytest.raises(ValueError, match="effect"):
            Tile(TileType.SWITCH, switch_kind=switch_kind, targets=("b1",))

# from_spec với đầu vào sai schema.
class TestTileSchemaErrors:
    @pytest.mark.parametrize("spec", [
        {"type": "lava"},
        {"type": "FLOOR"},
        {"type": ""},
        {"type": "switch", "switch": "medium", "targets": ["b1"]},
        {"type": "switch", "switch": "soft", "effect": "explode", "targets": ["b1"]},
    ], ids=["unknown_type", "uppercase", "empty_type", "bad_switch_kind", "bad_effect"])
    def test_unknown_enum_raises(self, spec):
        with pytest.raises(ValueError, match="is not a valid"):
            Tile.from_spec(spec)

    def test_missing_type_raises_key_error(self):
        # from_spec đọc spec["type"] trực tiếp
        with pytest.raises(KeyError):
            Tile.from_spec({"id": "x"})

    @pytest.mark.parametrize("spec", [None, "floor", ["type", "floor"], 42],
                             ids=["none", "string", "list", "int"])
    def test_non_mapping_raises_type_error(self, spec):
        with pytest.raises(TypeError):
            Tile.from_spec(spec)

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
        assert board.initial_bridge_mask == 0b10

    def test_default_all_bridges_open(self):
        assert _simple_board([{"type": "bridge", "id": "a"}]).initial_bridge_mask == 0

    def test_tile_at_out_of_bounds_returns_void(self):
        assert _simple_board([]).tile_at((99, 99)).kind == TileType.VOID

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
        assert mock_board.tile_at((2, 4)).tile_id == "bridge_a"
        switch = mock_board.tile_at((2, 1))
        assert switch.kind == TileType.SWITCH
        assert switch.targets == ("bridge_a",)

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

    @pytest.mark.parametrize("content, match", [
        ('{"tiles": [', "JSON hợp lệ"),
        ("[1, 2, 3]", "object JSON"),
    ], ids=["broken_json", "top_level_array"])
    def test_invalid_json_file_raises(self, tmp_path, content, match):
        path = tmp_path / "stage.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_stage(path)

    @pytest.mark.parametrize("source", [None, 42, [1, 2], b"{}"],
                             ids=["none", "int", "list", "bytes"])
    def test_unsupported_source_type_raises(self, source):
        with pytest.raises(TypeError):
            load_stage(source)

    @pytest.mark.parametrize("changes, match", [
        ({"tiles": []}, r"tiles\[\]"),
        ({"tiles": "not a list"}, r"tiles\[\]"),
        ({"tiles": [[]]}, "ít nhất 1 ô"),
        ({"start": [2]}, r"\[row, column\]"),
        ({"start": [2, 0, 1]}, r"\[row, column\]"),
        ({"start": [9, 9]}, "nằm ngoài sàn chơi"),
        ({"start": [0, 0]}, "ô trống"),
        ({"initial_closed_bridges": "bridge_a"}, "danh sách bridge ID"),
    ], ids=["empty_tiles", "tiles_not_list", "empty_row", "start_too_short",
            "start_too_long", "start_out_of_bounds", "start_on_void", "closed_bridges_not_list"])
    def test_invalid_fields_raise(self, mock_data, changes, match):
        with pytest.raises(ValueError, match=match):
            load_stage({**mock_data, **changes})

    def test_missing_tiles_raises(self, mock_data):
        del mock_data["tiles"]
        with pytest.raises(ValueError, match=r"tiles\[\]"):
            load_stage(mock_data)

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

    def test_floor_disallowing_standing_behaves_like_fragile(self):
        board = _simple_board([{"type": "floor", "allows_standing": False}])
        assert not is_supported(board, State(0, 1))
        assert is_supported(board, State(0, 0, Orientation.HORIZONTAL))

    def test_rolling_to_stand_on_fragile_returns_none(self, mock_board):
        # Nằm ngang (2,1)-(2,2) lăn RIGHT -> đứng tại (2,3) fragile
        assert get_next_state(mock_board, State(2, 1, Orientation.HORIZONTAL), Direction.RIGHT) is None

    def test_rolling_to_lie_on_fragile_is_allowed(self):
        # Đứng tại goal (0,3) lăn LEFT -> nằm trên 2 ô fragile (0,1)-(0,2)
        board = _simple_board([{"type": "fragile"}, {"type": "fragile"}])
        assert get_next_state(board, State(0, 3), Direction.LEFT) == State(0, 1, Orientation.HORIZONTAL)

class TestBridgeTile:
    @pytest.mark.parametrize("orientation, mask, expected", [
        (Orientation.STANDING, 0b1, False),
        (Orientation.STANDING, 0b0, True),
        (Orientation.HORIZONTAL, 0b1, False),
        (Orientation.HORIZONTAL, 0b0, True),
    ], ids=["standing_closed", "standing_open", "lying_half_closed", "lying_half_open"])
    def test_is_supported(self, mock_board, orientation, mask, expected):
        # Nằm ngang tại (2,4) chiếm bridge (2,4) + floor (2,5)
        assert is_supported(mock_board, State(2, 4, orientation, bridge_mask=mask)) is expected

    @pytest.mark.parametrize("mask, expected", [
        (0b1, None),
        (0b0, State(2, 4, Orientation.HORIZONTAL, bridge_mask=0b0)),
    ], ids=["closed", "open"])
    def test_rolling_onto_bridge(self, mock_board, mask, expected):
        # Nằm ngang (1,4)-(1,5) lăn DOWN -> (2,4) bridge + (2,5) floor
        state = State(1, 4, Orientation.HORIZONTAL, bridge_mask=mask)
        assert get_next_state(mock_board, state, Direction.DOWN) == expected

class TestSwitchTile:
    @pytest.mark.parametrize("state", [
        State(0, 1, bridge_mask=0b1),
        State(0, 0, Orientation.HORIZONTAL, bridge_mask=0b1),
        State(0, 1, Orientation.HORIZONTAL, bridge_mask=0b1),
    ], ids=["standing", "lying_right_half", "lying_left_half"])
    def test_soft_triggered_by_any_orientation(self, state):
        assert apply_switches(_switch_board("soft"), state).bridge_mask == 0b0

    @pytest.mark.parametrize("state, expected", [
        (State(0, 1, bridge_mask=0b1), 0b0),
        (State(0, 0, Orientation.HORIZONTAL, bridge_mask=0b1), 0b1),
        (State(0, 1, Orientation.HORIZONTAL, bridge_mask=0b1), 0b1),
    ], ids=["standing", "lying_right_half", "lying_left_half"])
    def test_heavy_triggered_only_when_standing(self, state, expected):
        assert apply_switches(_switch_board("heavy"), state).bridge_mask == expected

    def test_heavy_not_triggered_when_rolling_to_lie_on_it(self):
        # Đứng (0,2) lăn LEFT -> nằm (0,0)-(0,1), nửa phải trên heavy switch
        board = _switch_board("heavy")
        state = State(0, 2, bridge_mask=0b1)
        assert get_next_state(board, state, Direction.LEFT) == State(
            0, 0, Orientation.HORIZONTAL, bridge_mask=0b1)

    def test_only_toggles_its_target_bridge(self):
        # Bridge 'a' là bit 0, 'b' là bit 1; switch chỉ nhắm 'b'
        board = _simple_board([{"type": "switch", "targets": ["b"]},
                               {"type": "bridge", "id": "a"}, {"type": "bridge", "id": "b"}])
        assert apply_switches(board, State(0, 1, bridge_mask=0b11)).bridge_mask == 0b01

    def test_no_switch_under_block_keeps_state(self, mock_board):
        state = State(2, 2, bridge_mask=0b1)
        assert apply_switches(mock_board, state) is state

    @pytest.mark.parametrize("switch", ["soft", "heavy"])
    @pytest.mark.parametrize("mask, expected", [(0b1, 0b0), (0b0, 0b1)], ids=["open", "close"])
    def test_toggle_flips_bridge(self, switch, mask, expected):
        board = _switch_board(switch)
        assert apply_switches(board, State(0, 1, bridge_mask=mask)).bridge_mask == expected

    @pytest.mark.parametrize("switch", ["soft", "heavy"])
    @pytest.mark.parametrize("mask", [0b1, 0b0], ids=["closed", "already_open"])
    def test_permanent_always_opens(self, switch, mask):
        board = _switch_board(switch, effect="permanent")
        assert apply_switches(board, State(0, 1, bridge_mask=mask)).bridge_mask == 0b0

    @pytest.mark.parametrize("mask", [0b0, 0b1], ids=["open", "already_closed"])
    def test_permanent_with_closed_state_always_closes(self, mask):
        board = _switch_board("soft", effect="permanent", closed=False, metadata={"state": "closed"})
        assert apply_switches(board, State(0, 1, bridge_mask=mask)).bridge_mask == 0b1

    def test_switch_closing_bridge_under_block_returns_none(self):
        # Nằm lên switch (0,1) + bridge mở (0,2); switch toggle đóng cầu ngay dưới block
        board = _simple_board([{"type": "switch", "targets": ["b"]}, {"type": "bridge", "id": "b"}])
        assert get_next_state(board, State(0, 0), Direction.RIGHT) is None

class TestMockStage:
    """Tích hợp fragile + switch + bridge trên màn mock."""

    def test_successors_from_start(self, mock_board):
        # UP/DOWN rơi xuống void, LEFT ra ngoài biên; RIGHT nằm lên switch_a và mở bridge_a
        assert get_successors(mock_board, mock_board.initial_state) == (
            (Direction.RIGHT, State(2, 1, Orientation.HORIZONTAL, bridge_mask=0b0)),
        )

    def test_solution_reaches_goal(self, mock_board):
        # Kích switch_a mở bridge_a, vòng qua fragile, lăn qua cầu và đứng trên goal
        moves = [Direction.RIGHT, Direction.UP, Direction.RIGHT,
                 Direction.RIGHT, Direction.DOWN, Direction.RIGHT]
        state = mock_board.initial_state
        for direction in moves:
            state = get_next_state(mock_board, state, direction)
            assert state is not None
        assert state == State(2, 6, bridge_mask=0b0)
        assert is_goal(mock_board, state)