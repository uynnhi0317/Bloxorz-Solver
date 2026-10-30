"""Cấu hình và các fixture pytest dùng chung cho các bài kiểm thử solver"""

from pathlib import Path

import pytest

from game.board import Board
from game.stage_loader import load_stage


STAGES_DIR = Path(__file__).resolve().parent.parent / "stages"


#  Helper functions tạo board mẫu

def make_row_board(num_floors: int = 2) -> Board:
    """
    Tạo board 1 hàng: start (0,0) + floors + goal.
    Để Bloxorz đứng được ở goal trên hành lang 1D, khoảng cách từ start
    đến goal phải là bội số của 3:
    start (0,0) -> lăn 1: nằm ngang chiếm (0,1)-(0,2) -> lăn 2: đứng ở (0,3).
    Mặc định num_floors=2 tạo board độ dài 4: [floor(start), floor, floor, goal].
    """
    tiles = [{"type": "floor"}] + [{"type": "floor"} for _ in range(num_floors)] + [{"type": "goal"}]
    return Board.from_dict({"start": [0, 0], "tiles": [tiles]})


def make_unsolvable_board() -> Board:
    """Board 3x3 vô nghiệm: goal bị void bao quanh hoàn toàn."""
    return Board.from_dict({
        "start": [0, 0],
        "tiles": [
            [{"type": "floor"}, {"type": "void"}, {"type": "void"}],
            [{"type": "void"}, {"type": "void"}, {"type": "void"}],
            [{"type": "void"}, {"type": "void"}, {"type": "goal"}],
        ],
    })


def make_switch_inline_board() -> Board:
    """
    Board 1 hàng 7 ô có switch và cầu đóng:
    (0,0): start
    (0,1): floor
    (0,2): switch soft toggle điều khiển bridge 'b'
    (0,3): floor
    (0,4): floor
    (0,5): bridge 'b' (đóng sẵn)
    (0,6): goal
    Nước đi: RIGHT (nằm đè 1-2 kích switch mở cầu) -> RIGHT (đứng 3) -> RIGHT (nằm 4-5) -> RIGHT (đứng 6 goal).
    """
    return Board.from_dict({
        "start": [0, 0],
        "initial_closed_bridges": ["b"],
        "tiles": [[
            {"type": "floor"},
            {"type": "floor"},
            {"type": "switch", "targets": ["b"]},
            {"type": "floor"},
            {"type": "floor"},
            {"type": "bridge", "id": "b"},
            {"type": "goal"},
        ]],
    })


def make_split_required_board() -> Board:
    """
    Board yêu cầu bắt buộc phải dùng split switch và SWITCH_CONTROL để giải:
    - Block đi tới split switch và tách thành 2 cube con tại (1,0) và (3,0).
    - Cube 1 tại (1,0) bị chặn bởi cầu đóng br1.
    - Cube 2 tại (3,0) phải di chuyển sang (3,1) để bấm switch mở cầu br1.
    - Hai cube sau đó di chuyển gặp nhau ở ô kết nối (1,5) để gộp lại thành block 1x2 rồi đứng vào goal (1,6).
    """
    F = {"type": "floor"}
    V = {"type": "void"}
    G = {"type": "goal"}
    SW_OPEN = {"type": "switch", "targets": ["br1"]}
    BR1 = {"type": "bridge", "id": "br1"}
    SPLIT = {"type": "switch", "switch": "split", "destinations": [[1, 0], [3, 0]]}

    return Board.from_dict({
        "start": [0, 0],
        "initial_closed_bridges": ["br1"],
        "tiles": [
            [F, F, F, SPLIT, V, V, V],
            [F, BR1, F, F, F, F, G],
            [V, V, V, V, V, F, V],
            [F, SW_OPEN, F, F, F, F, V],
        ],
    })


# Fixtures nạp 15 Stages thật từ thư mục stages/
# --- Simple (1-3) ---
@pytest.fixture(scope="session")
def stage_01() -> Board:
    return load_stage(STAGES_DIR / "simple" / "stage_01.json")

@pytest.fixture(scope="session")
def stage_02() -> Board:
    return load_stage(STAGES_DIR / "simple" / "stage_02.json")

@pytest.fixture(scope="session")
def stage_03() -> Board:
    return load_stage(STAGES_DIR / "simple" / "stage_03.json")

# --- Fragile (4-6) ---
@pytest.fixture(scope="session")
def stage_04() -> Board:
    return load_stage(STAGES_DIR / "fragile" / "stage_04.json")

@pytest.fixture(scope="session")
def stage_05() -> Board:
    return load_stage(STAGES_DIR / "fragile" / "stage_05.json")

@pytest.fixture(scope="session")
def stage_06() -> Board:
    return load_stage(STAGES_DIR / "fragile" / "stage_06.json")

# --- Switch (7-10) ---
@pytest.fixture(scope="session")
def stage_07() -> Board:
    return load_stage(STAGES_DIR / "switch" / "stage_07.json")

@pytest.fixture(scope="session")
def stage_08() -> Board:
    return load_stage(STAGES_DIR / "switch" / "stage_08.json")

@pytest.fixture(scope="session")
def stage_09() -> Board:
    return load_stage(STAGES_DIR / "switch" / "stage_09.json")

@pytest.fixture(scope="session")
def stage_10() -> Board:
    return load_stage(STAGES_DIR / "switch" / "stage_10.json")

# --- Combined (11-15) ---
@pytest.fixture(scope="session")
def stage_11() -> Board:
    return load_stage(STAGES_DIR / "combined" / "stage_11.json")

@pytest.fixture(scope="session")
def stage_12() -> Board:
    return load_stage(STAGES_DIR / "combined" / "stage_12.json")

@pytest.fixture(scope="session")
def stage_13() -> Board:
    return load_stage(STAGES_DIR / "combined" / "stage_13.json")

@pytest.fixture(scope="session")
def stage_14() -> Board:
    return load_stage(STAGES_DIR / "combined" / "stage_14.json")

@pytest.fixture(scope="session")
def stage_15() -> Board:
    return load_stage(STAGES_DIR / "combined" / "stage_15.json")

