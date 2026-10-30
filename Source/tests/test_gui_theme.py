"""Kiểm thử phần giao diện theme Sakura (chạy không cửa sổ, SDL dummy).

Trọng tâm: lớp hiển thị phải đồng bộ với logic và không làm sai lệch kết quả của solver.
"""
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from game import Action, Board, Move, Orientation, State, get_next_state, is_goal, load_stage
from gui.animations import BridgeTransitions, SwitchPulses, triggered_switches
from gui.app import App, PlaySession, discover_stages
from gui.renderer import Renderer
from gui.theme import Background
from solvers import AStarSolver, BFSSolver, DFSSolver, UCSSolver

STAGES = discover_stages()
SOLVERS = {"BFS": BFSSolver, "DFS": DFSSolver, "UCS": UCSSolver, "A*": AStarSolver}
AREA = pygame.Rect(0, 60, 1000, 640)

F, V, G = {"type": "floor"}, {"type": "void"}, {"type": "goal"}
BRIDGE = {"type": "bridge", "id": "b"}
SOFT = {"type": "switch", "switch": "soft", "targets": ["b"]}
HEAVY = {"type": "switch", "switch": "heavy", "targets": ["b"]}
SPLIT = {"type": "switch", "switch": "split", "destinations": [[0, 0], [2, 4]]}


@pytest.fixture(scope="module", autouse=True)
def display():
    pygame.init()
    pygame.display.set_mode((1200, 760))
    yield   # không pygame.quit(): font đã cache sẽ thành đối tượng chết với các module test chạy sau


@pytest.fixture
def surface():
    return pygame.Surface((1200, 760))


def run_until_idle(session: PlaySession, surface, dt: float = 40, limit: int = 3000) -> None:
    """Chạy session tới khi hết animation + hết lời giải, có vẽ từng frame để bắt lỗi runtime."""
    for _ in range(limit):
        session.update(dt)
        session.draw(surface)
        if session.anim is None and not session.replaying and not session.flips.items and not session.pulses.items:
            return
    raise AssertionError("session không dừng")


def make_board(tiles, start=(1, 0), closed=()) -> Board:
    return Board.from_dict({"name": "t", "start": list(start), "initial_closed_bridges": list(closed), "tiles": tiles})


# ------------------------------------------------------------------ hiển thị đồng bộ với logic
@pytest.mark.parametrize("path", STAGES, ids=lambda p: p.stem)
def test_replay_matches_rules_on_every_stage(path: Path, surface):
    """Xem lại lời giải có animation phải đi đúng từng trạng thái mà game.rules tính ra và kết thúc ở goal."""
    board = load_stage(path)
    moves = AStarSolver().solve(board).moves
    expected = [board.initial_state]
    for m in moves:
        expected.append(get_next_state(board, expected[-1], m))
    assert is_goal(board, expected[-1])

    session = PlaySession(board, Renderer(board, AREA), 0)
    session.start_replay(list(moves))
    seen = [session.state]
    for _ in range(20000):
        session.update(40)
        session.draw(surface)
        if session.state != seen[-1]:
            seen.append(session.state)
        if session.status != "playing":
            break
    assert session.status == "won"
    assert seen == expected
    assert session.moves == len(moves)


@pytest.mark.parametrize("name", SOLVERS)
def test_app_runs_each_solver_and_reports_status(name):
    """Nút Solve chạy đúng solver: cùng lời giải với gọi trực tiếp, trạng thái running -> done, xem lại về đích."""
    app = App()
    for key, cls in SOLVERS.items():
        solver = cls()
        app.register_solver(key, lambda b, s, sv=solver: sv.solve(_Started(b, s)))
    app.start_stage(0)
    app._set_panel(True)
    for _ in range(200):
        app.update(40)
        app.draw()
        if app.session.anim is None:
            break
    direct = SOLVERS[name]().solve(app.session.board).moves
    app.run_solver(name)
    assert app.solve_buttons[name].status == "running"
    for _ in range(20000):
        app.update(40)
        app.draw()
        if app.solve_done:
            break
    assert app.solve_done
    assert app.solve_buttons[name].status == "done"
    assert app.last_stats[1].moves == direct


class _Started:
    def __init__(self, board, start):
        self._board, self.initial_state = board, start

    def __getattr__(self, name):
        return getattr(self._board, name)


def test_failed_solver_is_shown_as_failed():
    app = App()
    app.register_solver("BFS", lambda b, s: BFSSolver().solve(make_unsolvable()))
    app.start_stage(0)
    for _ in range(200):
        app.update(40)
        if app.session.anim is None:
            break
    app.run_solver("BFS")
    for _ in range(2000):
        app.update(40)
        app.draw()
        if not app.solving:
            break
    assert app.solve_buttons["BFS"].status == "failed"
    assert app.session.status == "playing"


def make_unsolvable() -> Board:
    return make_board([[F, V, G], [F, V, V]], start=(0, 0))


def test_restart_and_new_stage_reset_everything(surface):
    app = App()
    app.start_stage(0)
    s = app.session
    for _ in range(200):
        app.update(40)
        if s.anim is None:
            break
    assert s.manual_action(Action.MOVE_RIGHT) or s.manual_action(Action.MOVE_DOWN)
    assert s.moves == 1
    app.restart()
    assert app.session.state == app.session.board.initial_state
    assert app.session.moves == 0
    assert app.session.renderer.head_cell is None
    app.start_stage(1)
    assert app.session.index == 1


def test_new_game_button_starts_first_stage():
    app = App()
    app.tutorial_seen = True
    app.main_menu.selected = app.main_menu.items.index("New Game")
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="", mod=0))
    assert app.scene == "play"
    assert app.session.index == 0


# ------------------------------------------------------------------ cầu
def test_bridge_transition_lasts_about_300ms_and_follows_logic(surface):
    board = make_board([[V, V, V, V], [SOFT, F, BRIDGE, G], [V, V, V, V]], closed=("b",))
    closed_mask = board.initial_bridge_mask
    r = Renderer(board, AREA)
    flips = BridgeTransitions(r)

    flips.start(closed_mask, 0)                # mở: ván đi từ dưới nước (0) lên mặt cầu (1)
    flips.update(0)
    assert flips.fx()[(1, 2)].plank == pytest.approx(0.0)
    flips.update(150)
    mid = flips.fx()[(1, 2)].plank
    assert 0 < mid < 1
    r.ambient_fx = flips.fx()
    r.draw_scene(surface, 0, [])               # vẽ được ở giữa quá trình chuyển
    flips.update(150)
    assert flips.fx() == {}                    # xong trong 0,3 giây

    flips.start(0, closed_mask)                # đóng: ngược lại
    flips.update(0)
    assert flips.fx()[(1, 2)].plank == pytest.approx(1.0)
    flips.update(300)
    assert flips.fx() == {}


def test_bridge_animation_starts_when_state_changes(surface):
    board = make_board([[F, SOFT, F, BRIDGE, F, G]], start=(0, 0), closed=("b",))
    session = PlaySession(board, Renderer(board, AREA), 0)
    logical = get_next_state(board, board.initial_state, Action.MOVE_RIGHT)
    assert logical.bridge_mask != board.initial_bridge_mask
    assert session.try_action(Move(Action.MOVE_RIGHT))
    assert not session.flips.items             # chưa đổi khi animation lăn còn chạy
    for _ in range(100):
        session.update(16)
        session.draw(surface)
        if session.anim is None:
            break
    assert session.state == logical            # state hiển thị = state logic
    assert session.flips.items                 # cầu bắt đầu chuyển đúng lúc state đổi
    assert session.pulses.items                # công tắc phát sáng cùng lúc


# ------------------------------------------------------------------ công tắc
def test_triggered_switches_follow_game_rules():
    board = make_board([[F, F, HEAVY, F], [SOFT, F, F, F], [F, BRIDGE, G, F]], start=(0, 0))
    mask = board.initial_bridge_mask
    # nằm đè lên soft -> kích hoạt
    before = State(0, 0, Orientation.STANDING, mask)
    after = get_next_state(board, before, Action.MOVE_DOWN)
    assert triggered_switches(board, before, after, Move(Action.MOVE_DOWN)) == [(1, 0)]
    # nằm đè lên heavy -> KHÔNG kích hoạt
    before = State(0, 3, Orientation.STANDING, mask)
    after = get_next_state(board, before, Action.MOVE_LEFT)
    assert (0, 2) in after.occupied_cells and after.bridge_mask == mask
    assert triggered_switches(board, before, after, Move(Action.MOVE_LEFT)) == []
    # đứng thẳng lên heavy -> kích hoạt
    before = State(0, 0, Orientation.HORIZONTAL, mask)
    after = get_next_state(board, before, Action.MOVE_RIGHT)
    assert after.position == (0, 2) and after.bridge_mask != mask
    assert triggered_switches(board, before, after, Move(Action.MOVE_RIGHT)) == [(0, 2)]


def test_split_switch_is_reported_when_block_splits():
    board = make_board([[F, F, F, F, F], [F, SPLIT, F, F, F], [F, F, G, F, F]], start=(1, 2))
    before = State(1, 2, Orientation.HORIZONTAL)
    after = get_next_state(board, before, Action.MOVE_LEFT)
    assert after.split_positions is not None
    assert triggered_switches(board, before, after, Move(Action.MOVE_LEFT)) == [(1, 1)]


def test_switch_active_tracks_bridge_mask():
    board = make_board([[V, V, V], [SOFT, BRIDGE, G], [V, V, V]], start=(1, 0), closed=("b",))
    r = Renderer(board, AREA)
    tile = board.tile_at((1, 0))
    assert r._switch_active(tile, board.initial_bridge_mask) is False
    assert r._switch_active(tile, 0) is True


def test_pulses_finish_and_do_not_touch_state(surface):
    board = make_board([[F, F], [SOFT, G], [BRIDGE, F]], start=(0, 0))
    r = Renderer(board, AREA)
    pulses = SwitchPulses(r)
    r.pulses = pulses
    state = board.initial_state
    pulses.trigger((1, 0))
    r.draw(surface, state)
    pulses.update(800)
    assert pulses.items == []
    assert board.initial_state == state


# ------------------------------------------------------------------ khối tách
def test_split_pieces_are_head_and_body_and_keep_identity(surface):
    split = {"type": "switch", "switch": "split", "destinations": [[0, 4], [2, 0]]}
    board = make_board([[F, F, F, F, F], [F, split, F, F, F], [F, F, G, F, F]], start=(1, 2))
    session = PlaySession(board, Renderer(board, AREA), 0)
    session.state = State(1, 2, Orientation.HORIZONTAL)
    assert session.try_action(Move(Action.MOVE_LEFT))   # dẫm công tắc ( ): tách khối
    for _ in range(100):
        session.update(16)
        session.draw(surface)
        if session.anim is None:
            break
    assert session.state.split_positions == ((0, 4), (2, 0))
    boxes = session.renderer.state_boxes(session.state)
    assert sorted(b.figure for b in boxes) == ["body", "head"]
    assert sum(b.active for b in boxes) == 1
    assert session.renderer.head_cell == (0, 4)
    # đi phần đầu xuống hàng dưới: thứ tự chuẩn hóa trong split_positions đảo, hình vẫn phải đi theo cube
    for _ in range(2):
        assert session.try_action(Move(Action.MOVE_DOWN, session.selected))
        for _ in range(60):
            session.update(16)
            session.draw(surface)
            if session.anim is None:
                break
    assert session.state.split_positions == ((2, 0), (2, 4))
    head = next(b for b in session.renderer.state_boxes(session.state) if b.figure == "head")
    assert tuple(head.verts[0, :2]) == (2, 4)


def test_loss_animation_restores_status(surface):
    board = make_board([[F, F, V], [F, G, V]], start=(0, 0))
    session = PlaySession(board, Renderer(board, AREA), 0)
    assert session.try_action(Move(Action.MOVE_UP))    # rơi khỏi bàn
    run_until_idle(session, surface)
    assert session.status == "lost"
    assert session.state == board.initial_state        # không đổi trạng thái logic khi thua


# ------------------------------------------------------------------ nền, asset
def test_background_petal_count_and_motion(surface):
    bg = Background((1200, 760))
    total = len(bg.falling) + len(bg.floating)
    assert 30 <= total <= 50
    before = [p[1] for p in bg.falling]
    bg.update(500)
    bg.draw(surface)
    assert before != [p[1] for p in bg.falling]
    speeds = {round(p[2], 4) for p in bg.falling}
    assert len(speeds) > 5


def test_missing_thumbnail_falls_back_without_crash():
    app = App()
    assert app._thumb(9999, (100, 100)) is None
    app.scene = "select"
    app.draw()


@pytest.mark.parametrize("size", [(1024, 576), (1228, 691), (1536, 864), (1920, 1080)])
def test_all_scenes_draw_at_common_window_sizes(size):
    pygame.display.set_mode(size)
    app = App()
    app.screen = pygame.display.set_mode(size)
    for scene in ("menu", "select", "credits"):
        app.scene = scene
        app.update(16)
        app.draw()
    app.start_stage(len(app.stages) - 1)
    app._set_panel(True)
    for _ in range(5):
        app.update(16)
        app.draw()
    for rect in app.main_menu.rects:   # menu không tràn khỏi màn hình
        assert rect.bottom <= size[1]
