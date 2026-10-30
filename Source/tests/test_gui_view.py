"""Kiểm thử camera, nhân vật Vô Diện (vật rắn), mặt nước, tỉ lệ hình học và bố cục Solver Panel."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from game import Action, Board, Move, Orientation, State, get_next_state
from gui import tile_art
from gui.animations import RollAnimation, merged_rotation, roll_matrix
from gui.app import App, PlaySession
from gui.camera import MAX_ZOOM, MIN_ZOOM, PITCH, YAW, Camera, view_matrix
from gui.controls import _status_icon
from gui.mesh_render import IDENTITY
from gui.noface import build_model, figure_mesh
from gui.renderer import Renderer, figure_pose, make_box
from gui.theme import PALETTE, Background
from solvers import AStarSolver, BFSSolver

AREA = pygame.Rect(0, 60, 1000, 640)
F, V, G = {"type": "floor"}, {"type": "void"}, {"type": "goal"}
OPEN_BOARD = {"start": [3, 3], "tiles": [[F] * 8 for _ in range(7)] + [[F] * 7 + [G]]}


@pytest.fixture(scope="module", autouse=True)
def display():
    pygame.init()
    pygame.display.set_mode((1200, 760))
    yield


@pytest.fixture
def surface():
    return pygame.Surface((1200, 760))


def open_session() -> PlaySession:
    board = Board.from_dict(OPEN_BOARD)
    return PlaySession(board, Renderer(board, AREA), 0)


def land(session, surface, limit=200):
    for _ in range(limit):
        session.update(16)
        session.draw(surface)
        if session.anim is None:
            return
    raise AssertionError("animation không kết thúc")


def figure_world(box) -> np.ndarray:
    """Các đỉnh Vô Diện trong thế giới, tính đúng như renderer."""
    rot, origin = figure_pose(box)
    return origin + figure_mesh(box.figure).verts @ rot.T


def head_apex(box) -> np.ndarray:
    """Đỉnh đầu: điểm có z cục bộ lớn nhất."""
    mesh = figure_mesh(box.figure)
    return figure_world(box)[int(np.argmax(mesh.verts[:, 2]))]


# ------------------------------------------------------------------ Vô Diện là vật rắn
@pytest.mark.parametrize("action", list(Action))
def test_figure_is_continuous_and_rigid_through_a_roll(action, surface):
    """Lỗi cũ: giữa cú lăn hình nhảy sang tư thế khác. Nay từng frame phải nối liền và không biến dạng."""
    session = open_session()
    r = session.renderer
    anim = RollAnimation(r, session.state, action)
    start = figure_world(anim.current_box())
    previous = start
    shape = np.linalg.norm(start - start.mean(axis=0), axis=1)
    while not anim.done:
        anim.update(4)
        now = figure_world(anim.current_box())
        assert np.linalg.norm(now - previous, axis=1).max() < 0.25          # không có cú nhảy
        assert np.allclose(np.linalg.norm(now - now.mean(axis=0), axis=1), shape, atol=1e-6)   # không méo
        previous = now
    assert np.allclose(previous, figure_world(anim.end_box), atol=1e-6)     # frame cuối = tư thế sau khi đáp


def test_head_stays_where_the_roll_carried_it(surface):
    """Sau mỗi nước, hình vẽ lúc đứng yên phải trùng khớp frame cuối của animation lăn."""
    session = open_session()
    for action in (Action.MOVE_RIGHT, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_UP, Action.MOVE_RIGHT, Action.MOVE_RIGHT):
        anim = RollAnimation(session.renderer, session.state, action)
        assert session.try_action(Move(action))
        land(session, surface)
        rest = session.renderer.state_boxes(session.state)[0]
        assert np.allclose(figure_world(anim.end_box), figure_world(rest), atol=1e-6)


def test_rolling_back_restores_original_figure(surface):
    session = open_session()
    original = figure_world(session.renderer.state_boxes(session.state)[0])
    for action in (Action.MOVE_RIGHT, Action.MOVE_LEFT):
        assert session.try_action(Move(action))
        land(session, surface)
    assert session.state == session.board.initial_state
    assert np.allclose(figure_world(session.renderer.state_boxes(session.state)[0]), original, atol=1e-6)


def test_four_quarter_turns_are_identity():
    for action in Action:
        assert np.allclose(np.linalg.matrix_power(roll_matrix(action), 4), np.eye(3))
        assert np.allclose(roll_matrix(action) @ roll_matrix(action).T, np.eye(3))
        assert np.linalg.det(roll_matrix(action)) == pytest.approx(1.0)


def test_figure_orientation_is_not_part_of_game_state(surface):
    """Hướng hình chỉ là trạng thái hiển thị: state vẫn đúng như game.rules tính."""
    session = open_session()
    expected = session.state
    for action in (Action.MOVE_RIGHT, Action.MOVE_RIGHT, Action.MOVE_DOWN):
        expected = get_next_state(session.board, expected, action)
        assert session.try_action(Move(action))
        land(session, surface)
        assert session.state == expected
    assert not hasattr(session.state, "figure_rot")


def test_step_back_restores_figure_exactly(surface):
    session = open_session()
    moves = [Move(Action.MOVE_RIGHT), Move(Action.MOVE_DOWN), Move(Action.MOVE_LEFT)]
    session.start_replay(moves)
    session.toggle_replay_pause()
    snapshots = [session.renderer.figure_rot]
    for _ in moves:
        assert session.step_forward()
        land(session, surface)
        snapshots.append(session.renderer.figure_rot)
    for expected in reversed(snapshots[:-1]):
        assert session.step_back()
        assert np.allclose(session.renderer.figure_rot, expected)


def split_setup():
    tiles = [[F] * 5 for _ in range(4)]
    tiles[3][4] = G
    board = Board.from_dict({"start": [0, 0], "tiles": tiles})
    r = Renderer(board, AREA)
    r.selected, r.head_cell = (1, 1), (1, 1)
    return board, r, State.split((1, 1), (3, 0))


@pytest.mark.parametrize("action", list(Action))
def test_split_cube_rolls_rigidly_and_lands_on_the_floor(action):
    """Cube 1x1x1 lăn thật: có xoay, từng frame nối liền, không xuyên sàn, đáp đúng ô kề và chạm sàn."""
    board, r, state = split_setup()
    anim = RollAnimation(r, state, action, moving=0)
    previous = figure_world(anim.current_box())
    shape = np.linalg.norm(previous - previous.mean(axis=0), axis=1)
    assert previous[:, 2].min() == pytest.approx(0.0, abs=0.01)
    while not anim.done:
        anim.update(4)
        now = figure_world(anim.current_box())
        assert np.linalg.norm(now - previous, axis=1).max() < 0.2
        assert np.allclose(np.linalg.norm(now - now.mean(axis=0), axis=1), shape, atol=1e-6)
        assert now[:, 2].min() > -0.01
        previous = now
    assert not np.allclose(anim.end_box.rot, IDENTITY)        # đã xoay thật, không phải nhảy
    assert np.allclose(previous, figure_world(anim.end_box), atol=1e-6)
    assert previous[:, 2].min() == pytest.approx(0.0, abs=0.01)
    dr, dc = {Action.MOVE_UP: (-1, 0), Action.MOVE_DOWN: (1, 0), Action.MOVE_LEFT: (0, -1), Action.MOVE_RIGHT: (0, 1)}[action]
    centre = figure_pose(anim.end_box)[1][:2]
    assert np.allclose(centre, (1.5 + dr, 1.5 + dc), atol=1e-6)


def test_each_cube_keeps_its_own_orientation(surface):
    tiles = [[F] * 6 for _ in range(4)]
    tiles[3][5] = G
    board = Board.from_dict({"start": [0, 0], "tiles": tiles})
    session = PlaySession(board, Renderer(board, AREA), 0)
    r = session.renderer
    session.state = State.split((0, 0), (3, 0))
    r.head_cell = (0, 0)
    session._set_selected((0, 0))
    assert session.try_action(Move(Action.MOVE_RIGHT, (0, 0)))   # chỉ phần đầu lăn
    land(session, surface)
    assert np.allclose(r.cube_rots["head"], roll_matrix(Action.MOVE_RIGHT))
    assert np.allclose(r.cube_rots["body"], IDENTITY)
    assert session.try_action(Move(Action.MOVE_LEFT, (0, 1)))    # lăn ngược lại: trở về như cũ
    land(session, surface)
    assert np.allclose(r.cube_rots["head"], IDENTITY)


def test_step_back_restores_cube_orientation(surface):
    tiles = [[F] * 6 for _ in range(4)]
    tiles[3][5] = G
    board = Board.from_dict({"start": [0, 0], "tiles": tiles})
    session = PlaySession(board, Renderer(board, AREA), 0)
    session.state = State.split((0, 0), (3, 0))
    session.renderer.head_cell = (0, 0)
    session.start_replay([Move(Action.MOVE_RIGHT, (0, 0))])
    session.toggle_replay_pause()
    assert session.step_forward()
    land(session, surface)
    assert session.step_back()
    assert np.allclose(session.renderer.cube_rots["head"], IDENTITY)


def test_merged_rotation_points_head_toward_head_cube():
    for head, other in (((1, 2), (1, 1)), ((1, 1), (1, 2)), ((2, 1), (1, 1)), ((1, 1), (2, 1))):
        rot = merged_rotation(head, other)
        assert np.allclose(rot @ rot.T, np.eye(3)) and np.linalg.det(rot) == pytest.approx(1.0)
        assert np.allclose(rot @ (0, 0, 1), (head[0] - other[0], head[1] - other[1], 0))


def test_mask_is_part_of_the_rigid_mesh():
    mesh = figure_mesh("whole")
    assert mesh.layer.max() > 0                        # mặt nạ dán sau thân
    top = mesh.layer == mesh.layer.max()               # lớp trên cùng là chi tiết mặt nạ
    assert (mesh.normals[top] @ (1, 0, 0) > 0.99).all()   # và nằm phẳng trên mặt trước


@pytest.mark.parametrize("kind,size", [("whole", (0.96, 0.96, 1.96)), ("head", (0.68, 0.68, 0.68)), ("body", (0.68, 0.68, 0.68))])
def test_figure_keeps_box_dimensions(kind, size):
    mesh = figure_mesh(kind)
    extent = mesh.verts.max(axis=0) - mesh.verts.min(axis=0)
    assert np.allclose(extent, size, atol=0.02)


@pytest.mark.parametrize("kind", ["whole", "head", "body"])
def test_figure_is_a_square_edged_box(kind):
    """Mọi mặt thân đều phẳng và vuông góc với trục: không còn góc bo."""
    mesh = figure_mesh(kind)
    assert np.allclose(np.abs(mesh.normals[mesh.layer == 0]).max(axis=1), 1.0)
    verts = mesh.verts
    half = (verts[:, 0].max() - verts[:, 0].min()) / 2
    top = verts[:, 2].max()
    assert (np.abs(verts[:, 0]) + np.abs(verts[:, 1]))[verts[:, 2] > top - 0.005].max() == pytest.approx(2 * half, abs=0.01)


@pytest.mark.parametrize("kind", ["whole", "head"])
def test_mask_is_an_axis_aligned_rectangle(kind):
    model = build_model(kind).model
    masks = [f for f in model.faces if f.material == "mask"]
    assert len(masks) == 1 and len(masks[0].vertices) == 4
    pts = model.vertices[list(masks[0].vertices)]
    assert len({round(v, 6) for v in pts[:, 1]}) == 2 and len({round(v, 6) for v in pts[:, 2]}) == 2


def test_body_cube_has_no_face_features():
    """Thân dưới khi tách: có cloak + edge (viền cạnh), nhưng không mặt nạ."""
    materials = {f.material for f in build_model("body").model.faces}
    assert "mask" not in materials and "eye" not in materials
    assert materials == {"cloak", "edge"}


# ------------------------------------------------------------------ cầu và ánh sáng sàn
def test_bridge_has_no_railing_above_the_deck(surface):
    """Thân cầu không có gì nhô lên khỏi mặt sàn."""
    bridge = {"type": "bridge", "id": "b"}
    board = Board.from_dict({"start": [0, 0], "tiles": [[F, bridge, bridge, F, G]]})
    r = Renderer(board, AREA)
    plain = pygame.Surface((400, 400), pygame.SRCALPHA)
    r._draw_bridge(plain, 0, 1, 1.0)
    above = r.project([(0.0, 1.0, 0.02), (0.0, 2.0, 0.02)])
    # hàng điểm ảnh ngay trên mép xa của mặt cầu phải trống hoàn toàn
    y = int(above[:, 1].min()) - 3
    x0, x1 = int(above[:, 0].min()), int(above[:, 0].max())
    if 0 <= y < 400:
        assert all(plain.get_at((x, y))[3] == 0 for x in range(max(0, x0), min(400, x1)))
    assert isinstance(r.bridge_axis[(0, 1)], int)


def test_floor_is_brightest_under_the_spotlight_and_changes_smoothly():
    board = Board.from_dict(OPEN_BOARD)
    r = Renderer(board, AREA)
    centre = r._spot_gain(board.height / 2, board.width / 2)
    edge = r._spot_gain(0, 0)
    assert centre > edge > 1.0                       # chỗ tối nhất vẫn sáng hơn màu gốc
    # hai ô kề nhau dùng chung giá trị tại cạnh chung nên không có đường ranh
    assert abs(r._spot_gain(2, 3) - r._spot_gain(2.001, 3)) < 0.01


def test_textured_quad_applies_corner_gain(surface):
    from gui.raster import textured_quad
    from gui.resources import Texture
    import numpy as _np
    tex = Texture(None, _np.full((4, 4, 3), 100, dtype=_np.uint8), (100.0, 100.0, 100.0))
    s = pygame.Surface((64, 64), pygame.SRCALPHA)
    textured_quad(s, (0, 0), (64, 0), (0, 64), tex, (0, 0), (1, 1), (100, 100, 100), (1.0, 2.0, 2.0, 1.0))
    assert s.get_at((60, 30))[0] > s.get_at((3, 30))[0] + 50
    assert s.get_at((3, 30))[3] == 255


# ------------------------------------------------------------------ camera
def test_view_angle_is_fixed_and_zoom_is_limited():
    cam = Camera()
    assert np.allclose(cam.matrix(), view_matrix(20.0, 45.0)) and (YAW, PITCH) == (20.0, 45.0)
    assert not hasattr(cam, "rotate")
    cam.zoom_by(1000)
    assert cam.zoom == MAX_ZOOM
    cam.zoom_by(-1000)
    assert cam.zoom == MIN_ZOOM
    assert np.allclose(cam.matrix(), view_matrix(20.0, 45.0))   # zoom không đổi góc
    cam.reset()
    assert cam.zoom == 1.0


def test_zoom_scales_uniformly():
    cam = Camera()
    board = Board.from_dict(OPEN_BOARD)
    r = Renderer(board, AREA, camera=cam)
    base = r.scale
    cam.zoom_by(3)
    r.apply_camera()
    assert r.scale == pytest.approx(base * cam.zoom)


def app_in_stage(stage=0):
    app = App()
    app.register_solver("A*", lambda b, s: AStarSolver().solve(_Started(b, s)))
    app.register_solver("BFS", lambda b, s: BFSSolver().solve(_Started(b, s)))
    app.start_stage(stage)
    while app.session.anim is not None:
        app.update(40)
    return app


class _Started:
    def __init__(self, board, start):
        self._board, self.initial_state = board, start

    def __getattr__(self, name):
        return getattr(self._board, name)


def post(app, kind, **kw):
    app.handle_event(pygame.event.Event(kind, **kw))


def test_mouse_drag_does_not_rotate_and_wheel_zooms_without_touching_game(monkeypatch):
    app = app_in_stage()
    state, moves = app.session.state, app.session.moves
    x, y = app.board_area().center
    before = app.session.renderer._m.copy()
    post(app, pygame.MOUSEBUTTONDOWN, button=3, pos=(x, y))
    post(app, pygame.MOUSEMOTION, pos=(x + 50, y + 20), rel=(50, 20), buttons=(0, 0, 1))
    post(app, pygame.MOUSEBUTTONUP, button=3, pos=(x + 50, y + 20))
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (x, y))
    post(app, pygame.MOUSEWHEEL, x=0, y=2, flipped=False)
    app.update(16)
    app.draw()
    assert np.allclose(app.session.renderer._m, before)
    assert app.camera.zoom > 1.0
    assert app.session.state == state and app.session.moves == moves
    post(app, pygame.KEYDOWN, key=pygame.K_c, unicode="c", mod=0)
    assert app.camera.zoom == 1.0


def test_zooming_during_solver_replay_does_not_change_the_solution():
    app = app_in_stage()
    app._set_panel(True)
    direct = AStarSolver().solve(app.session.board).moves
    app.run_solver("A*")
    seen_fast = False
    for i in range(20000):
        if i % 3 == 0:
            app.camera.zoom_by(0.05 if i % 2 else -0.05)
        app.update(40)
        app.draw()
        seen_fast = seen_fast or app.session.renderer.fast
        if app.solve_done:
            break
    assert app.solve_done and seen_fast
    assert app.last_stats[1].moves == direct


def test_high_quality_returns_after_zoom_stops():
    app = app_in_stage()
    app.camera.zoom_by(1)
    app.update(16)
    app.draw()
    assert app.session.renderer.fast
    for _ in range(20):
        app.update(16)
    assert not app.session.renderer.fast
    app.draw()
    assert app.session.renderer._layers        # đã dựng lại ảnh sàn


def test_fast_mode_draws_every_stage_and_transition(surface):
    from gui.app import discover_stages
    from game import load_stage
    for path in discover_stages():
        board = load_stage(path)
        r = Renderer(board, AREA)
        r.fast = True
        r.draw(surface, board.initial_state)


# ------------------------------------------------------------------ tỉ lệ hình học
@pytest.mark.parametrize("size", [(1024, 576), (1920, 1080), (900, 1000), (1600, 500)])
def test_tiles_are_not_stretched_by_window_shape(size):
    """Một hệ số scale chung: tỉ lệ cạnh ô chỉ phụ thuộc camera, không phụ thuộc hình dạng cửa sổ."""
    board = Board.from_dict(OPEN_BOARD)
    r = Renderer(board, pygame.Rect(0, 0, *size))
    p = r.project([(0, 0, 0), (0, 1, 0), (1, 0, 0)])
    width = np.linalg.norm(p[1] - p[0])
    depth = np.linalg.norm(p[2] - p[0])
    m = view_matrix(YAW, PITCH)
    assert depth / width == pytest.approx(np.linalg.norm(m[:, 0]) / np.linalg.norm(m[:, 1]))


def test_cube_edges_keep_unit_length_when_rotated():
    box = make_box(0, 0, 0, 1, 1, 2)
    anim_box = RollAnimation(open_session().renderer, open_session().state, Action.MOVE_LEFT)
    anim_box.update(80)
    v = anim_box.current_box().verts
    assert np.linalg.norm(v[1] - v[0]) == pytest.approx(np.linalg.norm(box[1] - box[0]))


# ------------------------------------------------------------------ ô đích
def test_goal_tile_has_a_dark_hole():
    board = Board.from_dict(OPEN_BOARD)
    r = Renderer(board, AREA)
    gr, gc = board.goal
    img, (x0, y0) = r._static_layer(board.initial_bridge_mask)
    x, y = r.project([(gr + 0.5, gc + 0.5, 0.0)])[0]
    centre = img.get_at((int(x) - x0, int(y) - y0))
    assert sum(centre[:3]) < 0.8 * sum(PALETTE["goal"])


def test_goal_swirl_keeps_turning():
    board = Board.from_dict(OPEN_BOARD)
    r = Renderer(board, AREA)
    a, b = pygame.Surface((1000, 700)), pygame.Surface((1000, 700))
    r._draw_vortex(a, {})
    r.t += 300
    r._draw_vortex(b, {})
    pa, pb = pygame.surfarray.array3d(a), pygame.surfarray.array3d(b)
    assert pa.max() > 0 and (pa != pb).any()


def test_no_gate_stands_on_the_goal_tile(monkeypatch):
    board = Board.from_dict({"start": [0, 0], "tiles": [[F, F, G]]})
    r = Renderer(board, AREA)
    monkeypatch.setattr(r, "_draw_underwater", lambda *a: None)
    s = pygame.Surface((1000, 760))
    r.draw_scene(s, board.initial_bridge_mask, [])
    gr, gc = board.goal
    for z in (0.6, 1.0, 1.4):
        x, y = r.project([(gr + 0.5, gc + 0.5, z)])[0]
        assert s.get_at((int(x), int(y)))[:3] == (0, 0, 0)


def test_solver_done_shows_no_tick_but_failure_still_shows_x():
    a, b = pygame.Surface((40, 40)), pygame.Surface((40, 40))
    _status_icon(a, "done", 20, 20, 8)
    _status_icon(b, "failed", 20, 20, 8)
    assert pygame.surfarray.array3d(a).max() == 0
    assert pygame.surfarray.array3d(b).max() > 0


def test_select_scene_dims_the_background():
    app = App()
    app.scene = "select"
    app.draw()
    dimmed = app.screen.get_at((4, app.screen.get_height() // 2))[:3]
    app.scene = "credits"
    app.background.draw(app.screen)
    plain = app.screen.get_at((4, app.screen.get_height() // 2))[:3]
    assert sum(dimmed) < 0.6 * sum(plain)


# ------------------------------------------------------------------ mặt nước
def test_water_is_transparent_and_clearer_up_close():
    bg = Background((800, 600))
    a = pygame.surfarray.pixels_alpha(bg.sheet)
    near, far = int(a[10, -1]), int(a[10, 0])
    del a
    assert 0 < near < far < 255


def test_objects_under_the_water_show_through():
    bg = Background((800, 600))
    plain, marked = pygame.Surface((800, 600)), pygame.Surface((800, 600))
    bg.draw_under(plain)
    bg.draw_under(marked)
    marked.fill((255, 0, 0), (390, 500, 40, 40))     # vật màu đỏ nằm dưới nước
    bg.draw_water(plain)
    bg.draw_water(marked)
    a, b = plain.get_at((410, 520)), marked.get_at((410, 520))
    assert b[0] > a[0] + 40                          # nhìn xuyên thấy vật
    assert b[0] < 255 or b[1] > 0                    # nhưng có lớp nước phủ


def test_closed_bridge_is_drawn_under_the_water_pass(surface):
    bridge = {"type": "bridge", "id": "b"}
    board = Board.from_dict({"start": [0, 0], "initial_closed_bridges": ["b"], "tiles": [[F, F, bridge, F, G]]})
    r = Renderer(board, AREA)
    calls = []
    r.water_pass = lambda s: calls.append(pygame.surfarray.array3d(s).copy())
    r.draw(surface, board.initial_state)
    assert len(calls) == 1
    x, y = (int(v) for v in r.project([(0.5, 2.5, -1.0)])[0])
    assert calls[0][x, y].tolist() != [0, 0, 0]      # ván chìm đã vẽ trước khi phủ nước


def test_no_supports_are_drawn_under_the_board(monkeypatch):
    bridge = {"type": "bridge", "id": "b"}
    board = Board.from_dict({"start": [0, 0], "tiles": [[F, F, bridge, F, G]]})
    r = Renderer(board, AREA)
    boxes = []
    monkeypatch.setattr(r, "_draw_box", lambda *a, **k: boxes.append(a))
    r._under_layer(0)
    assert boxes == []


def test_wave_layers_wrap_seamlessly_and_add_light():
    bg = Background((800, 600))
    for layer in bg.waves:
        px = pygame.surfarray.array3d(layer).astype(int)
        assert px.max() > 40
        edge_jump = np.abs(px[0] - px[-1]).mean()
        inner_jump = np.abs(px[400] - px[399]).mean()
        assert edge_jump < inner_jump * 2 + 1        # không lộ đường nối khi cuộn ngang


def test_water_sparkles_flash_and_change_over_time():
    bg = Background((800, 600))
    a, b = pygame.Surface((800, 600)), pygame.Surface((800, 600))
    bg.draw_water(a)
    bg.update(700)
    bg.draw_water(b)
    assert (pygame.surfarray.array3d(a) != pygame.surfarray.array3d(b)).any()
    assert 0 < sum(1 for s in bg.sparkles if abs(s[0] - 0.76) < 0.15) < len(bg.sparkles)


def test_fragile_leaves_are_stable_and_stay_inside_the_tile():
    class Recorder:
        def __init__(self):
            self.polys, self.lines = [], []

        def poly(self, pts, color, z):
            self.polys.append((tuple(pts), color))

        def line(self, a, b, color, width, z):
            self.lines.append((a, b, color))

    first, second = Recorder(), Recorder()
    tile_art.fragile(first, 2, 3)
    tile_art.fragile(second, 2, 3)
    assert first.polys == second.polys and first.lines == second.lines
    assert first.lines                               # có gân lá
    assert len({c for _, c in first.polys}) > 10     # lá không cùng một màu phẳng
    pts = np.array([p for poly, _ in first.polys for p in poly])
    assert pts[:, 0].min() >= 2 and pts[:, 0].max() <= 3 + 0.02
    assert pts[:, 1].min() >= 3 and pts[:, 1].max() <= 4 + 0.02


# ------------------------------------------------------------------ chọn stage từ menu Pause
def press(app, key):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))


def open_pause_select(app):
    app.paused = True
    app.pause_menu.selected = app.pause_menu.items.index("Select Stage")
    press(app, pygame.K_RETURN)


def test_pause_menu_opens_stage_select_on_the_current_stage():
    app = app_in_stage(3)
    open_pause_select(app)
    assert app.scene == "select" and app.select_index == 3
    app.draw()


def test_escape_from_stage_select_returns_to_the_same_paused_game():
    app = app_in_stage(3)
    session = app.session
    open_pause_select(app)
    press(app, pygame.K_ESCAPE)
    assert app.scene == "play" and app.paused and app.session is session


def test_choosing_a_stage_from_pause_starts_it_unpaused():
    app = app_in_stage(3)
    open_pause_select(app)
    app.select_index = 5
    press(app, pygame.K_RETURN)
    assert app.scene == "play" and not app.paused and app.session.index == 5


def test_stage_select_from_main_menu_still_returns_to_main_menu():
    app = app_in_stage(3)
    open_pause_select(app)
    press(app, pygame.K_ESCAPE)
    app.scene = "menu"
    app.main_menu.selected = app.main_menu.items.index("Load Stage")
    press(app, pygame.K_RETURN)
    assert app.scene == "select"
    press(app, pygame.K_ESCAPE)
    assert app.scene == "menu"


@pytest.mark.parametrize("size", [(1024, 576), (1536, 864)])
def test_pause_menu_fits_in_the_window(size):
    pygame.display.set_mode(size)
    app = app_in_stage(0)
    app.screen = pygame.display.set_mode(size)
    app.paused = True
    app.draw()
    for rect in app.pause_menu.rects:
        assert pygame.Rect(0, 0, *size).contains(rect)
    pygame.display.set_mode((1200, 760))


# ------------------------------------------------------------------ Solver Panel
def overlapping(rects):
    return [(a, b) for i, a in enumerate(rects) for b in rects[i + 1:] if a.colliderect(b)]


@pytest.mark.parametrize("size", [(1024, 576), (1024, 640), (1228, 691), (1536, 864), (1920, 1080)])
def test_panel_layout_fits_and_nothing_overlaps(size):
    pygame.display.set_mode(size)
    app = app_in_stage(11)
    app.screen = pygame.display.set_mode(size)
    app.update(16)
    app._set_panel(True)
    app.run_solver("A*")
    for _ in range(20000):
        app.update(16)
        app.draw()
        if app.last_stats and app.session.has_solution:
            break
    assert app.session.has_solution
    w, h = size
    buttons = [b.rect for b in app.solve_buttons.values()] + [b.rect for b in app.replay_buttons]
    assert not overlapping(buttons)
    for rect in buttons:
        assert pygame.Rect(0, 0, w, h).contains(rect), f"{rect} tràn khỏi cửa sổ {size}"
    gaps = sorted(abs(a.y - b.y) - a.h for a in buttons for b in buttons if a.x == b.x and a.y != b.y)
    assert gaps[0] >= 8                              # các hàng nút có khoảng thở
    # các nút trong HUD không chồng lên nhau và không chồng lên panel
    hud = [app.btn_pause.rect, app.hud_info_rect] + [b.rect for b in app.hud_buttons]
    assert not overlapping(hud)
    panel_x = w - app._panel_w() - 24
    assert all(r.right <= panel_x for r in hud)
    pygame.display.set_mode((1200, 760))


def test_no_rotate_controls_left_in_the_hud():
    app = app_in_stage()
    assert not hasattr(app, "btn_view")
    assert len(app.hud_buttons) == 4
