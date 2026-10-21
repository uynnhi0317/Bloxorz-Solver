"""Renderer 2.5D.

Tọa độ trong world có dạng (row, col, z). z = 0 là mặt sàn.
Camera là một ma trận 2x3 (M)
Chiếu một điểm trong không gian 3D lên màn hình: screen = M @ (row, col, z)

Renderer CHỈ ĐỌC State/Board từ Source/game. Không đụng đến game logic (cầu, công tắc, tách khối...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame

from game import Board, Orientation, State, SwitchKind, Tile, TileType, Coord

Color = Tuple[int, int, int]

TILE_THICK = 0.3 
BLOCK_HEIGHT = 2.0  # vật cao nhất được phép vẽ (là khối đứng thẳng)

PALETTE = {
    "floor": (176, 184, 196),
    "fragile": (226, 138, 52),
    "bridge": (110, 168, 214),
    "bridge_closed_outline": (96, 106, 128),
    "goal_hole": (14, 14, 20),
    "goal_inner": (40, 40, 52),
    "soft": (70, 170, 90),
    "heavy": (210, 70, 70),
    "split": (90, 100, 230),
    "block": (178, 98, 66),
    "cube_active": (86, 164, 236),
    "cube_idle": (122, 132, 152),
}


"""
FACES: lưu lại thứ tự các đỉnh tạo thành mỗi mặt (6 mặt, mỗi mặt 4 đỉnh)
Thứ tự: sau (gồm các đỉnh có row trong tọa độ nhỏ hơn), trước, trái (col nhỏ hơn), phải, dưới (z nhỏ hơn), trên.
Hình minh họa:
        1                               3
        .==============================.
       /|                            / |
      / |                           /  |
   5 /  |                        7 /   |
    .=============================.    |             --> hand-drawn lol
    |   |_________________________|____|
    |  / 0                        |   / 2
    | /                           |  /
    |/                            | /
  4 .=============================./ 6

"""
FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
_LIGHT = np.array([0.45, 0.15, 0.88])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)

# Ma trận View (2x3). Rows = screen x and y (in tile-width units) for a world point
# (row, col, z)
VIEWS = {
    "top": ((0.0, 1.0, 0.0), (0.7, 0.0, -0.7)),
}
VIEW_YAWS = {"top": 0.0}

def make_view(yaw_deg=30, pitch_deg=45):
    """Hàm tạo bộ VIEW điều chỉnh góc camnera
    yaw_deg: độ xoay mặt bảng (yaw_deg > 0 --> góc phải trên cùng chếc lên trên)
    pitch_deg: độ dốc khi nhìn xuống từ camera (càng lớn càng cao, range : 0..90)."""

    if (pitch_deg <= 0): pitch_deg = 45
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    s, c = math.sin(yaw), math.cos(yaw)
    k, h = math.sin(pitch), math.cos(pitch)
    return ((s, c, 0.0), (k * c, -k * s, -h))


def register_view(name: str, yaw_deg: float, pitch_deg: float = 45) -> None:
    """Ghi lại bộ VIEW của góc máy mới (nếu muốn điều chỉnh góc máy)
    yaw_deg: độ xoay mặt bảng
    pitch_deg: độ dốc khi nhìn xuống từ camera"""

    VIEWS[name] = make_view(yaw_deg, pitch_deg)
    VIEW_YAWS[name] = yaw_deg


register_view("tilt", yaw_deg=20)
DEFAULT_VIEW = "tilt"

def shade(color: Color, f: float) -> Color:
    return tuple(max(0, min(255, int(ch * f))) for ch in color)  # type: ignore[return-value]


def make_box(r0: float, c0: float, z0: float, dr: float, dc: float, dz: float) -> np.ndarray:
    """Tạo mảng (8, 3) lưu tọa độ của 8 đinh của khối"""
    v = np.empty((8, 3))
    for i in range(8):
        v[i] = (r0 + dr * ((i >> 2) & 1), c0 + dc * ((i >> 1) & 1), z0 + dz * (i & 1))
    return v


def _clip_face(pts: np.ndarray, below: bool, eps: float = 1e-9):
    """Clip một mặt của khối 3D qua sàn. Cắt mặt đó thành nửa trên và nửa dưới z=0.
    """
    h = -pts[:, 2] if below else pts[:, 2]
    if (h >= -eps).all():
        return list(pts), [False] * len(pts)
    if (h < -eps).all():
        return [], []
    out, cut, n = [], [], len(pts)
    for i in range(n):
        a, b, ha, hb = pts[i], pts[(i + 1) % n], h[i], h[(i + 1) % n]
        ina, inb = ha >= -eps, hb >= -eps
        if ina:
            out.append(a)
            cut.append(False)
        if ina != inb:
            out.append(a + (b - a) * (ha / (ha - hb)))
            cut.append(ina)
    return out, cut


@dataclass
class Box:
    verts: np.ndarray
    color: Color


class Renderer:
    def __init__(self, board: Board, area: pygame.Rect, tile_width: float = 64.0,
                 view: str = DEFAULT_VIEW):
        self.board = board
        self.view_name = view
        self.set_area(area, tile_width)

    # ------------------------------------------------------------------ camera
    def set_area(self, area: pygame.Rect, tile_width: float = 64.0) -> None:
        """Canh level giữa màn hình (gọi lại nếu cửa sổ game đổi kích thước)."""
        b = self.board
        self._m = np.array(VIEWS[self.view_name], dtype=float)  # (2, 3)
        
        v = np.cross(self._m[0], self._m[1])
        self.view_dir = v if v[2] > 0 else -v
        corners = np.array([(r, c, z) for r in (0, b.height) for c in (0, b.width)
                            for z in (-TILE_THICK, BLOCK_HEIGHT)], dtype=float)
        xy = corners @ self._m.T
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        w, h = (hi - lo) * tile_width
        self.scale = tile_width * min(1.0, area.width / w, area.height / h)
        self.offset = np.array(area.center, dtype=float) - (lo + hi) / 2 * self.scale

    def project(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=float).reshape(-1, 3)
        return (p @ self._m.T) * self.scale + self.offset

    def _cell_depth(self, r: float, c: float) -> float:
        return float(r * self.view_dir[0] + c * self.view_dir[1])

    @staticmethod
    def _cells_under(box: "Box"):
        v = box.verts
        r0, r1 = int(math.floor(v[:, 0].min() + 1e-6)), int(math.ceil(v[:, 0].max() - 1e-6))
        c0, c1 = int(math.floor(v[:, 1].min() + 1e-6)), int(math.ceil(v[:, 1].max() - 1e-6))
        return [(r, c) for r in range(r0, r1) for c in range(c0, c1)]

    # ------------------------------------------------------------ public API
    def state_boxes(self, state: State, selected_cube: Optional[Coord] = None) -> List[Box]:
        if state.split_positions is not None:
            active = tuple(selected_cube) if selected_cube is not None else state.split_positions[0]
            return [
                Box(make_box(r, c, 0, 1, 1, 1),
                    PALETTE["cube_active"] if (r, c) == active else PALETTE["cube_idle"])
                for r, c in state.split_positions
            ]
        dims = {Orientation.STANDING: (1, 1, 2),
                Orientation.HORIZONTAL: (1, 2, 1),
                Orientation.VERTICAL: (2, 1, 1)}[state.orientation]
        return [Box(make_box(state.row, state.column, 0, *dims), PALETTE["block"])]

    def draw(self, surface, state, selected_cube: Optional[Coord] = None) -> None:
        if hasattr(state, "state") and hasattr(state, "selected_cube"):  # GameContext
            state, selected_cube = state.state, state.selected_cube
        self.draw_scene(surface, state.bridge_mask, self.state_boxes(state, selected_cube))

    def draw_scene(self, surface: pygame.Surface, bridge_mask: int, boxes: Sequence[Box],
                   alpha: int = 255) -> None:
        """Vẽ cảnh (box và bảng).
        Gồm 3 bước: vẽ phần box bên dưới z=0, vẽ tiles, vẽ phần box bên trên z=0
        """
        b = self.board
        order = sorted(((r, c) for r in range(b.height) for c in range(b.width)),
                       key=lambda rc: self._cell_depth(*rc))  # back to front
        self._draw_boxes(surface, boxes, alpha, below=True)
        for r, c in order:
            self._draw_tile(surface, r, c, b.tiles[r][c], bridge_mask)
        self._draw_boxes(surface, boxes, alpha, below=False)

    def _draw_boxes(self, surface: pygame.Surface, boxes: Sequence[Box], alpha: int, below: bool) -> None:
        boxes = [bx for bx in boxes if not below or bx.verts[:, 2].min() < -1e-9]
        if not boxes:
            return
        target = surface
        if alpha < 255:
            target = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for bx in sorted(boxes, key=lambda x: float(np.dot(x.verts.mean(axis=0), self.view_dir))):
            self._draw_box(target, bx.verts, bx.color, below)
        if target is not surface:
            target.set_alpha(max(0, alpha))
            surface.blit(target, (0, 0))

    # --------------------------------------------------------------- drawing
    def _draw_box(self, surface: pygame.Surface, verts: np.ndarray, color: Color,
                  below: Optional[bool] = None) -> None:
        """Vẽ khối hộp.
        below=None --> Vẽ cả khối
        below=False/True --> vẽ phần hộp nửa trên/dưới z=0
        """
        center = verts.mean(axis=0)
        faces = []
        for idx in FACES:
            pts = verts[list(idx)]
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            c = pts.mean(axis=0)
            if np.dot(n, c - center) < 0:  # make the normal point outward
                n = -n
            n = n / (np.linalg.norm(n) + 1e-9)
            faces.append((float(np.dot(c, self.view_dir)), pts, n))
        faces.sort(key=lambda f: f[0])  # painter's algorithm, far faces first
        edge = shade(color, 0.45)
        for _, pts, n in faces:
            poly, cut = (list(pts), [False] * len(pts)) if below is None else _clip_face(pts, below)
            if len(poly) < 3:
                continue
            screen = self.project(np.array(poly)).tolist()
            f = 0.55 + 0.45 * max(0.0, float(np.dot(n, _LIGHT)))
            pygame.draw.polygon(surface, shade(color, f), screen)
            for i in range(len(screen)):
                if not cut[i]:
                    pygame.draw.line(surface, edge, screen[i], screen[(i + 1) % len(screen)])

    def _flat(self, surface, cells, color, z=0.0, width=0) -> None:
        pts = self.project([(r, c, z) for r, c in cells]).tolist()
        pygame.draw.polygon(surface, color, pts, width)

    def _line(self, surface, a, b, color, width=3, z=0.0) -> None:
        p = self.project([(a[0], a[1], z), (b[0], b[1], z)]).tolist()
        pygame.draw.line(surface, color, p[0], p[1], width)

    def _draw_tile(self, surface, r: int, c: int, tile: Tile, mask: int) -> None:
        kind = tile.kind
        if kind == TileType.VOID:
            return
        if kind == TileType.BRIDGE:
            closed = (mask >> self.board.bridge_index(tile.tile_id)) & 1
            if closed:  # closed bridge = empty space, show a faint outline only
                self._flat(surface, [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)],
                           PALETTE["bridge_closed_outline"], width=1)
                return
            color = PALETTE["bridge"]
        elif kind == TileType.FRAGILE:
            color = PALETTE["fragile"]
        else:
            color = PALETTE["floor"]
            if (r + c) % 2:
                color = shade(color, 0.93)  # subtle checkerboard

        self._draw_box(surface, make_box(r, c, -TILE_THICK, 1, 1, TILE_THICK), color)
        cr, cc = r + 0.5, c + 0.5

        if kind == TileType.GOAL:
            m = 0.12
            self._flat(surface, [(r + m, c + m), (r + m, c + 1 - m), (r + 1 - m, c + 1 - m), (r + 1 - m, c + m)],
                       PALETTE["goal_hole"], z=0.002)
            m = 0.28
            self._flat(surface, [(r + m, c + m), (r + m, c + 1 - m), (r + 1 - m, c + 1 - m), (r + 1 - m, c + m)],
                       PALETTE["goal_inner"], z=0.003)
        elif kind == TileType.SWITCH:
            sc = PALETTE[tile.switch_kind.value]
            if tile.switch_kind == SwitchKind.SOFT:  # octagon
                pts = [(cr + 0.28 * math.sin(a), cc + 0.28 * math.cos(a))
                       for a in (i * math.pi / 4 + math.pi / 8 for i in range(8))]
                self._flat(surface, pts, sc, z=0.002)
            elif tile.switch_kind == SwitchKind.HEAVY:  # X
                d = 0.28
                self._line(surface, (cr - d, cc - d), (cr + d, cc + d), sc, 4, 0.002)
                self._line(surface, (cr - d, cc + d), (cr + d, cc - d), sc, 4, 0.002)
            else:  # split: "( )"
                for shift, a0, a1 in ((0.25, 140, 220), (-0.25, -40, 40)):
                    arc = [(cr + 0.55 * math.sin(math.radians(a)), cc + shift + 0.55 * math.cos(math.radians(a)))
                           for a in np.linspace(a0, a1, 7)]
                    for p, q in zip(arc, arc[1:]):
                        self._line(surface, p, q, sc, 3, 0.002)