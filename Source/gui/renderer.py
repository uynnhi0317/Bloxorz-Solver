"""Renderer 2.5D.

Tọa độ trong world có dạng (row, col, z). z = 0 là mặt sàn.
Camera là một ma trận 2x3 (M)
Chiếu một điểm trong không gian 3D lên màn hình: screen = M @ (row, col, z)

Renderer CHỈ ĐỌC State/Board từ Source/game. Không đụng đến game logic (cầu, công tắc, tách khối...).
"""
from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pygame

from game import Board, Coord, Orientation, State, SwitchEffect, SwitchKind, Tile, TileType
from game.tile import PermanentState
from . import noface, tile_art
from .camera import Camera
from .mesh_render import IDENTITY, get_mesh, mesh_ops
from .raster import textured_quad
from .resources import get_assets

Color = Tuple[int, int, int]

TILE_THICK = 0.3
WATER_Z = -0.62     # mặt hồ, thấp hơn đáy tile
BED_Z = -2.2        # đáy hồ
GHOST_Z = -1.0      # ván cầu đóng nằm dưới mặt nước
BLOCK_HEIGHT = 2.0  # vật cao nhất được phép vẽ (là khối đứng thẳng)

from .theme import PALETTE, flower_sprite, glow_sprite  # noqa: E402  (bảng màu nằm trong theme.py)


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

def rotate(verts: np.ndarray, axis: int, sign: int, theta: float, pivot: float = None,
           slide: float = 0.0, drop: float = 0.0) -> np.ndarray:
    """Xoay các đỉnh `verts` một góc `theta` quanh một đường nằm ngang trên sàn.
    """
    out = verts.copy()
    if pivot is None:
        pivot = verts[:, axis].max() if sign > 0 else verts[:, axis].min()
        pz = verts[:, 2].min()          # độ cao trục = đáy box
    else:
        pz = 0.0                        # độ cao trục = mặt sàn
    u = sign * (verts[:, axis] - pivot)  # khoảng cách tới trục; dương = phía đang lăn tới
    w = verts[:, 2] - pz                 # độ cao so với trục

    out[:, axis] = pivot + sign * (u * math.cos(theta) + w * math.sin(theta) + slide)
    out[:, 2] = pz - u * math.sin(theta) + w * math.cos(theta) - drop
    return out


LIGHT_BASE, LIGHT_PEAK = 1.04, 0.30   # hệ số sáng mặt sàn: nền + phần cộng thêm ở tâm vùng đèn rọi
LAYER_SS = 3                 # sàn tĩnh vẽ ở độ phân giải gấp 3 rồi thu nhỏ (cache) -> mép mịn
BOX_SS = 3                   # khối và bóng cũng vẽ gấp 3 trong khung nhỏ quanh chúng mỗi frame
SHADOW_ALPHA = 120           # độ đậm bóng đổ dưới khối
SHADOW_SHIFT = (0.16, 0.06)  # bóng lệch theo (row, col) cho mỗi đơn vị chiều cao, ngược hướng đèn


def _hull(points):
    """Bao lồi (monotone chain) của các điểm (x, y)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return half(pts) + half(reversed(pts))


def _clip_cell(poly, r0, r1, c0, c1):
    """Cắt đa giác lồi theo hình chữ nhật [r0,r1] x [c0,c1] (Sutherland-Hodgman)."""
    for axis, bound, keep_ge in ((0, r0, True), (0, r1, False), (1, c0, True), (1, c1, False)):
        out = []
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            ina = a[axis] >= bound if keep_ge else a[axis] <= bound
            inb = b[axis] >= bound if keep_ge else b[axis] <= bound
            if ina:
                out.append(a)
            if ina != inb:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        poly = out
        if not poly:
            break
    return poly


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
    """Một khối cần vẽ. figure = None: hộp thường; "whole"/"head"/"body": hình Vô Diện trong khối.
    verts: 8 đỉnh, luôn là ảnh của một hộp thẳng qua phép xoay cứng (animation xoay chúng).
    rot: hướng của hình ở tư thế khi verts còn thẳng theo lưới. active: cube đang điều khiển khi block tách."""
    verts: np.ndarray
    color: Color
    figure: Optional[str] = None
    rot: np.ndarray = field(default_factory=lambda: IDENTITY)
    active: bool = True


@dataclass
class TileFx:
    """Hiệu ứng tạm thời cho MỘT tile khi vẽ (animation dùng để bay/rơi/nghiêng tile).
    dz: dịch tile theo trục z (âm = xuống dưới), áp dụng sau cùng.
    xform: biến đổi thêm các đỉnh của tile (vd nghiêng khi ô fragile rơi).
    plank: chỉ cho bridge: 0 = ván chìm hẳn dưới nước, 1 = ván nằm đúng mặt cầu.
           None = theo trạng thái đóng/mở của mask.
    hidden: không vẽ tile này."""
    dz: float = 0.0
    xform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    plank: Optional[float] = None
    hidden: bool = False


def figure_pose(bx: Box) -> Tuple[np.ndarray, np.ndarray]:
    """(ma trận hướng, gốc) của hình Vô Diện. Hình là vật rắn: hướng = (phép xoay hiện tại của khối so với lưới)
    x (hướng lúc khối thẳng lưới)."""
    v = bx.verts
    frame = np.stack([v[4] - v[0], v[2] - v[0], v[1] - v[0]], axis=1)
    frame = frame / np.linalg.norm(frame, axis=0)
    origin = v.mean(axis=0) - (0.0, 0.0, noface.SHAPES[bx.figure].drop)
    return frame @ bx.rot, origin


class _Paint:
    """Bút vẽ trên mặt một tile cho tile_art: tự áp dụng TileFx để họa tiết đi cùng tile."""

    def __init__(self, renderer: "Renderer", surface: pygame.Surface, fx: Optional[TileFx]):
        self.r, self.surface, self.fx = renderer, surface, fx

    def _project(self, pts, z: float) -> list:
        arr = np.array([(a, b, z) for a, b in pts], dtype=float)
        if self.fx is not None:
            if self.fx.xform is not None:
                arr = self.fx.xform(arr)
            if self.fx.dz:
                arr[:, 2] += self.fx.dz
        return self.r.project(arr).tolist()

    def poly(self, pts, color, z: float = 0.002) -> None:
        pygame.draw.polygon(self.surface, color, self._project(pts, z))

    def line(self, a, b, color, width: int = 3, z: float = 0.002) -> None:
        p = self._project((a, b), z)
        pygame.draw.line(self.surface, color, p[0], p[1], max(1, int(width * self.r._ss * self.r.detail)))


class Renderer:
    def __init__(self, board: Board, area: pygame.Rect, tile_width: float = 64.0,
                 camera: Optional[Camera] = None):
        self.board = board
        self.camera = camera if camera is not None else Camera()
        self.fast = False       # đang zoom: vẽ giản lược mỗi frame thay vì dựng lại ảnh sàn gấp 3
        self.water_pass: Optional[Callable[[pygame.Surface], None]] = None  # vẽ mặt nước (Background.draw_water)
        # Hiệu ứng tile "nền" được trộn vào mọi lần vẽ (vd cầu đang trượt chạy song song với
        # nước đi tiếp theo, không chặn input). fx truyền vào draw_scene ưu tiên hơn.
        self.ambient_fx: Dict[Coord, TileFx] = {}
        self.selected: Optional[Coord] = None   # cube đang điều khiển khi block tách (trạng thái UI)
        self.cube_rots: Dict[str, np.ndarray] = {"head": IDENTITY, "body": IDENTITY}  # hướng từng cube khi tách
        self.figure_rot: np.ndarray = IDENTITY  # hướng hiện tại của Vô Diện (trạng thái UI, không thuộc State)
        self.head_cell: Optional[Coord] = None  # cube mang phần đầu Vô Diện (trạng thái UI, không thuộc State)
        self.split_active = False               # khối đang tách: công tắc ( ) sáng (trạng thái UI)
        self.goal_glow = 0.0                    # 0..1: độ sáng hố dịch chuyển (WinAnimation đặt mỗi frame)
        self.pulses = None                      # hiệu ứng công tắc (animations.SwitchPulses)
        self.t = 0.0                            # đồng hồ hiển thị (ms) cho xoáy sáng, đèn lồng...
        self._layers: Dict[tuple, pygame.Surface] = {}  # (bridge_mask, split) -> ảnh sàn tĩnh đã khử răng cưa
        self._ss = 1  # hệ số supersample hiện hành (1 = vẽ thẳng)
        self._sprites: Dict[tuple, Optional[Tuple[pygame.Surface, Tuple[int, int]]]] = {}
        self._under: Dict[int, Tuple[pygame.Surface, Tuple[int, int]]] = {}  # bridge_mask -> phần dưới mặt nước
        self._order_cache: Optional[List[Coord]] = None
        self.bridge_axis = self._resolve_bridges()
        self._switches = [((r, c), t) for r, row in enumerate(board.tiles) for c, t in enumerate(row)
                          if t.kind == TileType.SWITCH]
        self.set_area(area, tile_width)

    def tick(self, dt_ms: float) -> None:
        self.t += dt_ms

    # ------------------------------------------------------------------ bridge
    def _resolve_bridges(self) -> Dict[Coord, int]:
        """Với mỗi ô cầu: trục đi lại (0 = theo row, 1 = theo col). Ván cầu nằm vuông góc với trục này."""
        b, out = self.board, {}

        def is_bridge(r, c):
            return b.tile_at((r, c)).kind == TileType.BRIDGE

        def walkable(r, c):
            return b.tile_at((r, c)).kind != TileType.VOID

        for r, row in enumerate(b.tiles):
            for c, tile in enumerate(row):
                if tile.kind != TileType.BRIDGE:
                    continue
                along_c = is_bridge(r, c - 1) or is_bridge(r, c + 1)
                along_r = is_bridge(r - 1, c) or is_bridge(r + 1, c)
                if along_c == along_r:  # không rõ: xét ô kề đi được
                    along_c = walkable(r, c - 1) or walkable(r, c + 1)
                out[(r, c)] = 1 if along_c else 0
        return out

    def _switch_active(self, tile: Tile, mask: int) -> bool:
        """Công tắc đang 'bật': toggle/permanent-open khi mọi cầu đích đang mở; permanent-close khi đóng hết.
        Công tắc ( ) sáng khi khối đang tách."""
        if tile.kind != TileType.SWITCH:
            return False
        if tile.switch_kind == SwitchKind.SPLIT:
            return self.split_active
        bits = 0
        for bid in tile.targets:
            bits |= self.board.bridge_bit[bid]
        closed = mask & bits
        if tile.effect == SwitchEffect.PERMANENT and tile.permanent_state == PermanentState.CLOSE:
            return closed == bits
        return closed == 0

    # ------------------------------------------------------------------ camera
    def set_area(self, area: pygame.Rect, tile_width: float = 64.0) -> None:
        """Canh level giữa màn hình (gọi lại nếu cửa sổ game đổi kích thước).
        Một hệ số scale chung nên không méo khi đổi tỉ lệ cửa sổ."""
        b = self.board
        self.area, self.tile_width = area, tile_width
        corners = np.array([(r, c, z) for r in (0, b.height) for c in (0, b.width)
                            for z in (-TILE_THICK, BLOCK_HEIGHT)], dtype=float)
        xy = corners @ self.camera.matrix().T
        w, h = (xy.max(axis=0) - xy.min(axis=0)) * tile_width
        self.fit_scale = tile_width * min(1.0, area.width / w, area.height / h)
        self.apply_camera()

    def apply_camera(self) -> None:
        """Đọc góc nhìn từ camera; mọi ảnh đã dựng sẵn đều hết hiệu lực."""
        b = self.board
        self._layers, self._sprites, self._order_cache = {}, {}, None
        self._under = {}
        self._warm_queue = self.drawn_tiles(0)
        self.camera_version = self.camera.version
        self._m = self.camera.matrix()
        v = np.cross(self._m[0], self._m[1])
        self.view_dir = v if v[2] > 0 else -v
        self.scale = self.fit_scale * self.camera.zoom
        self.detail = max(0.5, min(1.0, self.scale / 64.0))  # nét mảnh đi khi bản đồ thu nhỏ
        centre = np.array((b.height / 2, b.width / 2, 0.5)) @ self._m.T
        self.offset = np.array(self.area.center, dtype=float) - centre * self.scale

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
    def state_boxes(self, state: State) -> List[Box]:
        """Dựng các khối rắn cho mỗi trạng thái.
        - Đang tách: 2 cube 1x1x1 (phần đầu và phần thân Vô Diện), cube đang được điều khiển sáng hơn.
        - Nguyên khối: 1 box 1x1x2 theo orientation.
        """
        if state.split_positions is not None:
            sp = state.split_positions
            head = self.head_cell if self.head_cell in sp else sp[0]
            return [
                Box(make_box(r, c, 0, 1, 1, 1),
                    PALETTE["cube_active"] if (r, c) == self.selected else PALETTE["cube_idle"],
                    "head" if (r, c) == head else "body", self.cube_rots["head" if (r, c) == head else "body"],
                    active=(r, c) == self.selected)
                for r, c in sp
            ]
        dims = {Orientation.STANDING: (1, 1, 2),
                Orientation.HORIZONTAL: (1, 2, 1),
                Orientation.VERTICAL: (2, 1, 1)}[state.orientation]
        return [Box(make_box(state.row, state.column, 0, *dims), PALETTE["block"], "whole", self.figure_rot)]

    def draw(self, surface: pygame.Surface, state: State) -> None:
        """Vẽ cảnh tĩnh ứng với một state."""
        self.draw_scene(surface, state.bridge_mask, self.state_boxes(state))

    def drawn_tiles(self, bridge_mask: int) -> List[Coord]:
        """Các ô thực sự có vật thể để vẽ (bỏ void), theo thứ tự row 0->n, col 0->n."""
        b = self.board
        return [(r, c) for r in range(b.height) for c in range(b.width) if b.tiles[r][c].kind != TileType.VOID]

    @contextlib.contextmanager
    def _supersample(self, ss: int, origin):
        """Tạm vẽ ở độ phân giải gấp ss, tọa độ tính từ gốc `origin` (góc trên trái khung đích)."""
        old = (self.scale, self.offset, self._ss)
        self.scale = old[0] * ss
        self.offset = (old[1] - np.array(origin, dtype=float)) * ss
        self._ss = ss
        try:
            yield
        finally:
            self.scale, self.offset, self._ss = old

    def _board_rect(self) -> pygame.Rect:
        b = self.board
        corners = [(r, c, z) for r in (0, b.height) for c in (0, b.width) for z in (-2 * TILE_THICK, 0.5)]
        xy = self.project(corners)
        m = 10  # chừa chỗ cho quầng sáng
        x0, y0 = np.floor(xy.min(axis=0)).astype(int) - m
        x1, y1 = np.ceil(xy.max(axis=0)).astype(int) + m
        return pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))

    def _static_layer(self, mask: int) -> Tuple[pygame.Surface, Tuple[int, int]]:
        """Ảnh sàn cho một bridge_mask: vẽ gấp 3 rồi thu nhỏ để mép mịn. Cache theo mask."""
        rect = self._board_rect()
        key = (mask, self.split_active)
        img = self._layers.get(key)
        if img is None:
            if len(self._layers) >= 8:
                self._layers.pop(next(iter(self._layers)))
            w, h = rect.size
            big = pygame.Surface((w * LAYER_SS, h * LAYER_SS), pygame.SRCALPHA)
            with self._supersample(LAYER_SS, rect.topleft):
                self._draw_tiles(big, mask, {})
            img = self._layers[key] = pygame.transform.smoothscale(big, (w, h))
        return img, rect.topleft

    def prepare(self, bridge_mask: int) -> None:
        """Dựng trước ảnh sàn tĩnh và sprite từng tile cho một mask (gọi lúc vào màn / khi công tắc
        vừa đổi cầu) để lần vẽ đầu tiên không bị khựng."""
        if self.fast:
            return
        self._static_layer(bridge_mask)
        self._under_layer(bridge_mask)
        for r, c in self.drawn_tiles(bridge_mask):
            self._sprite(r, c, bridge_mask)

    def prepare_view(self, bridge_mask: int) -> None:
        """Sau khi zoom: chỉ dựng ngay ảnh sàn đang hiển thị, sprite từng tile để warm_sprites làm dần."""
        self._static_layer(bridge_mask)
        self._under_layer(bridge_mask)

    def warm_sprites(self, bridge_mask: int, count: int = 3) -> None:
        """Dựng vài sprite mỗi frame, để lần đầu tile chuyển động không bị khựng và cũng không gây khựng lúc này."""
        for r, c in self._warm_queue[:count]:
            self._sprite(r, c, bridge_mask)
        del self._warm_queue[:count]

    # ------------------------------------------------------------------ sprite từng tile
    def _sprite(self, r: int, c: int, mask: int) -> Optional[Tuple[pygame.Surface, Tuple[int, int]]]:
        """Sprite đã khử răng cưa của MỘT tile ở tư thế tĩnh (vẽ gấp 3 rồi thu nhỏ, cắt sát).
        Tile bay/rơi chỉ cần dịch sprite theo trục dọc nên vẫn mịn. None nếu tile không vẽ gì."""
        b = self.board
        tile = b.tiles[r][c]
        closed = tile.kind == TileType.BRIDGE and bool(mask & b.bridge_bit[tile.tile_id])
        key = (r, c, closed, self._switch_active(tile, mask))
        if key not in self._sprites:
            xy = self.project([(rr, cc, z) for rr in (r, r + 1) for cc in (c, c + 1) for z in (-2 * TILE_THICK - 0.1, 0.5)])
            x0, y0 = (int(v) - 3 for v in np.floor(xy.min(axis=0)))
            x1, y1 = (int(v) + 4 for v in np.ceil(xy.max(axis=0)))
            w, h = x1 - x0, y1 - y0
            big = pygame.Surface((w * LAYER_SS, h * LAYER_SS), pygame.SRCALPHA)
            with self._supersample(LAYER_SS, (x0, y0)):
                self._draw_tile(big, r, c, tile, mask, None)
            img = pygame.transform.smoothscale(big, (w, h))
            br = img.get_bounding_rect()
            self._sprites[key] = (img.subsurface(br).copy(), (x0 + br.x, y0 + br.y)) if br.width else None
        return self._sprites[key]

    def _order(self) -> List[Coord]:
        if self._order_cache is None:
            b = self.board
            self._order_cache = sorted(((r, c) for r in range(b.height) for c in range(b.width)),
                                       key=lambda rc: self._cell_depth(*rc))  # back to front
        return self._order_cache

    def _draw_tiles_fx(self, surface: pygame.Surface, mask: int, fx: Mapping[Coord, TileFx]) -> None:
        """Vẽ sàn khi có hiệu ứng, vẫn mịn: tile chỉ bay/rơi (dz) = dịch sprite; tile đổi dạng
        (nghiêng, ván cầu trượt) = vẽ gấp 3 trong khung nhỏ riêng."""
        b = self.board
        under, main = [], []
        for r, c in self._order():
            tile = b.tiles[r][c]
            (under if tile.kind == TileType.BRIDGE and self._is_under(tile, mask, fx.get((r, c))) else main).append((r, c))
        dy_per_z = float(self._m[1, 2]) * self.scale
        dx_per_z = float(self._m[0, 2]) * self.scale

        for r, c in under + main:
            f = fx.get((r, c))
            if f is not None and f.hidden:
                continue
            tile = b.tiles[r][c]
            if tile.kind == TileType.VOID:
                continue
            if f is not None and f.plank is not None:
                self._draw_planks_moving(surface, r, c, tile, mask, f.plank)
                continue
            if f is not None and f.xform is not None:
                self._draw_tile_shaped(surface, r, c, tile, mask, f)
                continue
            sp = self._sprite(r, c, mask)
            if sp is None:
                continue
            img, (x, y) = sp
            dz = f.dz if f is not None else 0.0
            surface.blit(img, (round(x + dx_per_z * dz), round(y + dy_per_z * dz)))

    def _draw_tile_shaped(self, surface: pygame.Surface, r: int, c: int, tile: Tile, mask: int, f: TileFx) -> None:
        """Tile đang đổi dạng: vẽ gấp 3 trong một dải thẳng đứng quanh ô (đủ chỗ cho tile rơi xuống)."""
        xy = self.project([(rr, cc, z) for rr in (r - 1, r + 2) for cc in (c - 1, c + 2) for z in (-0.7, 0.5)])
        x0, y0 = (int(v) - 4 for v in np.floor(xy.min(axis=0)))
        x1 = int(np.ceil(xy[:, 0].max())) + 4
        y1 = max(int(np.ceil(xy[:, 1].max())) + 4, surface.get_height())  # kéo dài xuống đáy màn hình
        x0, y0, x1, y1 = max(x0, -50), max(y0, -50), min(x1, surface.get_width() + 50), min(y1, surface.get_height() + 50)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return
        big = pygame.Surface((w * LAYER_SS, h * LAYER_SS), pygame.SRCALPHA)
        with self._supersample(LAYER_SS, (x0, y0)):
            self._draw_tile(big, r, c, tile, mask, f)
        surface.blit(pygame.transform.smoothscale(big, (w, h)), (x0, y0))

    def _draw_planks_moving(self, surface: pygame.Surface, r: int, c: int, tile: Tile, mask: int, q: float) -> None:
        """Cầu đang đổi trạng thái: ván thật trượt lên/xuống và rõ/mờ dần."""
        if q <= 0.0:
            return
        xy = self.project([(rr, cc, z) for rr in (r, r + 1) for cc in (c, c + 1) for z in (-1.1, 0.5)])
        x0, y0 = (int(v) - 4 for v in np.floor(xy.min(axis=0)))
        x1, y1 = (int(v) + 5 for v in np.ceil(xy.max(axis=0)))
        w, h = x1 - x0, y1 - y0
        big = pygame.Surface((w * LAYER_SS, h * LAYER_SS), pygame.SRCALPHA)
        with self._supersample(LAYER_SS, (x0, y0)):
            self._draw_bridge(big, r, c, q)
        img = pygame.transform.smoothscale(big, (w, h))
        img.set_alpha(int(255 * min(1.0, q * 1.7)))
        surface.blit(img, (x0, y0))

    def _draw_tiles(self, surface: pygame.Surface, bridge_mask: int, fx: Mapping[Coord, TileFx]) -> None:
        b = self.board
        # Lượt 1: cầu đang chìm dưới nước -> vẽ trước để tile khác che lên trên
        main = []
        for r, c in self._order():
            tile = b.tiles[r][c]
            if tile.kind == TileType.BRIDGE and self._is_under(tile, bridge_mask, fx.get((r, c))):
                self._draw_tile(surface, r, c, tile, bridge_mask, fx.get((r, c)))
            else:
                main.append((r, c))
        for r, c in main:
            self._draw_tile(surface, r, c, b.tiles[r][c], bridge_mask, fx.get((r, c)))

    def draw_scene(self, surface: pygame.Surface, bridge_mask: int, boxes: Sequence[Box],
                   alpha: int = 255, tile_fx: Optional[Mapping[Coord, TileFx]] = None,
                   hide_below: bool = False) -> None:
        """Vẽ cảnh (box và bảng).
        Gồm các bước: bóng bàn chơi trên nước, phần box dưới z=0, sàn, hiệu ứng trên mặt sàn, bóng đổ,
        rồi box theo độ sâu.
        tile_fx: hiệu ứng riêng cho từng tile (xem TileFx). Không có hiệu ứng nào -> dùng ảnh sàn
        tĩnh đã cache (mịn nhất, nhanh nhất); có hiệu ứng -> vẽ trực tiếp từng tile.
        hide_below: không vẽ phần box dưới z=0 (khối chìm vào cổng, không lộ ra dưới mép ô).
        """
        fx = {**self.ambient_fx, **(tile_fx or {})}
        if self.fast:
            self._draw_fast_under(surface, bridge_mask)
        else:
            self._draw_underwater(surface, bridge_mask, fx)
        if not hide_below:
            self._draw_boxes(surface, boxes, alpha, below=True)
        if self.water_pass is not None:
            self.water_pass(surface)   # mặt nước trong suốt phủ lên mọi thứ dưới nó
        if self.fast:
            self._draw_fast_tiles(surface, bridge_mask, fx)
        elif fx:
            self._draw_tiles_fx(surface, bridge_mask, fx)
        else:
            img, pos = self._static_layer(bridge_mask)
            surface.blit(img, pos)

        self._draw_switch_glow(surface, bridge_mask, fx)
        self._draw_vortex(surface, fx)
        if self.pulses is not None:
            self.pulses.draw_under(surface)
        self._draw_shadows(surface, boxes, bridge_mask, fx, alpha)
        self._draw_active_rings(surface, boxes)
        self._draw_by_depth(surface, boxes, alpha)
        if self.pulses is not None:
            self.pulses.draw_over(surface)

    # ------------------------------------------------------------------ hiệu ứng trên mặt sàn
    def _under_layer(self, mask: int) -> Tuple[pygame.Surface, Tuple[int, int]]:
        """Những gì nằm dưới mặt nước, nhìn xuyên qua được: bóng bàn chơi và sỏi dưới đáy, ván cầu đóng."""
        img = self._under.get(mask)
        if img is None:
            b = self.board
            pts = [(r, c, BED_Z) for r in (-3, b.height + 3) for c in (-3, b.width + 3)] + \
                  [(r, c, 0.0) for r in (-3, b.height + 3) for c in (-3, b.width + 3)]
            xy = self.project(pts)
            x0, y0 = np.floor(xy.min(axis=0)).astype(int) - 4
            x1, y1 = np.ceil(xy.max(axis=0)).astype(int) + 4
            w, h = int(x1 - x0), int(y1 - y0)
            ss = 2
            big = pygame.Surface((w * ss, h * ss), pygame.SRCALPHA)
            with self._supersample(ss, (int(x0), int(y0))):
                # bóng của bàn chơi in trên đáy hồ, làm mờ bằng thu nhỏ rồi phóng to
                shadow = pygame.Surface(big.get_size(), pygame.SRCALPHA)
                for r, c in self.drawn_tiles(0):
                    if b.tiles[r][c].kind != TileType.BRIDGE:
                        poly = self.project([(r, c, BED_Z), (r, c + 1, BED_Z), (r + 1, c + 1, BED_Z), (r + 1, c, BED_Z)])
                        pygame.draw.polygon(shadow, (30, 70, 76, 110), poly.tolist())
                small = pygame.transform.smoothscale(shadow, (max(1, big.get_width() // 12), max(1, big.get_height() // 12)))
                big.blit(pygame.transform.smoothscale(small, big.get_size()), (0, 0))
                self._paint_pebbles(big)
                self._paint_closed_bridges(big, mask)
            self._under[mask] = img = (pygame.transform.smoothscale(big, (w, h)), (int(x0), int(y0)))
        return img

    def _paint_pebbles(self, surface: pygame.Surface) -> None:
        """Sỏi dưới đáy hồ."""
        b = self.board
        rng = random.Random(b.height * 31 + b.width)
        for _ in range((b.height + 6) * (b.width + 6) // 7):
            r, c = rng.uniform(-3, b.height + 3), rng.uniform(-3, b.width + 3)
            rad = rng.uniform(0.05, 0.16)
            shade_k = rng.uniform(0.8, 1.2)
            col = tuple(min(255, int(v * shade_k)) for v in (126, 160, 146))
            poly = [(r + rad * math.sin(a) * rng.uniform(0.75, 1.1), c + rad * 1.3 * math.cos(a) * rng.uniform(0.75, 1.1), BED_Z)
                    for a in (i * math.pi / 4 for i in range(8))]
            pygame.draw.polygon(surface, col, self.project(poly).tolist())

    def _paint_closed_bridges(self, surface: pygame.Surface, mask: int) -> None:
        """Ván cầu đóng chìm dưới nước."""
        b = self.board
        for r, c in self._order():
            tile = b.tiles[r][c]
            if tile.kind == TileType.BRIDGE and mask & b.bridge_bit[tile.tile_id]:
                tile_art.bridge_ghost(_Paint(self, surface, None), r, c, self.bridge_axis[(r, c)], GHOST_Z)

    def _draw_underwater(self, surface: pygame.Surface, mask: int, fx: Mapping[Coord, TileFx]) -> None:
        """Nhạt đi khi tile còn đang bay vào/rơi đi."""
        img, pos = self._under_layer(mask)
        moving = sum(1 for f in fx.values() if f.hidden or f.dz)
        img.set_alpha(int(255 * max(0.0, 1 - moving / max(1, len(self.drawn_tiles(0))))))
        surface.blit(img, pos)

    # ------------------------------------------------------------------ chế độ nhanh khi đang zoom
    def _fast_tables(self) -> None:
        """Góc của mọi ô, tính sẵn để chiếu một lần bằng numpy mỗi frame."""
        b = self.board
        cells = [(r, c) for r, c in self.drawn_tiles(0)]
        self._fast_cells = cells
        self._fast_boxes = np.stack([make_box(r, c, -TILE_THICK, 1, 1, TILE_THICK) for r, c in cells])
        self._fast_kind = [b.tiles[r][c] for r, c in cells]

    def _fast_color(self, tile: Tile, mask: int) -> Color:
        if tile.kind == TileType.FRAGILE:
            return PALETTE["leaves"][0]
        if tile.kind == TileType.GOAL:
            return PALETTE["goal"]
        if tile.kind == TileType.BRIDGE:
            return PALETTE["bridge"]
        if tile.kind == TileType.SWITCH:
            return PALETTE[tile.switch_kind.value] if self._switch_active(tile, mask) else PALETTE[tile.switch_kind.value + "_off"]
        return PALETTE["floor"]

    def _draw_fast_under(self, surface: pygame.Surface, mask: int) -> None:
        b = self.board
        for r, c in self._order():
            tile = b.tiles[r][c]
            if tile.kind == TileType.BRIDGE and mask & b.bridge_bit[tile.tile_id]:
                tile_art.bridge_ghost(_Paint(self, surface, None), r, c, self.bridge_axis[(r, c)], GHOST_Z)

    def _draw_fast_tiles(self, surface: pygame.Surface, mask: int, fx: Mapping[Coord, TileFx]) -> None:
        """Mặt trên và hai mặt đứng của mọi ô, màu phẳng, không họa tiết. Đủ để định hướng khi đang zoom."""
        if not hasattr(self, "_fast_cells"):
            self._fast_tables()
        vd = self.view_dir
        side_r = FACES[1] if vd[0] > 0 else FACES[0]
        side_c = FACES[3] if vd[1] > 0 else FACES[2]
        top = FACES[5]
        order = sorted(range(len(self._fast_cells)), key=lambda i: self._cell_depth(*self._fast_cells[i]))
        pts = self._fast_boxes.copy()
        visible = np.ones(len(pts), dtype=bool)
        for i, cell in enumerate(self._fast_cells):
            tile, f = self._fast_kind[i], fx.get(cell)
            if tile.kind == TileType.BRIDGE and (mask & self.board.bridge_bit[tile.tile_id]) and (f is None or f.plank is None):
                visible[i] = False
            if f is not None:
                if f.hidden:
                    visible[i] = False
                pts[i, :, 2] += f.dz
                if f.plank is not None:
                    pts[i, :, 2] -= (1 - f.plank) * 0.8
        scr = (pts @ self._m.T) * self.scale + self.offset
        for i in order:
            if not visible[i]:
                continue
            col = self._fast_color(self._fast_kind[i], mask)
            s = scr[i]
            pygame.draw.polygon(surface, shade(col, 0.55), s[list(side_r)].tolist())
            pygame.draw.polygon(surface, shade(col, 0.68), s[list(side_c)].tolist())
            pygame.draw.polygon(surface, col, s[list(top)].tolist())
            pygame.draw.aalines(surface, shade(col, 0.6), True, s[list(top)].tolist())

    def _ground_ring(self, r: float, c: float, rad: float, n: int = 28) -> np.ndarray:
        return self.project([(r + rad * math.sin(2 * math.pi * i / n), c + rad * math.cos(2 * math.pi * i / n), 0.01)
                             for i in range(n)])

    def _draw_switch_glow(self, surface: pygame.Surface, mask: int, fx: Mapping[Coord, TileFx]) -> None:
        """Công tắc đang bật thở sáng nhẹ, nên phân biệt được với công tắc chưa bật."""
        breathe = 0.72 + 0.28 * math.sin(self.t / 420)
        rad = max(4, int(self.scale * 0.5))
        for (r, c), tile in self._switches:
            if (r, c) in fx or not self._switch_active(tile, mask):
                continue
            spr = glow_sprite(rad, PALETTE[tile.switch_kind.value], round(0.40 * breathe, 1))
            x, y = self.project([(r + 0.5, c + 0.5, 0.0)])[0]
            surface.blit(spr, (x - rad, y - rad), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_vortex(self, surface: pygame.Surface, fx: Mapping[Coord, TileFx]) -> None:
        """Xoáy sáng trong hố dịch chuyển: chạy liên tục, bùng lên khi Vô Diện về đích (goal_glow)."""
        r, c = self.board.goal
        if (r, c) in fx:
            return
        cr, cc, glow = r + 0.5, c + 0.5, self.goal_glow
        rad = 0.32 + 0.30 * glow
        if glow > 0.01:
            big = max(8, int(self.scale * (0.7 + 1.5 * glow)) // 8 * 8)  # theo bậc để cache sprite nhỏ
            x, y = self.project([(cr, cc, 0.4 * glow)])[0]
            surface.blit(glow_sprite(big, PALETTE["goal_swirl"], round(0.9 * glow, 1)), (x - big, y - big),
                         special_flags=pygame.BLEND_RGB_ADD)
        arms = []
        spin = self.t / (620 - 380 * glow)
        for arm in range(3):
            pts = []
            for i in range(13):
                k = i / 12
                a = spin + arm * 2 * math.pi / 3 + k * 3.4
                pts.append((cr + rad * (1 - k) * math.sin(a), cc + rad * (1 - k) * math.cos(a), 0.012))
            arms.append(self.project(pts))
        allp = np.concatenate(arms)
        x0, y0 = np.floor(allp.min(axis=0)).astype(int) - 3
        x1, y1 = np.ceil(allp.max(axis=0)).astype(int) + 4
        layer = pygame.Surface((int(x1 - x0), int(y1 - y0)))
        level = 0.55 + 0.45 * glow
        col = tuple(int(v * level) for v in PALETTE["goal_swirl"])
        width = max(1, int(round(2 * self.detail)))
        for pts in arms:
            pygame.draw.lines(layer, col, False, (pts - (x0, y0)).tolist(), width)
        surface.blit(layer, (int(x0), int(y0)), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_by_depth(self, surface: pygame.Surface, boxes: Sequence[Box], alpha: int) -> None:
        """Vẽ box từ xa tới gần."""
        vd = self.view_dir[:2]
        for bx in sorted(boxes, key=lambda b: float(np.dot(b.verts[:, :2].mean(axis=0), vd))):
            self._draw_boxes(surface, [bx], alpha, below=False)

    def _draw_shadows(self, surface: pygame.Surface, boxes: Sequence[Box], mask: int,
                      fx: Mapping[Coord, TileFx], alpha: int) -> None:
        """Bóng đổ của khối lên sàn: bao lồi của chân đế + đỉnh khối (lệch ngược hướng đèn),
        cắt theo từng ô sàn có thật nên không bao giờ tràn ra khoảng trống. Khối bay cao thì bóng nhạt đi."""
        b = self.board
        for bx in boxes:
            v = bx.verts
            zmin = float(v[:, 2].min())
            if zmin < -1e-6:
                continue  # đang chìm xuống: không có bóng
            pts = [(float(r - SHADOW_SHIFT[0] * (z - zmin)), float(c - SHADOW_SHIFT[1] * (z - zmin)))
                   for r, c, z in v]
            # bóng tiếp xúc: chân khối nở ra 0.12 ô mọi phía (nhìn thấy cả phía trước), cộng bóng đổ lệch
            grow = [(float(r) + dr, float(c) + dc) for r, c, z in v if z - zmin < 1e-6
                    for dr in (-0.12, 0.12) for dc in (-0.12, 0.12)]
            hull = _hull(grow + pts)
            if len(hull) < 3:
                continue
            a = int(SHADOW_ALPHA * max(0.0, 1 - zmin / 6.0) * alpha / 255)
            if a <= 0:
                continue
            if bx.figure in ("head", "body"):   # cube tách nhỏ hơn ô -> bóng thu theo
                mr, mc = float(v[:, 0].mean()), float(v[:, 1].mean())
                hull = [(mr + (p[0] - mr) * 0.72, mc + (p[1] - mc) * 0.72) for p in hull]
            rs, cs = [p[0] for p in hull], [p[1] for p in hull]
            polys = []
            for r in range(max(0, int(math.floor(min(rs)))), min(b.height, int(math.ceil(max(rs))))):
                for c in range(max(0, int(math.floor(min(cs)))), min(b.width, int(math.ceil(max(cs))))):
                    tile, f = b.tiles[r][c], fx.get((r, c))
                    if tile.kind == TileType.VOID or f is not None:
                        continue
                    if tile.kind == TileType.BRIDGE and mask & b.bridge_bit[tile.tile_id]:
                        continue
                    piece = _clip_cell(hull, r, r + 1, c, c + 1)
                    if len(piece) >= 3:
                        polys.append(self.project([(pr, pc, 0.004) for pr, pc in piece]).tolist())
            if not polys:
                continue
            allp = [p for poly in polys for p in poly]
            x0, y0 = int(min(p[0] for p in allp)) - 2, int(min(p[1] for p in allp)) - 2
            x1, y1 = int(max(p[0] for p in allp)) + 3, int(max(p[1] for p in allp)) + 3
            ss = BOX_SS
            layer = pygame.Surface(((x1 - x0) * ss, (y1 - y0) * ss), pygame.SRCALPHA)
            for poly in polys:
                pygame.draw.polygon(layer, (0, 0, 0, a), [((x - x0) * ss, (y - y0) * ss) for x, y in poly])
            surface.blit(pygame.transform.smoothscale(layer, (x1 - x0, y1 - y0)), (x0, y0))

    def _is_under(self, tile: Tile, mask: int, fx: Optional[TileFx]) -> bool:
        if fx is not None and fx.plank is not None:
            return True
        return bool(mask & self.board.bridge_bit[tile.tile_id])  # đóng = đang chìm dưới nước

    def _draw_boxes(self, surface: pygame.Surface, boxes: Sequence[Box], alpha: int, below: bool) -> None:
        """Vẽ các khối trong một khung nhỏ quanh chúng, ở độ phân giải gấp BOX_SS rồi thu nhỏ (mép mịn)."""
        boxes = [bx for bx in boxes if not below or bx.verts[:, 2].min() < -1e-9]
        if not boxes:
            return
        boxes = sorted(boxes, key=lambda x: float(np.dot(x.verts.mean(axis=0), self.view_dir)))
        pts = self.project(np.concatenate([bx.verts for bx in boxes]))
        x0, y0 = (int(v) - 3 for v in np.floor(pts.min(axis=0)))
        x1, y1 = (int(v) + 4 for v in np.ceil(pts.max(axis=0)))
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0 or w > 4000 or h > 4000:
            return
        layer = pygame.Surface((w * BOX_SS, h * BOX_SS), pygame.SRCALPHA)
        with self._supersample(BOX_SS, (x0, y0)):
            for bx in boxes:
                if bx.figure is None:
                    self._draw_box(layer, bx.verts, bx.color, below)
                else:
                    self._replay(layer, self._figure_ops(bx, below))
        img = pygame.transform.smoothscale(layer, (w, h))
        if alpha < 255:
            img.set_alpha(max(0, alpha))
        surface.blit(img, (x0, y0))
        if not below:
            for bx in boxes:
                if bx.figure in ("head", "body") and bx.active:
                    self._draw_active_marker(surface, bx, alpha)

    def _figure_ops(self, bx: Box, below: bool) -> list:
        rot, origin = figure_pose(bx)
        return mesh_ops(noface.figure_mesh(bx.figure), rot, origin, self.view_dir, _LIGHT, self.project,
                        _clip_face, below, bx.color, 1.0 if bx.active else 0.72, rim=True)

    def _draw_active_rings(self, surface: pygame.Surface, boxes: Sequence[Box]) -> None:
        """Vòng hồng trên sàn quanh cube đang điều khiển (vẽ trước khối để khối đứng lên trên)."""
        for bx in boxes:
            if bx.figure in ("head", "body") and bx.active and abs(float(bx.verts[:, 2].min())) < 0.02:
                r, c = bx.verts[:, :2].mean(axis=0)
                ring = self._ground_ring(float(r), float(c), 0.46)
                pygame.draw.lines(surface, PALETTE["soft"], True, ring.tolist(), max(1, int(2 * self.detail)))

    def _draw_active_marker(self, surface: pygame.Surface, bx: Box, alpha: int) -> None:
        """Cube đang điều khiển: bông hoa nhấp nhô phía trên đầu (vòng dưới chân vẽ ở _draw_active_rings)."""
        r, c = bx.verts[:, :2].mean(axis=0)
        ztop = float(bx.verts[:, 2].max())
        rad = max(5, int(self.scale * 0.13))
        spr = flower_sprite(rad)
        x, y = self.project([(r, c, ztop + 0.12 + 0.05 * math.sin(self.t / 240))])[0]
        spr.set_alpha(alpha)
        surface.blit(spr, (x - spr.get_width() / 2, y - spr.get_height() / 2))

    # --------------------------------------------------------------- drawing
    def _draw_box(self, surface: pygame.Surface, verts: np.ndarray, color: Color,
                  below: Optional[bool] = None, bevel: bool = True) -> None:
        """Vẽ khối hộp.
        below=None --> Vẽ cả khối
        below=False/True --> vẽ phần hộp nửa trên/dưới z=0
        Viền khử răng cưa (aaline); mặt trên có bevel: vạch sáng ở cạnh hướng đèn, vạch tối ở cạnh khuất.
        """
        self._replay(surface, self._box_ops(verts, color, below, bevel))

    def _replay(self, surface: pygame.Surface, ops: list) -> None:
        ss = self._ss
        for op in ops:
            if op[0] == 0:
                pygame.draw.polygon(surface, op[1], op[2])
            elif ss == 1:
                pygame.draw.aaline(surface, op[1], op[2], op[3])
            else:  # đang supersample: đường dày ss px, thu nhỏ xuống sẽ thành nét 1 px mịn
                pygame.draw.line(surface, op[1], op[2], op[3], ss)

    def _box_ops(self, verts: np.ndarray, color: Color, below: Optional[bool], bevel: bool) -> list:
        ops: list = []
        center = verts.mean(axis=0)
        faces = []
        for idx in FACES:
            pts = verts[list(idx)]
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            c = pts.mean(axis=0)
            if np.dot(n, c - center) < 0:  # make the normal point outward
                n = -n
            n = n / (np.linalg.norm(n) + 1e-9)
            if float(np.dot(n, self.view_dir)) <= 1e-6:
                continue  # mặt quay lưng về camera: luôn bị che, bỏ qua cho nhanh
            faces.append((float(np.dot(c, self.view_dir)), pts, n))
        faces.sort(key=lambda f: f[0])  # painter's algorithm, far faces first
        edge = shade(color, 0.45)
        for _, pts, n in faces:
            poly, cut = (list(pts), [False] * len(pts)) if below is None else _clip_face(pts, below)
            if len(poly) < 3:
                continue
            screen = self.project(np.array(poly)).tolist()
            f = 0.55 + 0.45 * max(0.0, float(np.dot(n, _LIGHT)))
            ops.append((0, shade(color, f), screen))
            for i in range(len(screen)):
                if not cut[i]:
                    ops.append((1, edge, screen[i], screen[(i + 1) % len(screen)]))
            if bevel and n[2] > 0.9 and len(poly) == 4 and below is not True:
                ops += self._bevel(np.array(poly), color)
        return ops

    def _bevel(self, pts: np.ndarray, color: Color) -> list:
        """Vạch bevel thụt vào 12% trên mặt trên: cạnh hướng đèn sáng, cạnh khuất tối."""
        ops = []
        cen = pts.mean(axis=0)
        lxy = _LIGHT[:2] / np.linalg.norm(_LIGHT[:2])
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            out = (a + b) / 2 - cen
            out[2] = 0.0
            norm = np.linalg.norm(out)
            if norm < 1e-9:
                continue
            d = float(np.dot(out[:2] / norm, lxy))
            if abs(d) < 0.3:
                continue
            a2, b2 = a + (cen - a) * 0.12, b + (cen - b) * 0.12
            p = self.project(np.array([a2, b2])).tolist()
            ops.append((1, shade(color, 1.35 if d > 0 else 0.78), p[0], p[1]))
        return ops

    def _draw_tile(self, surface, r: int, c: int, tile: Tile, mask: int,
                   fx: Optional[TileFx] = None) -> None:
        kind = tile.kind
        if kind == TileType.VOID or (fx is not None and fx.hidden):
            return
        paint = _Paint(self, surface, fx)
        if kind == TileType.BRIDGE:
            closed = bool(mask & self.board.bridge_bit[tile.tile_id])
            q = fx.plank if fx is not None and fx.plank is not None else (0.0 if closed else 1.0)
            if q > 0.0:   # cầu đóng hẳn nằm dưới mặt nước, xem _paint_closed_bridges
                self._draw_bridge(surface, r, c, q, fx)
            return

        if kind == TileType.FRAGILE:
            color = PALETTE["fragile"]
        elif kind == TileType.GOAL:
            color = PALETTE["goal"]
        else:
            color = PALETTE["floor"]
            if (r + c) % 2:
                color = shade(color, 0.94)  # subtle checkerboard
        verts = make_box(r, c, -TILE_THICK, 1, 1, TILE_THICK)
        if fx is not None and fx.xform is not None:
            verts = fx.xform(verts)
        if fx is not None and fx.dz:
            verts = verts.copy()
            verts[:, 2] += fx.dz
        self._draw_box(surface, verts, color)
        if kind != TileType.FRAGILE and (fx is None or fx.xform is None):
            self._texture_top(surface, verts, color, "stone.png", (r * 0.37, c * 0.61), (0.5, 0.5), spotlit=True)

        if kind == TileType.GOAL:
            tile_art.goal(paint, r, c)
        elif kind == TileType.SWITCH:
            tile_art.switch(paint, tile, r, c, self._switch_active(tile, mask))
        elif kind == TileType.FRAGILE:
            tile_art.fragile(paint, r, c)
        else:
            tile_art.floor(paint, r, c)

    def _spot_gain(self, r: float, c: float) -> float:
        """Độ sáng mặt sàn như có đèn rọi xuống giữa bàn chơi. Tính theo vị trí nên liền mạch qua các ô."""
        b = self.board
        spread = max(3.0, 0.6 * max(b.height, b.width))
        d2 = (r - b.height / 2) ** 2 + (c - b.width / 2) ** 2
        return LIGHT_BASE + LIGHT_PEAK * math.exp(-d2 / (spread * spread))

    def _texture_top(self, surface, verts: np.ndarray, color: Color, name: str, uv0, span, spotlit: bool = False) -> None:
        """Dán texture lên mặt trên của hộp (chỉ khi dựng ảnh tĩnh). Thiếu texture: mặt giữ nguyên màu phẳng."""
        tex = get_assets().load_texture(name)
        if tex is None or self._ss == 1:
            return
        top = self.project(verts[[1, 3, 7, 5]])
        gains = [self._spot_gain(p[0], p[1]) for p in verts[[1, 3, 7, 5]]] if spotlit else (1.0, 1.0, 1.0, 1.0)
        textured_quad(surface, top[0], top[1], top[3], tex, uv0, span, shade(color, 0.55 + 0.45 * float(_LIGHT[2])), gains)
        self._replay(surface, [(1, shade(color, 0.45), a.tolist(), b.tolist()) for a, b in zip(top, np.roll(top, -1, axis=0))])
        self._replay(surface, self._bevel(verts[[1, 3, 7, 5]], color))

    def _draw_bridge(self, surface, r: int, c: int, q: float, fx: Optional[TileFx] = None) -> None:
        """Thân cầu: ván gỗ nâu và thanh đỡ, không có lan can. q = 1: nằm đúng mặt cầu; q nhỏ dần: từng ván tụt xuống nước lần lượt."""
        axis = self.bridge_axis[(r, c)]
        red, dark = PALETTE["bridge"], PALETTE["bridge_dark"]
        dz0 = fx.dz if fx is not None else 0.0
        n = 5
        planks = []
        for i in range(n):
            p = min(1.0, max(0.0, q * 1.4 - 0.4 * i / (n - 1)))
            planks.append(-(1 - p * p * (3 - 2 * p)) * 0.8 + dz0)

        def box(a, b, lo, hi, z0, dz, col):  # a,b theo trục đi lại; lo,hi theo trục ngang
            if axis == 1:
                v = make_box(r + lo, c + a, z0, hi - lo, b - a, dz)
            else:
                v = make_box(r + a, c + lo, z0, b - a, hi - lo, dz)
            self._draw_box(surface, v, col, bevel=False)
            if q >= 1.0 and col != dark:   # đã nằm yên: dán vân gỗ lên mặt ván
                self._texture_top(surface, v, col, "torii_wood.png", (r * 0.29 + a, c * 0.43 + lo), (0.3, 0.5))

        for side in (0.12, 0.80):       # thanh đỡ chạy dọc
            box(0.0, 1.0, side, side + 0.08, -0.26 + planks[n // 2], 0.08, dark)
        for i in range(n):
            a, b = i / n + 0.012, (i + 1) / n - 0.012
            box(a, b, 0.04, 0.96, -0.09 + planks[i], 0.09, red if i % 2 else shade(red, 1.1))
