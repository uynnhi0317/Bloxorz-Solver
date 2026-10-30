"""Renderer 2.5D.

Tọa độ trong world có dạng (row, col, z). z = 0 là mặt sàn.
Camera là một ma trận 2x3 (M)
Chiếu một điểm trong không gian 3D lên màn hình: screen = M @ (row, col, z)

Renderer CHỈ ĐỌC State/Board từ Source/game. Không đụng đến game logic (cầu, công tắc, tách khối...).
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pygame

from game import Board, Coord, Orientation, State, SwitchEffect, SwitchKind, Tile, TileType
from game.tile import PermanentState
from . import noface, tile_art

Color = Tuple[int, int, int]

TILE_THICK = 0.3
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


LAYER_SS = 3                 # sàn tĩnh vẽ ở độ phân giải gấp 3 rồi thu nhỏ (cache) -> mép mịn
BOX_SS = 3                   # khối và bóng cũng vẽ gấp 3 trong khung nhỏ quanh chúng mỗi frame
SHADOW_ALPHA = 120           # độ đậm bóng đổ dưới khối
SHADOW_SHIFT = (0.16, 0.06)  # bóng lệch theo (row, col) cho mỗi đơn vị chiều cao, ngược hướng đèn
LANTERN_SIZE = 1.45          # đèn lồng đá đánh dấu ô xuất phát


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
    pose: tư thế của hình (xem noface.POSES). active: cube đang điều khiển khi block tách.
    fade: độ hiện của mặt nạ (animation lăn dùng để đổi tư thế)."""
    verts: np.ndarray
    color: Color
    figure: Optional[str] = None
    pose: str = "standing"
    active: bool = True
    fade: float = 1.0


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
                 view: str = DEFAULT_VIEW):
        self.board = board
        self.view_name = view
        # Hiệu ứng tile "nền" được trộn vào mọi lần vẽ (vd cầu đang trượt chạy song song với
        # nước đi tiếp theo, không chặn input). fx truyền vào draw_scene ưu tiên hơn.
        self.ambient_fx: Dict[Coord, TileFx] = {}
        self.selected: Optional[Coord] = None   # cube đang điều khiển khi block tách (trạng thái UI)
        self.head_cell: Optional[Coord] = None  # cube mang phần đầu Vô Diện (trạng thái UI, không thuộc State)
        self.split_active = False               # khối đang tách: công tắc ( ) sáng (trạng thái UI)
        self.goal_glow = 0.0                    # 0..1: độ sáng cổng torii (WinAnimation đặt mỗi frame)
        self.pulses = None                      # hiệu ứng công tắc (animations.SwitchPulses)
        self.t = 0.0                            # đồng hồ hiển thị (ms) cho xoáy sáng, đèn lồng...
        self._layers: Dict[tuple, pygame.Surface] = {}  # (bridge_mask, split) -> ảnh sàn tĩnh đã khử răng cưa
        self._ss = 1  # hệ số supersample hiện hành (1 = vẽ thẳng)
        self._sprites: Dict[tuple, Optional[Tuple[pygame.Surface, Tuple[int, int]]]] = {}
        self._props: Dict[str, Tuple[pygame.Surface, Tuple[int, int]]] = {}
        self._water: Optional[Tuple[pygame.Surface, Tuple[int, int]]] = None
        self._order_cache: Optional[List[Coord]] = None
        self.bridge_axis = self._resolve_bridges()
        self._switches = [((r, c), t) for r, row in enumerate(board.tiles) for c, t in enumerate(row)
                          if t.kind == TileType.SWITCH]
        self.set_area(area, tile_width)

    def tick(self, dt_ms: float) -> None:
        self.t += dt_ms

    # ------------------------------------------------------------------ bridge
    def _resolve_bridges(self) -> Dict[Coord, Tuple[int, float, float]]:
        """Với mỗi ô cầu: (trục đi lại, vị trí đầu, vị trí cuối trong nhịp cầu 0..1).
        Ván cầu nằm vuông góc trục đi lại; lan can vồng lên giữa nhịp như cầu cong Nhật Bản."""
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
                axis = 1 if along_c else 0
                dr, dc = (0, 1) if axis == 1 else (1, 0)
                back = 0
                while is_bridge(r - dr * (back + 1), c - dc * (back + 1)):
                    back += 1
                fwd = 0
                while is_bridge(r + dr * (fwd + 1), c + dc * (fwd + 1)):
                    fwd += 1
                n = back + fwd + 1
                out[(r, c)] = (axis, back / n, (back + 1) / n)
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
        """Canh level giữa màn hình (gọi lại nếu cửa sổ game đổi kích thước)."""
        b = self.board
        self._layers, self._sprites, self._props, self._order_cache = {}, {}, {}, None  # đổi camera -> dựng lại
        self._water = None
        self._m = np.array(VIEWS[self.view_name], dtype=float)  # (2, 3)

        v = np.cross(self._m[0], self._m[1])
        self.view_dir = v if v[2] > 0 else -v
        corners = np.array([(r, c, z) for r in (0, b.height) for c in (0, b.width)
                            for z in (-TILE_THICK, BLOCK_HEIGHT)], dtype=float)
        xy = corners @ self._m.T
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        w, h = (hi - lo) * tile_width
        self.scale = tile_width * min(1.0, area.width / w, area.height / h)
        self.detail = max(0.5, min(1.0, self.scale / 64.0))  # nét mảnh đi khi bản đồ thu nhỏ
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
                    "head" if (r, c) == head else "body", active=(r, c) == self.selected)
                for r, c in sp
            ]
        dims = {Orientation.STANDING: (1, 1, 2),
                Orientation.HORIZONTAL: (1, 2, 1),
                Orientation.VERTICAL: (2, 1, 1)}[state.orientation]
        return [Box(make_box(state.row, state.column, 0, *dims), PALETTE["block"], "whole", state.orientation.value)]

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
        self._static_layer(bridge_mask)
        for r, c in self.drawn_tiles(bridge_mask):
            self._sprite(r, c, bridge_mask)

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
        """Cầu đang đổi trạng thái: bóng ván dưới nước + ván thật trượt lên/xuống và rõ/mờ dần."""
        ghost = self._sprite(r, c, mask | self.board.bridge_bit[tile.tile_id])
        if ghost is not None:
            surface.blit(*ghost)
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
        rồi box và vật dựng (torii, đèn lồng) theo độ sâu.
        tile_fx: hiệu ứng riêng cho từng tile (xem TileFx). Không có hiệu ứng nào -> dùng ảnh sàn
        tĩnh đã cache (mịn nhất, nhanh nhất); có hiệu ứng -> vẽ trực tiếp từng tile.
        hide_below: không vẽ phần box dưới z=0 (khối chìm vào cổng, không lộ ra dưới mép ô).
        """
        fx = {**self.ambient_fx, **(tile_fx or {})}
        self._draw_water_shadow(surface, fx)
        if not hide_below:
            self._draw_boxes(surface, boxes, alpha, below=True)
        if fx:
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
        self._draw_by_depth(surface, boxes, alpha, fx)
        if self.pulses is not None:
            self.pulses.draw_over(surface)

    # ------------------------------------------------------------------ hiệu ứng trên mặt sàn
    def _draw_water_shadow(self, surface: pygame.Surface, fx: Mapping[Coord, TileFx]) -> None:
        """Bóng mờ của cả bàn chơi trên mặt hồ; nhạt đi khi tile còn đang bay vào/rơi đi."""
        cells = [rc for rc in self.drawn_tiles(0) if self.board.tiles[rc[0]][rc[1]].kind != TileType.BRIDGE]
        if self._water is None:
            rect = self._board_rect().inflate(60, 60)
            img = pygame.Surface(rect.size, pygame.SRCALPHA)
            for r, c in cells:
                poly = self.project([(r, c, -0.75), (r, c + 1, -0.75), (r + 1, c + 1, -0.75), (r + 1, c, -0.75)])
                pygame.draw.polygon(img, (8, 12, 28, 170), (poly - np.array(rect.topleft)).tolist())
            w, h = rect.size
            blur = pygame.transform.smoothscale(img, (max(1, w // 8), max(1, h // 8)))
            self._water = (pygame.transform.smoothscale(blur, (w, h)), rect.topleft)
        img, pos = self._water
        moving = sum(1 for f in fx.values() if f.hidden or f.dz)
        img.set_alpha(int(255 * max(0.0, 1 - moving / max(1, len(cells)))))
        surface.blit(img, pos)

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
        """Xoáy sáng ở tâm cổng torii: chạy liên tục, bùng lên khi Vô Diện về đích (goal_glow)."""
        r, c = self.board.goal
        if (r, c) in fx:
            return
        cr, cc, glow = r + 0.56, c + 0.5, self.goal_glow
        rad = 0.32 + 0.30 * glow
        if glow > 0.01:
            big = max(8, int(self.scale * (0.7 + 1.5 * glow)) // 8 * 8)  # theo bậc để cache sprite nhỏ
            x, y = self.project([(cr, cc, 0.4 * glow)])[0]
            surface.blit(glow_sprite(big, (255, 190, 170), round(0.9 * glow, 1)), (x - big, y - big),
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
        col = (int(247 * level), int(190 * level), int(208 * level))
        width = max(1, int(round(2 * self.detail)))
        for pts in arms:
            pygame.draw.lines(layer, col, False, (pts - (x0, y0)).tolist(), width)
        surface.blit(layer, (int(x0), int(y0)), special_flags=pygame.BLEND_RGB_ADD)

    # ------------------------------------------------------------------ vật dựng: torii, đèn lồng
    def _prop_anchor(self, name: str) -> Tuple[float, float]:
        if name == "torii":
            r, c = self.board.goal
            return r + 0.16, c + 0.5     # sát mép sau ô đích để khối đứng phía trước cổng
        r, c = self.board.start
        return r + 1.02, c - 0.22        # góc trước-trái ô xuất phát, không bị khối che

    def _prop_cell(self, name: str) -> Coord:
        return self.board.goal if name == "torii" else self.board.start

    def _prop(self, name: str) -> Tuple[pygame.Surface, Tuple[int, int]]:
        if name not in self._props:
            pr, pc = self._prop_anchor(name)
            xy = self.project([(pr + dr, pc + dc, z) for dr in (-0.4, 0.4) for dc in (-0.9, 0.9) for z in (0.0, 2.0)])
            x0, y0 = (int(v) - 3 for v in np.floor(xy.min(axis=0)))
            x1, y1 = (int(v) + 4 for v in np.ceil(xy.max(axis=0)))
            w, h = x1 - x0, y1 - y0
            big = pygame.Surface((w * LAYER_SS, h * LAYER_SS), pygame.SRCALPHA)
            with self._supersample(LAYER_SS, (x0, y0)):
                (self._paint_torii if name == "torii" else self._paint_lantern)(big, pr, pc)
            img = pygame.transform.smoothscale(big, (w, h))
            br = img.get_bounding_rect()
            self._props[name] = (img.subsurface(br).copy(), (x0 + br.x, y0 + br.y))
        return self._props[name]

    def _solid(self, surface, r0, c0, z0, dr, dc, dz, color) -> None:
        self._draw_box(surface, make_box(r0, c0, z0, dr, dc, dz), color, bevel=False)

    def _front_poly(self, surface, r: float, pts, color) -> None:
        """Đa giác trong mặt phẳng đứng row = r, điểm dạng (col, z)."""
        pygame.draw.polygon(surface, color, self.project([(r, c, z) for c, z in pts]).tolist())

    def _paint_torii(self, surface, pr: float, pc: float) -> None:
        red, cap = PALETTE["torii"], PALETTE["torii_cap"]
        dark = shade(red, 0.62)
        half, top = 0.37, 1.62
        for dc in (-half, half):  # hai cột, chân đen
            self._solid(surface, pr - 0.05, pc + dc - 0.05, 0.0, 0.1, 0.1, top - 0.05, red)
            self._solid(surface, pr - 0.065, pc + dc - 0.065, 0.0, 0.13, 0.13, 0.12, cap)
        front = pr + 0.05
        # nuki: xà ngang dưới, có thẻ tên nhỏ ở giữa
        self._front_poly(surface, front, [(pc - half - 0.1, 1.30), (pc + half + 0.1, 1.30),
                                          (pc + half + 0.1, 1.20), (pc - half - 0.1, 1.20)], dark)
        self._front_poly(surface, front, [(pc - 0.06, 1.50), (pc + 0.06, 1.50), (pc + 0.06, 1.28), (pc - 0.06, 1.28)], cap)
        # kasagi: xà trên cong hai đầu + mũ đen
        wide = half + 0.24
        curve = [(pc + wide * t, top + 0.09 * abs(t) ** 2.4) for t in np.linspace(-1, 1, 11)]
        beam = curve + [(c, z - 0.11) for c, z in reversed(curve)]
        self._front_poly(surface, front, beam, red)
        capped = [(c, z + 0.06) for c, z in curve] + [(c, z) for c, z in reversed(curve)]
        self._front_poly(surface, front, capped, cap)

    def _paint_lantern(self, surface, pr: float, pc: float) -> None:
        """Đèn lồng đá tōrō: đế, trụ, hộp đèn, mái."""
        stone, k = PALETTE["lantern"], LANTERN_SIZE
        for half, z, h, col in ((0.13, 0.0, 0.05, shade(stone, 0.8)), (0.05, 0.05, 0.2, stone),
                                (0.10, 0.25, 0.04, shade(stone, 0.9))):
            self._solid(surface, pr - half * k, pc - half * k, z * k, 2 * half * k, 2 * half * k, h * k, col)
        self._solid(surface, pr - 0.075 * k, pc - 0.075 * k, 0.29 * k, 0.15 * k, 0.15 * k, 0.14 * k, PALETTE["lantern_light"])
        base = self.project([(pr + dr * k, pc + dc * k, 0.43 * k) for dr in (-0.15, 0.15) for dc in (-0.15, 0.15)])
        apex = self.project([(pr, pc, 0.60 * k)])
        pygame.draw.polygon(surface, shade(stone, 0.72), _hull([tuple(p) for p in base.tolist() + apex.tolist()]))
        tip = self.project([(pr, pc, 0.64 * k)])[0]
        pygame.draw.circle(surface, stone, tip.tolist(), 0.035 * k * self.scale)

    def _draw_prop(self, surface: pygame.Surface, name: str, fx: Mapping[Coord, TileFx]) -> None:
        f = fx.get(self._prop_cell(name))
        if f is not None and (f.hidden or f.xform is not None):
            return
        img, (x, y) = self._prop(name)
        dz = f.dz if f is not None else 0.0
        y = round(y + float(self._m[1, 2]) * self.scale * dz)
        if name == "torii" and self.goal_glow > 0.01:
            lit = img.copy()
            g = int(150 * self.goal_glow)
            lit.fill((g, int(g * 0.8), int(g * 0.6), 0), special_flags=pygame.BLEND_RGBA_ADD)
            surface.blit(lit, (x, y))
            return
        surface.blit(img, (x, y))
        if name == "lantern" and dz == 0.0:  # ánh đèn lập lòe
            pr, pc = self._prop_anchor(name)
            flick = 0.78 + 0.14 * math.sin(self.t / 130) + 0.08 * math.sin(self.t / 57)
            rad = max(4, int(self.scale * 0.42))
            gx, gy = self.project([(pr, pc, 0.36 * LANTERN_SIZE)])[0]
            surface.blit(glow_sprite(rad, PALETTE["lantern_light"], round(0.7 * flick, 1)), (gx - rad, gy - rad),
                         special_flags=pygame.BLEND_RGB_ADD)

    def _draw_by_depth(self, surface: pygame.Surface, boxes: Sequence[Box], alpha: int,
                       fx: Mapping[Coord, TileFx]) -> None:
        """Vẽ box và vật dựng từ xa tới gần, để khối đi sau cổng/đèn thì bị che đúng."""
        vd = self.view_dir[:2]
        items = [(float(np.dot(bx.verts[:, :2].mean(axis=0), vd)), 1, i) for i, bx in enumerate(boxes)]
        for name in ("torii", "lantern"):
            items.append((float(np.dot(self._prop_anchor(name), vd)), 0, name))
        for _, is_box, it in sorted(items, key=lambda x: (x[0], x[1])):
            if is_box:
                self._draw_boxes(surface, [boxes[it]], alpha, below=False)
            else:
                self._draw_prop(surface, it, fx)

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
                    self._replay(layer, noface.figure_ops(bx.verts, bx.figure, bx.pose, bx.color, self.view_dir, _LIGHT,
                                                          self.project, _clip_face, below, bx.fade, bx.active))
        img = pygame.transform.smoothscale(layer, (w, h))
        if alpha < 255:
            img.set_alpha(max(0, alpha))
        surface.blit(img, (x0, y0))
        if not below:
            for bx in boxes:
                if bx.figure in ("head", "body") and bx.active:
                    self._draw_active_marker(surface, bx, alpha)

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
            if q <= 0.0:
                tile_art.bridge_ghost(paint, r, c, self.bridge_axis[(r, c)][0])
            else:
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

        if kind == TileType.GOAL:
            tile_art.goal(paint, r, c)
        elif kind == TileType.SWITCH:
            tile_art.switch(paint, tile, r, c, self._switch_active(tile, mask))
        elif kind == TileType.FRAGILE:
            tile_art.fragile(paint, r, c)
        else:
            tile_art.floor(paint, r, c)

    def _draw_bridge(self, surface, r: int, c: int, q: float, fx: Optional[TileFx] = None) -> None:
        """Ván cầu gỗ đỏ. q = 1: nằm đúng mặt cầu; q nhỏ dần: từng ván tụt xuống nước lần lượt."""
        axis, t0, t1 = self.bridge_axis[(r, c)]
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

        for side in (0.12, 0.80):       # thanh đỡ chạy dọc
            box(0.0, 1.0, side, side + 0.08, -0.26 + planks[n // 2], 0.08, dark)
        for i in range(n):
            a, b = i / n + 0.012, (i + 1) / n - 0.012
            box(a, b, 0.04, 0.96, -0.09 + planks[i], 0.09, red if i % 2 else shade(red, 1.1))
        if q < 0.75:
            return
        # lan can cong: vồng lên giữa nhịp cầu, cột gỗ ở hai đầu ô
        rail_dz = -(1 - q) * 3.2 + dz0
        width = max(1, int(3 * self._ss * self.detail))
        for lo in (0.02, 0.98):
            pts = []
            for i in range(7):
                k = i / 6
                z = 0.13 + 0.16 * math.sin(math.pi * (t0 + (t1 - t0) * k)) + rail_dz
                pts.append((r + lo, c + k, z) if axis == 1 else (r + k, c + lo, z))
            scr = self.project(pts).tolist()
            for a2, b2 in zip(scr, scr[1:]):
                pygame.draw.line(surface, dark, a2, b2, width)
            for j in (0, 6):
                seg = self.project([(pts[j][0], pts[j][1], rail_dz), pts[j]]).tolist()
                pygame.draw.line(surface, dark, seg[0], seg[1], width)
