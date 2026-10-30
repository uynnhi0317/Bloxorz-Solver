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

from game import Board, Coord, Orientation, State, SwitchKind, Tile, TileType

Color = Tuple[int, int, int]

TILE_THICK = 0.3 
BLOCK_HEIGHT = 2.0  # vật cao nhất được phép vẽ (là khối đứng thẳng)

from .theme import PALETTE  # noqa: E402  (bảng màu nằm trong theme.py)


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
    verts: np.ndarray
    color: Color


@dataclass
class TileFx:
    """Hiệu ứng tạm thời cho MỘT tile khi vẽ (animation dùng để bay/rơi/lật tile).
    dz: dịch tile theo trục z (âm = xuống dưới), áp dụng sau cùng.
    xform: biến đổi thêm 8 đỉnh của tile (vd nghiêng khi ô fragile rơi).
    theta: chỉ cho bridge: góc lật quanh bản lề (0 = mở phẳng, pi = gập dưới tile kề bên).
           None = theo trạng thái đóng/mở của mask.
    under: vẽ ở lượt "bên dưới" (trước các tile khác) - dùng cho bridge đang gập dưới tile kề.
    hidden: không vẽ tile này."""
    dz: float = 0.0
    xform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    theta: Optional[float] = None
    under: bool = False
    hidden: bool = False

STOW_DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
STOW_PEEK = 0.12  # cầu gập thò ra khỏi mép tile kề một đoạn nhỏ cho dễ thấy


class Renderer:
    def __init__(self, board: Board, area: pygame.Rect, tile_width: float = 64.0,
                 view: str = DEFAULT_VIEW):
        self.board = board
        self.view_name = view
        # Hiệu ứng tile "nền" được trộn vào mọi lần vẽ (vd cầu đang lật chạy song song với
        # nước đi tiếp theo, không chặn input). fx truyền vào draw_scene ưu tiên hơn.
        self.ambient_fx: Dict[Coord, TileFx] = {}
        self.selected: Optional[Coord] = None  # cube đang điều khiển khi block tách (trạng thái UI)
        self._cache: Dict[tuple, list] = {}  # tile tĩnh -> danh sách lệnh vẽ đã chiếu sẵn
        self._layers: Dict[int, pygame.Surface] = {}  # bridge_mask -> ảnh sàn tĩnh đã khử răng cưa + glow
        self._ss = 1  # hệ số supersample hiện hành (1 = vẽ thẳng)
        self._origin = (0, 0)  # gốc tọa độ của khung đang vẽ (để key cache không lẫn giữa các khung)
        self._sprites: Dict[tuple, Optional[Tuple[pygame.Surface, Tuple[int, int]]]] = {}
        self._order_cache: Optional[List[Coord]] = None
        self.hinges = self._resolve_hinges()
        self.set_area(area, tile_width)

    # ------------------------------------------------------------------ bridge
    def _resolve_hinges(self) -> Dict[Coord, Tuple[int, int, float]]:
        """Với mỗi cầu: (axis, sign, pivot) của bản lề = cạnh chung với tile kề bên mà nó gập xuống dưới.
        Chọn theo metadata {"stow": "up|down|left|right"} của tile cầu trong JSON; không có thì
        tự chọn tile kề: ưu tiên sàn thường, rồi fragile/switch, cuối cùng là goal. Cầu không có tile kề hợp lệ -> không có
        bản lề (khi đóng sẽ không vẽ gì)."""
        b, out = self.board, {}
        for r, row in enumerate(b.tiles):
            for c, tile in enumerate(row):
                if tile.kind != TileType.BRIDGE:
                    continue
                pref = dict(tile.metadata).get("stow")
                dirs = ([pref] if pref in STOW_DIRS else []) + ["up", "left", "down", "right"]
                pick = None
                for rank in (0, 1, 2):  # 0: sàn thường, 1: fragile/switch, 2: goal
                    for d in dirs:
                        dr, dc = STOW_DIRS[d]
                        nb = b.tile_at((r + dr, c + dc))
                        if nb.kind in (TileType.VOID, TileType.BRIDGE):
                            continue
                        nb_rank = 0 if nb.kind == TileType.FLOOR else 2 if nb.kind == TileType.GOAL else 1
                        if nb_rank == rank:
                            pick = (dr, dc)
                            break
                    if pick:
                        break
                if pick:
                    dr, dc = pick
                    axis = 0 if dr else 1
                    d = dr or dc
                    out[(r, c)] = (axis, -d, float((r if axis == 0 else c) + (1 if d > 0 else 0)))
        return out

    def _hinge_rotate(self, verts: np.ndarray, cell: Coord, theta: float) -> np.ndarray:
        """Lật tile cầu quanh bản lề (cạnh chung với tile kề, ở độ cao đáy tile)."""
        axis, sign, pivot = self.hinges[cell]
        v = verts.copy()
        v[:, 2] += TILE_THICK
        out = rotate(v, axis, sign, theta, pivot, STOW_PEEK * theta / math.pi)
        out[:, 2] -= TILE_THICK
        return out

    # ------------------------------------------------------------------ camera
    def set_area(self, area: pygame.Rect, tile_width: float = 64.0) -> None:
        """Canh level giữa màn hình (gọi lại nếu cửa sổ game đổi kích thước)."""
        b = self.board
        self._cache, self._layers, self._sprites, self._order_cache = {}, {}, {}, None  # đổi camera -> dựng lại
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
    def state_boxes(self, state: State) -> List[Box]:
        """Dựng các khối rắn cho mỗi trạng thái.
        - Đang tách: 2 cube 1x1x1, cube đang được điều khiển (self.selected) có màu active.
        - Nguyên khối: 1 box 1x1x2 theo orientation.
        """
        if state.split_positions is not None:
            return [
                Box(make_box(r, c, 0, 1, 1, 1),
                    PALETTE["cube_active"] if (r, c) == self.selected else PALETTE["cube_idle"])
                for r, c in state.split_positions
            ]
        dims = {Orientation.STANDING: (1, 1, 2),
                Orientation.HORIZONTAL: (1, 2, 1),
                Orientation.VERTICAL: (2, 1, 1)}[state.orientation]
        return [Box(make_box(state.row, state.column, 0, *dims), PALETTE["block"])]

    def draw(self, surface: pygame.Surface, state: State) -> None:
        """Vẽ cảnh tĩnh ứng với một state."""
        self.draw_scene(surface, state.bridge_mask, self.state_boxes(state))

    def drawn_tiles(self, bridge_mask: int) -> List[Coord]:
        """Các ô thực sự có vật thể để vẽ (bỏ void; cầu đóng chỉ tính nếu có gập dưới tile kề),
        theo thứ tự row 0->n, col 0->n."""
        b = self.board
        out = []
        for r in range(b.height):
            for c in range(b.width):
                t = b.tiles[r][c]
                if t.kind == TileType.VOID:
                    continue
                if (t.kind == TileType.BRIDGE and bridge_mask & b.bridge_bit[t.tile_id]
                        and (r, c) not in self.hinges):
                    continue  # cầu đóng không có chỗ gập -> không có gì để vẽ
                out.append((r, c))
        return out

    @contextlib.contextmanager
    def _supersample(self, ss: int, origin):
        """Tạm vẽ ở độ phân giải gấp ss, tọa độ tính từ gốc `origin` (góc trên trái khung đích)."""
        old = (self.scale, self.offset, self._ss, self._origin)
        self.scale = old[0] * ss
        self.offset = (old[1] - np.array(origin, dtype=float)) * ss
        self._ss, self._origin = ss, (int(origin[0]), int(origin[1]))
        try:
            yield
        finally:
            self.scale, self.offset, self._ss, self._origin = old

    def _board_rect(self) -> pygame.Rect:
        b = self.board
        corners = [(r, c, z) for r in (0, b.height) for c in (0, b.width) for z in (-2 * TILE_THICK, 0.0)]
        xy = self.project(corners)
        m = 10  # chừa chỗ cho quầng sáng
        x0, y0 = np.floor(xy.min(axis=0)).astype(int) - m
        x1, y1 = np.ceil(xy.max(axis=0)).astype(int) + m
        return pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))

    def _static_layer(self, mask: int) -> Tuple[pygame.Surface, Tuple[int, int]]:
        """Ảnh sàn cho một bridge_mask: vẽ gấp 3 rồi thu nhỏ để mép mịn. Cache theo mask."""
        rect = self._board_rect()
        img = self._layers.get(mask)
        if img is None:
            if len(self._layers) >= 8:
                self._layers.pop(next(iter(self._layers)))
            w, h = rect.size
            big = pygame.Surface((w * LAYER_SS, h * LAYER_SS), pygame.SRCALPHA)
            with self._supersample(LAYER_SS, rect.topleft):
                self._draw_tiles(big, mask, {})
            img = self._layers[mask] = pygame.transform.smoothscale(big, (w, h))
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
        key = (r, c, closed)
        if key not in self._sprites:
            r0, r1, c0, c1 = (r - 1, r + 2, c - 1, c + 2) if tile.kind == TileType.BRIDGE else (r, r + 1, c, c + 1)
            xy = self.project([(rr, cc, z) for rr in (r0, r1) for cc in (c0, c1) for z in (-2 * TILE_THICK - 0.1, 0.5)])
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
        (nghiêng, lật cầu) = vẽ gấp 3 trong khung nhỏ riêng. Quầng sáng đậm nhạt theo số tile đã vào chỗ."""
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
            if f is not None and (f.xform is not None or f.theta is not None):
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

    def _draw_tiles(self, surface: pygame.Surface, bridge_mask: int, fx: Mapping[Coord, TileFx]) -> None:
        b = self.board
        order = sorted(((r, c) for r in range(b.height) for c in range(b.width)),
                       key=lambda rc: self._cell_depth(*rc))  # back to front
        # Lượt 1: cầu đang gập dưới tile kề -> vẽ trước để tile kề che lên trên
        main = []
        for r, c in order:
            tile = b.tiles[r][c]
            if tile.kind == TileType.BRIDGE and self._is_under(tile, bridge_mask, fx.get((r, c))):
                self._draw_tile(surface, r, c, tile, bridge_mask, fx.get((r, c)))
            else:
                main.append((r, c))
        for r, c in main:
            self._draw_tile(surface, r, c, b.tiles[r][c], bridge_mask, fx.get((r, c)))

    def draw_scene(self, surface: pygame.Surface, bridge_mask: int, boxes: Sequence[Box],
                   alpha: int = 255, tile_fx: Optional[Mapping[Coord, TileFx]] = None) -> None:
        """Vẽ cảnh (box và bảng).
        Gồm các bước: phần box dưới z=0, sàn, bóng đổ, phần box trên z=0.
        tile_fx: hiệu ứng riêng cho từng tile (xem TileFx). Không có hiệu ứng nào -> dùng ảnh sàn
        tĩnh đã cache (mịn nhất, nhanh nhất); có hiệu ứng -> vẽ trực tiếp từng tile.
        """
        fx = {**self.ambient_fx, **(tile_fx or {})}
        self._draw_boxes(surface, boxes, alpha, below=True)
        if fx:
            self._draw_tiles_fx(surface, bridge_mask, fx)
        else:
            img, pos = self._static_layer(bridge_mask)
            surface.blit(img, pos)

        self._draw_shadows(surface, boxes, bridge_mask, fx, alpha)
        self._draw_boxes(surface, boxes, alpha, below=False)

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
        if fx is not None and (fx.hidden or fx.theta is not None):
            return fx.under
        return bool(mask & self.board.bridge_bit[tile.tile_id])  # đóng = đang gập dưới

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
                self._draw_box(layer, bx.verts, bx.color, below)
        img = pygame.transform.smoothscale(layer, (w, h))
        if alpha < 255:
            img.set_alpha(max(0, alpha))
        surface.blit(img, (x0, y0))

    # --------------------------------------------------------------- drawing
    def _draw_box(self, surface: pygame.Surface, verts: np.ndarray, color: Color,
                  below: Optional[bool] = None, bevel: bool = True, key: Optional[tuple] = None) -> None:
        """Vẽ khối hộp.
        below=None --> Vẽ cả khối
        below=False/True --> vẽ phần hộp nửa trên/dưới z=0
        key: nếu có (tile tĩnh) thì lưu các lệnh vẽ đã chiếu sẵn để các frame sau chỉ việc phát lại.
        Viền khử răng cưa (aaline); mặt trên có bevel: vạch sáng ở cạnh hướng đèn, vạch tối ở cạnh khuất.
        """
        ops = self._cache.get(key) if key is not None else None
        if ops is None:
            ops = self._box_ops(verts, color, below, bevel)
            if key is not None:
                self._cache[key] = ops
        self._replay(surface, ops)

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

    def _flat(self, surface, cells, color, z=0.0, width=0) -> None:
        pts = self.project([(r, c, z) for r, c in cells]).tolist()
        pygame.draw.polygon(surface, color, pts, width)
        if width == 0 and self._ss == 1:
            pygame.draw.aalines(surface, color, True, pts)

    def _line(self, surface, a, b, color, width=3, z=0.0) -> None:
        p = self.project([(a[0], a[1], z), (b[0], b[1], z)]).tolist()
        pygame.draw.line(surface, color, p[0], p[1], width * self._ss)

    def _draw_tile(self, surface, r: int, c: int, tile: Tile, mask: int,
                   fx: Optional[TileFx] = None) -> None:
        kind = tile.kind
        if kind == TileType.VOID or (fx is not None and fx.hidden):
            return
        theta = 0.0
        if kind == TileType.BRIDGE:
            closed = bool(mask & self.board.bridge_bit[tile.tile_id])
            if (r, c) in self.hinges:
                theta = fx.theta if fx is not None and fx.theta is not None else (math.pi if closed else 0.0)
            elif closed and (fx is None or fx.theta is None):  # không có chỗ gập: cầu đóng = ô trống
                return
            color = PALETTE["bridge"]
        elif kind == TileType.FRAGILE:
            color = PALETTE["fragile"]
        else:
            color = PALETTE["floor"]
            if (r + c) % 2:
                color = shade(color, 0.93)  # subtle checkerboard

        dz = fx.dz if fx is not None else 0.0
        key = (r, c, color, round(theta, 4), self._ss, self._origin) if fx is None or (fx.xform is None and not dz and fx.theta is None) else None
        if key is not None and key in self._cache:
            self._replay(surface, self._cache[key])
        else:
            verts = make_box(r, c, -TILE_THICK, 1, 1, TILE_THICK)
            if theta > 1e-6:
                verts = self._hinge_rotate(verts, (r, c), theta)
            if fx is not None and fx.xform is not None:
                verts = fx.xform(verts)
            if dz:
                verts = verts.copy()
                verts[:, 2] += dz
            self._draw_box(surface, verts, color, key=key)
        cr, cc = r + 0.5, c + 0.5

        if kind == TileType.GOAL:
            m = 0.12
            self._flat(surface, [(r + m, c + m), (r + m, c + 1 - m), (r + 1 - m, c + 1 - m), (r + 1 - m, c + m)],
                       PALETTE["goal_hole"], z=0.002 + dz)
            m = 0.28
            self._flat(surface, [(r + m, c + m), (r + m, c + 1 - m), (r + 1 - m, c + 1 - m), (r + 1 - m, c + m)],
                       PALETTE["goal_inner"], z=0.003 + dz)
        elif kind == TileType.SWITCH:
            sc = PALETTE[tile.switch_kind.value]
            if tile.switch_kind == SwitchKind.SOFT:  # octagon
                pts = [(cr + 0.28 * math.sin(a), cc + 0.28 * math.cos(a))
                       for a in (i * math.pi / 4 + math.pi / 8 for i in range(8))]
                self._flat(surface, pts, sc, z=0.002 + dz)
            elif tile.switch_kind == SwitchKind.HEAVY:  # X
                d = 0.28
                self._line(surface, (cr - d, cc - d), (cr + d, cc + d), sc, 4, 0.002 + dz)
                self._line(surface, (cr - d, cc + d), (cr + d, cc - d), sc, 4, 0.002 + dz)
            else:  # split: "( )"
                for shift, a0, a1 in ((0.25, 140, 220), (-0.25, -40, 40)):
                    arc = [(cr + 0.55 * math.sin(math.radians(a)), cc + shift + 0.55 * math.cos(math.radians(a)))
                           for a in np.linspace(a0, a1, 7)]
                    for p, q in zip(arc, arc[1:]):
                        self._line(surface, p, q, sc, 3, 0.002 + dz)