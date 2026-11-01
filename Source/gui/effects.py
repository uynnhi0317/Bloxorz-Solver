"""effects.py - Hiệu ứng ánh sáng cộng (additive) vẽ ĐÈ lên cảnh, không nằm trong cache sàn.

    draw_under(): bloom trên sàn (goal / fragile / split), tia sáng goal, quầng sáng quanh khối
                  -> gọi sau khi vẽ sàn + bóng, TRƯỚC khi vẽ khối.
    draw_over():  hạt bay (goal / ember / split)  -> gọi sau khi vẽ khối.

Glow dùng sprite gradient mềm (falloff bậc 2.2) cache theo (kích thước, màu, độ sáng đã lượng tử hoá),
nên rìa luôn tan dần vào nền. Thời gian lấy từ pygame.time nên renderer không cần được update().
renderer.goal_boost (0..1): goal sáng hơn / tia cao hơn / nhiều hạt hơn (đặt khi thắng).
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pygame

from game import SwitchKind, TileType
from .theme import PALETTE, WATER_Z
ADD = pygame.BLEND_RGB_ADD

GOAL_COL = (255, 226, 150)
GOAL_COL_HI = (255, 250, 226)
SPLIT_COL = (186, 182, 255)
WATER_COL = (50, 150, 210)
WATER_HI = (170, 235, 255)
WATER_REACH = (3, 5, 6)   # tầm nhoè 3 lớp (đơn vị tile)
WATER_W = (1.0, 0.9, 0.7)       # tổng 2.6 -> tại bờ ~1.3: giữ sáng hết mức một đoạn ngoài bờ rồi mới mờ
WATER_GAIN = 0.65
WATER_ENABLED = True

# màu khối trong PALETTE -> (màu glow, độ sáng)
BLOCK_GLOW = {
    tuple(PALETTE["block"]): ((190, 210, 255), 0.50),
    tuple(PALETTE["cube_active"]): ((170, 205, 255), 0.55),
    tuple(PALETTE["cube_idle"]): ((120, 160, 215), 0.40),
}


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _q8(v: float) -> int:
    return max(8, int(v) // 8 * 8)

def _box_blur(a: np.ndarray, rad: int) -> np.ndarray:
    """Nhoè hộp tách trục bằng cumsum. a: (w, h) float."""
    k = 2 * rad + 1
    for axis in (0, 1):
        n = a.shape[axis]
        pad = [(rad, rad) if ax == axis else (0, 0) for ax in (0, 1)]
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
        c = np.insert(c, 0, 0.0, axis=axis)
        a = (np.take(c, range(k, k + n), axis=axis) - np.take(c, range(0, n), axis=axis)) / k
    return a


class Effects:
    MAX_SPRITES = 400

    def __init__(self, renderer) -> None:
        self.r = renderer
        b = renderer.board
        self.fragile = [(r, c) for r, row in enumerate(b.tiles) for c, t in enumerate(row)
                        if t.kind == TileType.FRAGILE]
        self.splits = [(r, c) for r, row in enumerate(b.tiles) for c, t in enumerate(row)
                       if t.kind == TileType.SWITCH and t.switch_kind == SwitchKind.SPLIT]
        self._sp: Dict[tuple, pygame.Surface] = {}
        self._boost = 0.0
        self._last = None
        self._t = 0.0
        self._water: Dict[tuple, tuple] = {}     # (mask, gone) -> (layer, pos)
        self._shore: Dict[tuple, list] = {}      # (mask, gone) -> danh sách cạnh bờ
        # cầu: bit -> các ô cầu; dùng cho hiệu ứng "công tắc -> cầu"
        self._bridge_cells: Dict[int, list] = {}
        for r, row in enumerate(b.tiles):
            for c, t in enumerate(row):
                if t.kind == TileType.BRIDGE:
                    self._bridge_cells.setdefault(b.bridge_bit[t.tile_id], []).append((r, c))
        self._last_mask = None
        self._pulses: list = []                  # [(switch_cells, bridge_cells, t0)]
        self._hulls: Dict[tuple, list] = {}      # ô -> bao lồi trên màn hình (để che đường bờ)
        self._solid: Dict[tuple, set] = {}
        self._shore_layer = None

    def reset(self) -> None:
        self._sp.clear()
        self._water = {}
        self._shore = {}
        self._last_mask = None
        self._pulses = []
        self._hulls = {}
        self._solid = {}
        self._shore_layer = None

    # ------------------------------------------------------------ sprites
    def _trim(self) -> None:
        if len(self._sp) > self.MAX_SPRITES:
            self._sp.clear()

    def _blob(self, rx: float, ry: float, color, gain: float):
        rx, ry = max(2, int(rx)), max(2, int(ry))
        gain = round(gain * 20) / 20
        if gain <= 0:
            return None
        key = ("b", rx, ry, tuple(color), gain)
        spr = self._sp.get(key)
        if spr is None:
            self._trim()
            yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
            d = np.sqrt((xx / rx) ** 2 + (yy / ry) ** 2)
            a = np.clip(1 - d, 0, 1) ** 2.2 * gain
            arr = (a[..., None] * np.array(color, dtype=float)).clip(0, 255).astype(np.uint8)
            spr = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            self._sp[key] = spr
        return spr

    def _beam(self, w: int, h: int, color, gain: float):
        gain = round(gain * 20) / 20
        key = ("beam", w, h, tuple(color), gain)
        spr = self._sp.get(key)
        if spr is None:
            self._trim()
            x = np.linspace(-1, 1, w)[None, :]
            u = np.linspace(1, 0, h)[:, None]               # hàng 0 = đỉnh (u=1), hàng cuối = đáy (u=0)
            sig = 0.28 + 0.30 * u                           # loe dần lên trên
            core = np.exp(-(x / sig) ** 2 * 2.2)
            axis = np.exp(-(x / 0.12) ** 2 * 2.0)
            a = (core * 0.8 + axis * 0.6 * (1 - u)) * (1 - u) ** 1.4 * np.clip(u / 0.05, 0, 1) * gain
            arr = (a[..., None] * np.array(color, dtype=float)).clip(0, 255).astype(np.uint8)
            spr = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            self._sp[key] = spr
        return spr

    def _glow(self, surf, x, y, rx, ry, color, gain) -> None:
        spr = self._blob(rx, ry, color, gain)
        if spr is not None:
            surf.blit(spr, (int(x - spr.get_width() / 2), int(y - spr.get_height() / 2)), special_flags=ADD)

    # ------------------------------------------------------------ time
    def _tick(self) -> None:
        now = pygame.time.get_ticks() / 1000.0
        dt = 0.0 if self._last is None else min(0.1, max(0.0, now - self._last))
        self._last, self._t = now, now
        target = float(getattr(self.r, "goal_boost", 0.0))
        self._boost += (target - self._boost) * (1 - math.exp(-dt * 4.0))

    def _level(self) -> float:
        return round(self._boost * 8) / 8

    # ------------------------------------------------------------ particles
    def _sparks(self, surf, cr, cc, dz, n, T, hmax, spread, color, size, gain, seed) -> None:
        pts, meta = [], []
        for i in range(n):
            ph = (i * 0.6180339 + seed) % 1.0
            u = ((self._t / T) + ph) % 1.0
            ang = i * 2.399963 + seed * 6.28
            rad = spread * (0.35 + 0.65 * ((i * 0.37 + seed) % 1.0))
            dr = rad * math.sin(ang) + 0.04 * math.sin(self._t * 1.4 + i)
            dc = rad * math.cos(ang) + 0.04 * math.cos(self._t * 1.1 + i)
            pts.append((cr + dr, cc + dc, 0.05 + u * hmax + dz))
            meta.append((max(2, int(self.r.scale * size[i % len(size)])), math.sin(math.pi * u) ** 0.9 * gain))
        if not pts:
            return
        for (x, y), (sz, a) in zip(self.r.project(pts), meta):
            self._glow(surf, x, y, sz, sz, color, a)

    # ------------------------------------------------------------ goal
    def _goal_state(self, fx: Mapping):
        gr, gc = self.r.board.goal
        f = fx.get((gr, gc))
        if f is None:
            return gr, gc, 0.0, 1.0
        if f.hidden:
            return gr, gc, 0.0, 0.0
        return gr, gc, f.dz, max(0.0, 1.0 + f.dz / 12.0)    # tile đang bay vào / rơi ra: mờ theo

    # ------------------------------------------------------------ public
    def draw_under(self, surf, boxes: Sequence, mask: int, fx: Mapping, alpha: int) -> None:
        self._tick()
        self._detect_link(boxes, mask, fx)
        r, sc = self.r, self.r.scale
        for (rr, cc), f in (fx.items() if WATER_ENABLED else ()):          # tile đang xuyên qua mặt nước: loé sáng ở chân tile
            if f.hidden or not f.dz:
                continue
            k = max(0.0, 1.0 - abs(f.dz - WATER_Z) / 1.2)
            if k > 0.02:
                x, y = r.project([(rr + 0.5, cc + 0.5, WATER_Z)])[0]
                self._glow(surf, x, y, 0.9 * sc, 0.5 * sc, WATER_HI, 0.55 * k)

        L = self._level()

        # goal: bloom trên sàn + tia sáng
        gr, gc, gdz, strength = self._goal_state(fx)
        if strength > 0.02:
            cx, cy = r.project([(gr + 0.5, gc + 0.5, gdz)])[0]
            pulse = 0.92 + 0.08 * math.sin(self._t * 2.0)
            self._glow(surf, cx, cy, 1.9 * sc, 1.05 * sc, GOAL_COL, (0.80 + 0.20 * L) * strength * pulse)
            self._glow(surf, cx, cy, 0.8 * sc, 0.45 * sc, GOAL_COL_HI, 0.85 * strength)
            if strength > 0.98:
                hz = 3.2 + 3.0 * L
                w = max(8, int(1.5 * sc) // 4 * 4)
                h = max(16, int(hz * abs(float(r._m[1, 2])) * sc) // 8 * 8)
                beam = self._beam(w, h, _lerp(GOAL_COL, GOAL_COL_HI, L), 1.0)
                surf.blit(beam, (int(cx - w / 2), int(cy - h)), special_flags=ADD)

        # split switch: quầng sáng tím trắng
        for cell in self.splits:
            if cell in fx:
                continue
            pulse = 0.5 + 0.5 * math.sin(self._t * 1.4 + cell[0] + cell[1])
            cx, cy = r.project([(cell[0] + 0.5, cell[1] + 0.5, 0.02)])[0]
            self._glow(surf, cx, cy, 0.62 * sc, 0.36 * sc, SPLIT_COL, 0.26 + 0.12 * round(pulse * 4) / 4)

        # quầng sáng quanh khối
        for bx in boxes:
            spec = BLOCK_GLOW.get(tuple(bx.color))
            if spec is None:
                continue
            xy = r.project(bx.verts)
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            cx, cy = (lo + hi) / 2
            rx = _q8(0.5 * (hi[0] - lo[0]) + 0.6 * sc)
            ry = _q8(0.5 * (hi[1] - lo[1]) + 0.55 * sc)
            self._glow(surf, cx, cy, rx, ry, spec[0], spec[1] * alpha / 255.0)

    def draw_over(self, surf, mask: int, fx: Mapping, alpha: int) -> None:
        r = self.r
        L = self._level()

        gr, gc, gdz, strength = self._goal_state(fx)
        if strength > 0.98:
            self._sparks(surf, gr + 0.5, gc + 0.5, gdz, int(14 + 12 * L), 2.6 - 0.8 * L, 2.6 + 2.2 * L, 0.30,
                         _lerp(GOAL_COL, GOAL_COL_HI, L), (0.045, 0.065, 0.09), 0.7 + 0.3 * L, 0.0)

        for cell in self.splits:
            if cell in fx:
                continue
            self._sparks(surf, cell[0] + 0.5, cell[1] + 0.5, 0.0, 3, 3.6, 1.0, 0.25,
                         SPLIT_COL, (0.03, 0.045), 0.8, (cell[0] * 0.29 + cell[1] * 0.53) % 1.0)
        self._draw_links(surf)

    # ------------------------------------------------------------ switch -> bridge link
    LINK_MS = 1100.0
    GOLD = (255, 214, 110)

    def _detect_link(self, boxes, mask: int, fx: Mapping) -> None:
        """Mask đổi (công tắc vừa bật) -> tạo xung sáng chạy từ công tắc tới các cầu bị tác động."""
        last, self._last_mask = self._last_mask, mask
        if last is None or last == mask:
            return
        bridges = {c for cells in self._bridge_cells.values() for c in cells}
        if any(cell not in bridges for cell in fx):      # đang intro/outro/rơi: chỉ đồng bộ, không phát xung
            return
        diff = mask ^ last
        cells = [c for bit, cs in self._bridge_cells.items() if diff & bit for c in cs]
        if not cells:
            return
        cand = [cell for cell, ms in self.r.board.switch_masks.items()
                if any(m & diff for m in ms)]
        under = {cell for bx in boxes for cell in self.r._cells_under(bx)}
        sw = [c for c in cand if c in under] or cand
        self._pulses.append((sw, cells, self._t))

    def _draw_links(self, surf) -> None:
        if not self._pulses:
            return
        r, sc, now = self.r, self.r.scale, self._t
        self._pulses = [p for p in self._pulses if (now - p[2]) * 1000 < self.LINK_MS]
        for sw, cells, t0 in self._pulses:
            a = (now - t0) * 1000 / self.LINK_MS                  # 0..1
            for s in sw:                                           # công tắc loé lên rồi tắt
                x, y = r.project([(s[0] + 0.5, s[1] + 0.5, 0.05)])[0]
                self._glow(surf, x, y, 0.75 * sc, 0.45 * sc, self.GOLD, 0.9 * math.sin(math.pi * min(1.0, a * 1.6)))
            for cell in cells:
                # tia sáng bay từ công tắc tới cầu (0 .. 0.45), sau đó cầu loé (0.35 .. 1)
                if sw:
                    s = sw[0]
                    for j in range(5):
                        u = a / 0.45 - j * 0.07
                        if not 0.0 < u < 1.0:
                            continue
                        u = u * u * (3 - 2 * u)
                        pr = s[0] + (cell[0] - s[0]) * u + 0.5
                        pc = s[1] + (cell[1] - s[1]) * u + 0.5
                        x, y = r.project([(pr, pc, 0.35)])[0]
                        d = max(3, int(sc * (0.09 - 0.01 * j)))
                        self._glow(surf, x, y, d, d, self.GOLD, 1.0 - 0.18 * j)
                k = math.sin(math.pi * min(1.0, max(0.0, (a - 0.35) / 0.65)))
                if k > 0.02:
                    x, y = r.project([(cell[0] + 0.5, cell[1] + 0.5, 0.05)])[0]
                    self._glow(surf, x, y, 0.8 * sc, 0.48 * sc, self.GOLD, 0.85 * k)

    # ------------------------------------------------------------ water
    def _gone(self, fx: Mapping):
        """Các ô fragile đang rơi / đã vỡ: không còn là mặt đất nên không có nước / bờ quanh nó."""
        out = []
        for cell in self.fragile:
            f = fx.get(cell)
            if f is not None and (f.hidden or f.xform is not None):
                out.append(cell)
        return frozenset(out)

    def _solid_cells(self, mask: int, gone=frozenset()):
        b = self.r.board
        out = set()
        for i, row in enumerate(b.tiles):
            for j, t in enumerate(row):
                if t.kind == TileType.VOID or (i, j) in gone:
                    continue
                if t.kind == TileType.BRIDGE and mask & b.bridge_bit[t.tile_id]:
                    continue                      # cầu đang đóng = ô trống
                out.add((i, j))
        return out

    def _shore_edges(self, mask: int, gone=frozenset()):
        """[(a, b, front, cell)]: cạnh ô giáp void / ngoài bảng; front = mặt bên quay về camera."""
        key = (mask, gone)
        if key not in self._shore:
            if len(self._shore) > 12:
                self._shore.clear()
            solid, vd, out = self._solid_cells(mask, gone), self.r.view_dir, []
            for i, j in solid:
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (i + di, j + dj) in solid:
                        continue
                    front = di * vd[0] + dj * vd[1] > 1e-6
                    if di:
                        rr = i + (1 if di > 0 else 0)
                        out.append(((rr, j), (rr, j + 1), front, (i, j)))
                    else:
                        cc = j + (1 if dj > 0 else 0)
                        out.append(((i, cc), (i + 1, cc), front, (i, j)))
            self._shore[key] = out
        return self._shore[key]

    def _water_layer(self, mask: int, gone=frozenset()):
        """Ảnh glow nước (nền đen, blit cộng sáng). Tính ở 1/Q độ phân giải rồi phóng lên."""
        key = (mask, gone)
        if key not in self._water:
            if len(self._water) >= 8:
                self._water.pop(next(iter(self._water)))
            r, sc, Q = self.r, self.r.scale, 4
            rect = r._board_rect().inflate(int(6 * sc), int(5 * sc))
            w, h = rect.size
            surf = pygame.Surface((max(1, w // Q), max(1, h // Q)))
            for i, j in self._solid_cells(mask, gone):
                pts = r.project([(i, j, WATER_Z), (i, j + 1, WATER_Z), (i + 1, j + 1, WATER_Z), (i + 1, j, WATER_Z)])
                pygame.draw.polygon(surf, (255, 255, 255),
                                    [((x - rect.x) / Q, (y - rect.y) / Q) for x, y in pts.tolist()])
            base = pygame.surfarray.array3d(surf)[..., 0].astype(float) / 255.0
            total = np.zeros_like(base)
            for reach, wt in zip(WATER_REACH, WATER_W):
                rad = max(1, int(reach * sc / 2 / Q))
                total += wt * _box_blur(_box_blur(base, rad), rad)
            g = np.clip(total, 0.0, 1.0) ** 1.3 * WATER_GAIN
            lo, hi = np.array(WATER_COL, float), np.array(WATER_HI, float)
            col = lo + (hi - lo) * (g[..., None] ** 2)
            small = pygame.surfarray.make_surface((g[..., None] * col).clip(0, 255).astype(np.uint8))
            self._water[key] = (pygame.transform.smoothscale(small, (w, h)).convert(), rect.topleft)
        return self._water[key]

    def warm(self, mask: int = 0) -> None:
        if not WATER_ENABLED:
            return
        self._water_layer(mask)

    def draw_water(self, surf, mask: int = 0, fx: Mapping = {}) -> None:
        """Gọi TRƯỚC khi vẽ sàn."""
        if not WATER_ENABLED:
            return
        layer, pos = self._water_layer(mask, self._gone(fx))
        surf.blit(layer, pos, special_flags=ADD)

    def _tile_hull(self, cell, dz: float = 0.0):
        """Bao lồi (trên màn hình) của khối tile: dùng để che đường bờ bằng các tile ở gần camera hơn."""
        from .renderer import _hull, make_box, TILE_THICK
        key = (cell, round(dz, 3))
        h = self._hulls.get(key)
        if h is None:
            if len(self._hulls) > 400:
                self._hulls.clear()
            v = make_box(cell[0], cell[1], -TILE_THICK, 1, 1, TILE_THICK)
            v[:, 2] += dz
            h = _hull([tuple(p) for p in self.r.project(v).tolist()])
            self._hulls[key] = h
        return h

    def draw_shore(self, surf, mask: int, fx: Mapping) -> None:
        """Đường mực nước ở mép sàn. Vẽ theo thứ tự xa -> gần như vẽ tile: mỗi tile trước hết xoá phần đường
        của các tile xa hơn mà nó che, rồi mới vẽ đường của chính nó. Gọi ngay SAU khi vẽ sàn, TRƯỚC bóng đổ."""
        if not WATER_ENABLED:
            return
        self._tick()
        r, sc = self.r, self.r.scale
        gone = self._gone(fx)
        by_cell: Dict[tuple, list] = {}
        for n, (a, b, front, cell) in enumerate(self._shore_edges(mask, gone)):
            if front:
                by_cell.setdefault(cell, []).append((n, a, b))
        if not by_cell:
            return
        key = (mask, gone)
        solid = self._solid.get(key)
        if solid is None:
            if len(self._solid) > 12:
                self._solid.clear()
            solid = self._solid[key] = self._solid_cells(mask, gone)
        rect = r._board_rect().inflate(int(2 * sc), int(2 * sc))
        if self._shore_layer is None or self._shore_layer.get_size() != rect.size:
            self._shore_layer = pygame.Surface(rect.size)
        layer = self._shore_layer
        layer.fill((0, 0, 0))
        ox, oy = rect.topleft
        w = max(1, int(sc / 32))
        for cell in r._order():                               # xa -> gần
            if cell not in solid:
                continue
            f, e, dz = fx.get(cell), 1.0, 0.0
            if f is not None:
                if f.hidden or f.xform is not None or f.theta is not None:
                    continue                                   # tile ẩn / đang rơi / cầu đang lật: không che, không có bờ
                dz = f.dz
                e = max(0.0, min(1.0, 1.0 + dz / 1.5))        # tile còn chìm sâu -> chưa có đường mực nước
            pygame.draw.polygon(layer, (0, 0, 0), [(x - ox, y - oy) for x, y in self._tile_hull(cell, dz)])
            if e <= 0.02:
                continue
            for n, a, b in by_cell.get(cell, ()):
                pulse = 0.5 + 0.5 * math.sin(self._t * 1.6 + n * 0.7)
                p = r.project([(a[0], a[1], WATER_Z), (b[0], b[1], WATER_Z)])
                for t in (0.2, 0.5, 0.8):
                    x, y = p[0] + (p[1] - p[0]) * t
                    self._glow(layer, x - ox, y - oy, 0.55 * sc, 0.32 * sc, WATER_HI,
                               (0.30 + 0.15 * round(pulse * 4) / 4) * e)
                pygame.draw.line(layer, tuple(int(v * e * 0.7) for v in (190, 240, 255)),
                                 (p[0][0] - ox, p[0][1] - oy), (p[1][0] - ox, p[1][1] - oy), w)
        surf.blit(layer, rect.topleft, special_flags=ADD)