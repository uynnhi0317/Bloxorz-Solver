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
EMBER_COL = (255, 112, 40)
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
        self._water = None
        self._shore = None
        self._shore_cache = None

    def reset(self) -> None:
        self._sp.clear()
        self._water = None
        self._shore_cache = None

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
        self.draw_shore(surf, fx)
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

        # fragile: than hồng thở nhẹ
        for cell in self.fragile:
            if cell in fx:
                continue
            ph = cell[0] * 1.7 + cell[1] * 2.3
            pulse = 0.5 + 0.5 * math.sin(self._t * 1.7 + ph)
            cx, cy = r.project([(cell[0] + 0.5, cell[1] + 0.5, 0.02)])[0]
            self._glow(surf, cx, cy, 0.66 * sc, 0.38 * sc, EMBER_COL, 0.20 + 0.14 * round(pulse * 4) / 4)

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

        for cell in self.fragile:
            if cell in fx:
                continue
            self._sparks(surf, cell[0] + 0.5, cell[1] + 0.5, 0.0, 2, 3.2, 1.3, 0.30,
                         (255, 150, 60), (0.03, 0.04), 0.9, (cell[0] * 0.37 + cell[1] * 0.61) % 1.0)

        for cell in self.splits:
            if cell in fx:
                continue
            self._sparks(surf, cell[0] + 0.5, cell[1] + 0.5, 0.0, 3, 3.6, 1.0, 0.25,
                         SPLIT_COL, (0.03, 0.045), 0.8, (cell[0] * 0.29 + cell[1] * 0.53) % 1.0)

        # ------------------------------------------------------------ water
    def _solid_cells(self):
        return {(i, j) for i, row in enumerate(self.r.board.tiles)
                for j, t in enumerate(row) if t.kind != TileType.VOID}

    def _shore_edges(self):
        """[(a, b, front, cell)]: cạnh ô giáp void / ngoài bảng; front = mặt bên quay về camera."""
        if self._shore is None:
            solid, vd, out = self._solid_cells(), self.r.view_dir, []
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
            self._shore = out
        return self._shore

    def _water_layer(self):
        """Ảnh glow nước (nền đen, blit cộng sáng). Tính ở 1/Q độ phân giải rồi phóng lên."""
        if self._water is None:
            r, sc, Q = self.r, self.r.scale, 4
            rect = r._board_rect().inflate(int(6 * sc), int(5 * sc))
            w, h = rect.size
            mask = pygame.Surface((max(1, w // Q), max(1, h // Q)))
            for i, j in self._solid_cells():
                pts = r.project([(i, j, WATER_Z), (i, j + 1, WATER_Z), (i + 1, j + 1, WATER_Z), (i + 1, j, WATER_Z)])
                pygame.draw.polygon(mask, (255, 255, 255),
                                    [((x - rect.x) / Q, (y - rect.y) / Q) for x, y in pts.tolist()])
            base = pygame.surfarray.array3d(mask)[..., 0].astype(float) / 255.0
            total = np.zeros_like(base)
            for reach, wt in zip(WATER_REACH, WATER_W):
                rad = max(1, int(reach * sc / 2 / Q))
                total += wt * _box_blur(_box_blur(base, rad), rad)
            g = np.clip(total, 0.0, 1.0) ** 1.3 * WATER_GAIN
            lo, hi = np.array(WATER_COL, float), np.array(WATER_HI, float)
            col = lo + (hi - lo) * (g[..., None] ** 2)
            small = pygame.surfarray.make_surface((g[..., None] * col).clip(0, 255).astype(np.uint8))
            self._water = (pygame.transform.smoothscale(small, (w, h)).convert(), rect.topleft)
        return self._water

    def warm(self) -> None:
        if not WATER_ENABLED:
            return
        self._water_layer()

    def draw_water(self, surf) -> None:
        """Gọi TRƯỚC khi vẽ sàn."""
        if not WATER_ENABLED:
            return
        layer, pos = self._water_layer()
        surf.blit(layer, pos, special_flags=ADD)
        

    def _shore_px(self):
        """Toạ độ màn hình của các cạnh bờ phía trước (camera cố định nên tính một lần)."""
        if self._shore_cache is None:
            r, out = self.r, []
            for n, (a, b, front, cell) in enumerate(self._shore_edges()):
                if front:
                    p = r.project([(a[0], a[1], WATER_Z), (b[0], b[1], WATER_Z)])
                    out.append((n, cell, p[0].tolist(), p[1].tolist()))
            self._shore_cache = out
        return self._shore_cache

    # def draw_shore(self, surf, fx) -> None:
    #     if not WATER_ENABLED:
    #         return
    #     sc = self.r.scale
    #     w = max(1, int(sc / 32))
    #     for n, cell, p0, p1 in self._shore_px():
    #         f, e = fx.get(cell), 1.0
    #         if f is not None:
    #             if f.hidden:
    #                 continue
    #             e = max(0.0, min(1.0, 1.0 + f.dz / 1.5))
    #             if e <= 0.02:
    #                 continue
    #         pulse = 0.5 + 0.5 * math.sin(self._t * 1.6 + n * 0.7)
    #         self._glow(surf, (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0.7 * sc, 0.32 * sc,
    #                    WATER_HI, (0.32 + 0.15 * round(pulse * 4) / 4) * e)
    #         pygame.draw.line(surf, tuple(int(v * e) for v in (190, 240, 255)), p0, p1, w)

    def draw_shore(self, surf, fx) -> None:
        r, sc = self.r, self.r.scale
        w = max(1, int(sc / 32))
        for n, (a, b, front, cell) in enumerate(self._shore_edges()):
            if not front:
                continue
            f, e = fx.get(cell), 1.0
            if f is not None:
                if f.hidden:
                    continue
                e = max(0.0, min(1.0, 1.0 + f.dz / 1.5))   # tile còn chìm sâu -> chưa có đường mực nước
                if e <= 0.02:
                    continue
            pulse = 0.5 + 0.5 * math.sin(self._t * 1.6 + n * 0.7)
            p = r.project([(a[0], a[1], WATER_Z), (b[0], b[1], WATER_Z)])
            for t in (0.2, 0.5, 0.8):
                x, y = p[0] + (p[1] - p[0]) * t
                self._glow(surf, x, y, 0.55 * sc, 0.32 * sc, WATER_HI, (0.30 + 0.15 * round(pulse * 4) / 4) * e)
            pygame.draw.line(surf, tuple(int(v * e) for v in (190, 240, 255)), p[0], p[1], w)