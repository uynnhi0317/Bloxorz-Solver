"""textures.py - Procedural tile art (Hollow Knight / Silksong flavour, no original assets).

Renderer gọi:
    art.body_color(r, c, tile)                  -> màu thân (cạnh bên) của tile
    art.decorate(surface, r, c, tile, theta, dz) -> vẽ hoa văn lên mặt trên của tile

Mọi hoa văn là đa giác / đường vẽ trong toạ độ (row, col) của ô, chiếu qua camera của renderer,
nên hoạt động đúng với supersample, sprite cache, tile bay/rơi (dz) và cầu lật (theta).
Số ngẫu nhiên được seed theo (r, c) -> hình dạng đá/vết nứt ổn định giữa các lần vẽ.
Glow / hạt bay nằm ở effects.py (vẽ cộng sáng, không cache).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame

from game import SwitchKind, Tile, TileType

Pt = Tuple[float, float]

CLASSIC = {"floor": (112, 128, 164), "floor_alt": (104, 119, 153),
           "fragile": (214, 140, 72), "crack": (122, 72, 34)}


def shade(c, f: float):
    return tuple(max(0, min(255, int(v * f))) for v in c)


ART = {
    # bridge (brass / gold)
    "bridge_body": (150, 112, 40), "bridge_dark": (112, 78, 26), "bridge_mid": (184, 138, 48),
    "bridge_light": (238, 196, 90), "bridge_hi": (255, 240, 170),
    # goal
    "goal_body": (58, 64, 86), "goal_rim": (214, 222, 244),
    # soft / heavy switch plate
    "plate_body": (52, 40, 36), "plate_dark": (36, 28, 28), "plate": (64, 50, 44),
    "rust": [(116, 64, 36), (92, 52, 34), (26, 20, 20)],
    "gold": (238, 192, 72), "gold_dark": (122, 90, 30), "gold_hi": (255, 242, 170),
    # split switch
    "split_body": (38, 38, 60), "split_plate": (46, 46, 72), "split_light": (228, 230, 255),
    "split_rim": (136, 136, 204),
}


# ---------------------------------------------------------------- geometry (row, col)
def rect(r: float, c: float, m: float) -> List[Pt]:
    return [(r + m, c + m), (r + m, c + 1 - m), (r + 1 - m, c + 1 - m), (r + 1 - m, c + m)]


def circ(cr: float, cc: float, rad: float, n: int = 14, sq: float = 1.0) -> List[Pt]:
    return [(cr + rad * sq * math.sin(2 * math.pi * i / n), cc + rad * math.cos(2 * math.pi * i / n))
            for i in range(n)]


def scale_pts(pts: Sequence[Pt], cr: float, cc: float, f: float) -> List[Pt]:
    return [(cr + (p[0] - cr) * f, cc + (p[1] - cc) * f) for p in pts]


def shift(pts: Sequence[Pt], dr: float, dc: float) -> List[Pt]:
    return [(p[0] + dr, p[1] + dc) for p in pts]

def jag(rng: random.Random, a: Pt, b: Pt, n: int, amp: float) -> List[Pt]:
    """Đường gấp khúc từ a đến b (vết nứt)."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    ln = math.hypot(dr, dc) or 1.0
    pr, pc = -dc / ln, dr / ln
    out = [a]
    for i in range(1, n):
        t = i / n
        o = rng.uniform(-amp, amp)
        out.append((a[0] + dr * t + pr * o, a[1] + dc * t + pc * o))
    out.append(b)
    return out


def edge_pt(rng: random.Random, r: int, c: int, side: int) -> Pt:
    u = rng.uniform(0.12, 0.88)
    return [(r + 0.03, c + u), (r + 0.97, c + u), (r + u, c + 0.03), (r + u, c + 0.97)][side]


class TileArt:
    def __init__(self, renderer) -> None:
        self.r = renderer

    # ------------------------------------------------------------ colours
    def body_color(self, r: int, c: int, tile: Tile):
        k = tile.kind
        if k == TileType.FRAGILE:
            return CLASSIC["fragile"]
        if k == TileType.BRIDGE:
            return ART["bridge_body"]
        if k == TileType.GOAL:
            return ART["goal_body"]
        if k == TileType.SWITCH:
            return ART["split_body"] if tile.switch_kind == SwitchKind.SPLIT else ART["plate_body"]
        return CLASSIC["floor_alt"] if (r + c) % 2 else CLASSIC["floor"]
    # ------------------------------------------------------------ drawing primitives
    def _pts(self, ctx, pts: Sequence[Pt], z: float):
        cell, theta, dz, xform = ctx
        arr = np.array([(p[0], p[1], z) for p in pts], dtype=float)
        if theta > 1e-6:
            arr = self.r._hinge_rotate(arr, cell, theta)
        if xform is not None:
            arr = xform(arr)
        arr[:, 2] += dz
        return self.r.project(arr).tolist()

    def poly(self, s, ctx, pts: Sequence[Pt], color, z: float = 0.003) -> None:
        if len(pts) >= 3:
            pygame.draw.polygon(s, color, self._pts(ctx, pts, z))

    def line(self, s, ctx, pts: Sequence[Pt], color, width: float, z: float = 0.004) -> None:
        p = self._pts(ctx, pts, z)
        w = max(1, int(round(width * self.r.scale / 64.0)))
        if len(p) >= 2:
            pygame.draw.lines(s, color, False, p, w)
        if w >= 3:
            rad = w // 2
            for q in p:
                pygame.draw.circle(s, color, (int(q[0]), int(q[1])), rad)

    # ------------------------------------------------------------ dispatcher
    def decorate(self, s, r: int, c: int, tile: Tile, theta: float, dz: float, xform=None) -> None:
        if theta > math.pi / 2 - 0.05: # cầu lật úp, mặt trên quay xuống, ko vẽ hoa văn
            return
        ctx = ((r, c), theta, dz, xform)
        k = tile.kind
        if k == TileType.FRAGILE:
            self._fragile_classic(s, ctx, r, c)
        elif k == TileType.BRIDGE:
            self._bridge(s, ctx, r, c)
        elif k == TileType.GOAL:
            self._goal(s, ctx, r, c)
        elif k == TileType.SWITCH:
            if tile.switch_kind == SwitchKind.SPLIT:
                self._split(s, ctx, r, c)
            else:
                self._plate(s, ctx, r, c)
                (self._soft if tile.switch_kind == SwitchKind.SOFT else self._heavy)(s, ctx, r, c)

    # ------------------------------------------------------------ fragile: bản gốc (cam hổ phách, vài vết nứt)
    def _fragile_classic(self, s, ctx, r, c) -> None:
        rng = random.Random(r * 6151 + c * 3571 + 5)
        for _ in range(3):
            a = edge_pt(rng, r, c, rng.randrange(4))
            b = (r + 0.5 + rng.uniform(-0.2, 0.2), c + 0.5 + rng.uniform(-0.2, 0.2))
            self.line(s, ctx, jag(rng, a, b, 4, 0.05), CLASSIC["crack"], 1.4)

    # ------------------------------------------------------------ bridge: đồng thau vàng
    def _bridge(self, s, ctx, r, c) -> None:
        cr, cc = r + 0.5, c + 0.5
        self.poly(s, ctx, rect(r, c, 0.04), ART["bridge_dark"])
        self.poly(s, ctx, rect(r, c, 0.09), ART["bridge_mid"])
        self.poly(s, ctx, rect(r, c, 0.16), ART["bridge_light"])
        # đường chỉ nổi: sáng ở cạnh trên-trái, tối ở cạnh dưới-phải
        self.line(s, ctx, [(r + 0.16, c + 0.16), (r + 0.16, c + 0.84)], ART["bridge_hi"], 1.2)
        self.line(s, ctx, [(r + 0.16, c + 0.16), (r + 0.84, c + 0.16)], ART["bridge_hi"], 1.2)
        self.line(s, ctx, [(r + 0.84, c + 0.16), (r + 0.84, c + 0.84)], ART["bridge_dark"], 1.2)
        self.line(s, ctx, [(r + 0.16, c + 0.84), (r + 0.84, c + 0.84)], ART["bridge_dark"], 1.2)
        # hoa văn thoi chính giữa + vòng tròn
        dia = lambda m: [(cr - m, cc), (cr, cc + m), (cr + m, cc), (cr, cc - m)]
        self.poly(s, ctx, dia(0.30), ART["bridge_dark"])
        self.poly(s, ctx, dia(0.255), ART["bridge_mid"])
        self.poly(s, ctx, dia(0.20), ART["bridge_light"])
        self.poly(s, ctx, circ(cr, cc, 0.095, 14), ART["bridge_dark"])
        self.poly(s, ctx, circ(cr - 0.006, cc - 0.006, 0.075, 14), ART["bridge_hi"])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):               # rãnh từ đỉnh thoi ra mép
            self.line(s, ctx, [(cr + dr * 0.31, cc + dc * 0.31), (cr + dr * 0.40, cc + dc * 0.40)],
                      ART["bridge_dark"], 1.4)
        for dr, dc in ((0.12, 0.12), (0.12, 0.88), (0.88, 0.12), (0.88, 0.88)):   # đinh tán
            self.poly(s, ctx, circ(r + dr + 0.012, c + dc + 0.012, 0.038, 8), ART["bridge_dark"])
            self.poly(s, ctx, circ(r + dr, c + dc, 0.034, 8), ART["bridge_hi"])

    # ------------------------------------------------------------ goal: lỗ sáng
    def _goal(self, s, ctx, r, c) -> None:
        cr, cc = r + 0.5, c + 0.5
        self.poly(s, ctx, circ(cr, cc, 0.45, 28), ART["goal_rim"])
        for i in range(12):                                              # vạch chia trên vành (kiểu đồng hồ)
            a = 2 * math.pi * i / 12
            self.line(s, ctx, [(cr + 0.37 * math.sin(a), cc + 0.37 * math.cos(a)),
                               (cr + 0.45 * math.sin(a), cc + 0.45 * math.cos(a))], (70, 78, 108), 1.3)
        self.poly(s, ctx, circ(cr, cc, 0.385, 28), (34, 38, 58))
        self.poly(s, ctx, circ(cr, cc, 0.34, 28), (255, 222, 146))
        self.poly(s, ctx, circ(cr, cc, 0.26, 24), (255, 237, 184))
        self.poly(s, ctx, circ(cr, cc, 0.17, 20), (255, 249, 222))
        self.poly(s, ctx, circ(cr, cc, 0.08, 14), (255, 255, 246))

    # ------------------------------------------------------------ soft / heavy: bản kim loại rỉ
    def _plate(self, s, ctx, r, c) -> None:
        rng = random.Random(r * 4001 + c * 977 + 3)
        self.poly(s, ctx, rect(r, c, 0.06), ART["plate_dark"])
        self.poly(s, ctx, rect(r, c, 0.11), ART["plate"])
        for _ in range(10):                                              # đốm rỉ
            col = rng.choice(ART["rust"])
            pr, pc = r + rng.uniform(0.15, 0.85), c + rng.uniform(0.15, 0.85)
            self.poly(s, ctx, circ(pr, pc, rng.uniform(0.012, 0.045), 7, rng.uniform(0.7, 1.2)), col)
        for _ in range(2):                                               # vệt rỉ chảy
            x = c + rng.uniform(0.2, 0.8)
            self.line(s, ctx, [(r + 0.14, x), (r + rng.uniform(0.4, 0.8), x + rng.uniform(-0.05, 0.05))],
                      ART["rust"][0], 1.1)
        for dr, dc in ((0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)):  # bu-lông
            self.poly(s, ctx, circ(r + dr + 0.01, c + dc + 0.01, 0.046, 8), (20, 16, 16))
            self.poly(s, ctx, circ(r + dr, c + dc, 0.042, 8), (92, 80, 70))
            self.poly(s, ctx, circ(r + dr - 0.01, c + dc - 0.01, 0.018, 6), (170, 150, 110))

    def _soft(self, s, ctx, r, c) -> None:
        cr, cc = r + 0.5, c + 0.5
        self.poly(s, ctx, circ(cr + 0.022, cc + 0.022, 0.27, 22), (18, 14, 14))
        self.poly(s, ctx, circ(cr, cc, 0.27, 22), ART["gold_dark"])
        self.poly(s, ctx, circ(cr - 0.008, cc - 0.008, 0.25, 22), ART["gold"])
        self.poly(s, ctx, circ(cr, cc, 0.17, 22), ART["plate_dark"])
        self.poly(s, ctx, circ(cr, cc, 0.135, 22), (52, 40, 34))
        self.poly(s, ctx, circ(cr - 0.008, cc - 0.008, 0.06, 14), ART["gold"])
        arc = [(cr + 0.215 * math.sin(math.radians(a)), cc + 0.215 * math.cos(math.radians(a)))
               for a in range(130, 251, 20)]
        self.line(s, ctx, arc, ART["gold_hi"], 1.4)

    def _heavy(self, s, ctx, r, c) -> None:
        cr, cc = r + 0.5, c + 0.5
        a = 0.25
        diags = (((cr - a, cc - a), (cr + a, cc + a)), ((cr - a, cc + a), (cr + a, cc - a)))
        for p, q in diags:
            self.line(s, ctx, [(p[0] + 0.022, p[1] + 0.022), (q[0] + 0.022, q[1] + 0.022)], (18, 14, 14), 7.0)
        for p, q in diags:
            self.line(s, ctx, [p, q], ART["gold_dark"], 6.2)
        for p, q in diags:
            self.line(s, ctx, [p, q], ART["gold"], 4.0)
        for p, q in diags:
            self.line(s, ctx, [(p[0] - 0.008, p[1] - 0.008), (q[0] - 0.008, q[1] - 0.008)], ART["gold_hi"], 1.2)
        self.poly(s, ctx, circ(cr, cc, 0.06, 12), ART["gold_dark"])
        self.poly(s, ctx, circ(cr - 0.006, cc - 0.006, 0.045, 12), ART["gold_hi"])

    # ------------------------------------------------------------ split: "linh hồn" tách đôi
    def _split(self, s, ctx, r, c) -> None:
        cr, cc = r + 0.5, c + 0.5
        self.poly(s, ctx, rect(r, c, 0.06), (24, 24, 42))
        self.poly(s, ctx, rect(r, c, 0.11), ART["split_plate"])
        self.poly(s, ctx, circ(cr, cc, 0.37, 26), (62, 60, 108))
        self.poly(s, ctx, circ(cr, cc, 0.33, 26), ART["split_plate"])
        ang = math.radians(32)
        nv = (math.cos(ang), -math.sin(ang))
        R = 0.27
        for h in (1, -1):
            th0 = ang if h > 0 else ang + math.pi
            arc = [(cr + R * math.sin(th0 + math.pi * i / 14), cc + R * math.cos(th0 + math.pi * i / 14))
                   for i in range(15)]
            ox, oy = h * 0.04 * nv[0], h * 0.04 * nv[1]
            pts = shift(arc, ox, oy)
            self.poly(s, ctx, pts, ART["split_rim"])
            self.poly(s, ctx, shift(scale_pts(pts, cr + ox, cc + oy, 0.88), -0.008, -0.008), ART["split_light"])
            self.poly(s, ctx, circ(cr + h * 0.11 * nv[0], cc + h * 0.11 * nv[1], 0.055, 12, 1.25), (28, 28, 48))