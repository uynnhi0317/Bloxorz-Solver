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


def shade(c, f: float):
    return tuple(max(0, min(255, int(v * f))) for v in c)


ART = {
    # floor
    "cobble_body": (74, 84, 106), "moss_body": (56, 80, 62),
    "mortar": (28, 32, 44), "mortar_moss": (62, 92, 70),
    "cobble_stones": [(122, 132, 156), (108, 118, 142), (134, 142, 162)],
    "moss_stones": [(98, 118, 94), (86, 106, 84), (108, 126, 96)],
    "moss_dark": (60, 126, 58), "moss_light": (120, 188, 94),
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
    # fragile (coal + ember)
    "frag_body": (36, 30, 34), "frag_ember": (120, 38, 18),
    "frag_slab": [(46, 40, 44), (38, 33, 37), (52, 45, 48)],
}


# ---------------------------------------------------------------- geometry (row, col)
def rect(r: float, c: float, m: float) -> List[Pt]:
    return [(r + m, c + m), (r + m, c + 1 - m), (r + 1 - m, c + 1 - m), (r + 1 - m, c + m)]


def circ(cr: float, cc: float, rad: float, n: int = 14, sq: float = 1.0) -> List[Pt]:
    return [(cr + rad * sq * math.sin(2 * math.pi * i / n), cc + rad * math.cos(2 * math.pi * i / n))
            for i in range(n)]


def sq_shape(cr, cc, hr, hc, e=2.6, n=18, rng: Optional[random.Random] = None, jit=0.0) -> List[Pt]:
    """Hình chữ nhật bo tròn (siêu elip) bán kính nửa cạnh hr x hc."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        ca, sa = math.cos(a), math.sin(a)
        x = math.copysign(abs(ca) ** (2 / e), ca)
        y = math.copysign(abs(sa) ** (2 / e), sa)
        k = 1 + (rng.uniform(-jit, jit) if rng else 0.0)
        pts.append((cr + y * hr * k, cc + x * hc * k))
    return pts


def scale_pts(pts: Sequence[Pt], cr: float, cc: float, f: float) -> List[Pt]:
    return [(cr + (p[0] - cr) * f, cc + (p[1] - cc) * f) for p in pts]


def shift(pts: Sequence[Pt], dr: float, dc: float) -> List[Pt]:
    return [(p[0] + dr, p[1] + dc) for p in pts]


def layout(rng: random.Random, r: int, c: int, nrows: int, gap: float, cols=(2, 3)):
    """Chia ô thành các hàng đá có độ cao/độ rộng ngẫu nhiên: [(cr, cc, nửa_cao, nửa_rộng)]."""
    hs = [rng.uniform(0.8, 1.2) for _ in range(nrows)]
    tot = sum(hs)
    out, y = [], 0.0
    for h in hs:
        h /= tot
        ws = [rng.uniform(0.7, 1.3) for _ in range(rng.choice(cols))]
        wt, x = sum(ws), 0.0
        for w in ws:
            w /= wt
            out.append((r + y + h / 2, c + x + w / 2, max(0.03, h / 2 - gap), max(0.03, w / 2 - gap)))
            x += w
        y += h
    return out


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
            return ART["frag_body"]
        if k == TileType.BRIDGE:
            return ART["bridge_body"]
        if k == TileType.GOAL:
            return ART["goal_body"]
        if k == TileType.SWITCH:
            return ART["split_body"] if tile.switch_kind == SwitchKind.SPLIT else ART["plate_body"]
        return ART["moss_body"] if (r + c) % 2 else ART["cobble_body"]

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
        if k == TileType.FLOOR:
            self._cobbles(s, ctx, r, c, moss=(r + c) % 2 == 1)
        elif k == TileType.FRAGILE:
            self._fragile(s, ctx, r, c)
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

    # ------------------------------------------------------------ floor: cobblestone / moss
    def _moss(self, s, ctx, rng, r, c, cr, cc, size) -> None:
        for _ in range(rng.randint(2, 3)):
            rad = size * rng.uniform(0.35, 0.6)
            m = rad + 0.03
            mr = min(max(cr + rng.uniform(-0.5, 0.5) * size, r + m), r + 1 - m)
            mc = min(max(cc + rng.uniform(-0.5, 0.5) * size, c + m), c + 1 - m)
            self.poly(s, ctx, circ(mr, mc, rad, 9, 0.85), ART["moss_dark"])
            self.poly(s, ctx, circ(mr - rad * 0.2, mc - rad * 0.2, rad * 0.6, 8, 0.85), ART["moss_light"])

    def _cobbles(self, s, ctx, r, c, moss: bool) -> None:
        rng = random.Random(r * 7919 + c * 104729 + 11)
        self.poly(s, ctx, rect(r, c, 0.0), ART["mortar_moss"] if moss else ART["mortar"])
        pal = ART["moss_stones"] if moss else ART["cobble_stones"]
        gap = 0.018 if moss else 0.038
        for cr, cc, hr, hc in layout(rng, r, c, 3, gap):
            base = rng.choice(pal)
            pts = sq_shape(cr, cc, hr, hc, 2.6, 18, rng, 0.05)
            self.poly(s, ctx, pts, shade(base, 0.72 if moss else 0.55))   # rim                                   # rim tối
            self.poly(s, ctx, shift(scale_pts(pts, cr, cc, 0.93), -0.012, -0.008), base)  # thân
            self.poly(s, ctx, shift(scale_pts(pts, cr, cc, 0.55), -0.03, -0.02), shade(base, 1.15))  # sáng
            if moss:
                if rng.random() < 0.75:
                    self._moss(s, ctx, rng, r, c, cr - hr * 0.25, cc - hc * 0.15, min(hr, hc))
            elif rng.random() < 0.4:   # vết nứt chân chim trên đá
                a = (cr - hr * 0.45, cc + rng.uniform(-hc, hc) * 0.5)
                b = (cr + hr * 0.45, cc + rng.uniform(-hc, hc) * 0.5)
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + rng.uniform(-0.04, 0.04))
                self.line(s, ctx, [a, mid, b], shade(base, 0.42), 0.9)
        if moss:     # rêu lấp vào khe giữa các viên đá
            for _ in range(3):
                self._moss(s, ctx, rng, r, c, r + rng.uniform(0.1, 0.9), c + rng.uniform(0.1, 0.9), 0.05)
        else:        # vài đốm rêu nhỏ cho đỡ trơ
            for _ in range(rng.choice((0, 1, 2))):
                self._moss(s, ctx, rng, r, c, r + rng.uniform(0.1, 0.9), c + rng.uniform(0.1, 0.9), 0.04)

    # ------------------------------------------------------------ fragile: than hồng, nứt
    def _fragile(self, s, ctx, r, c) -> None:
        rng = random.Random(r * 6151 + c * 3571 + 5)
        self.poly(s, ctx, rect(r, c, 0.0), ART["frag_ember"])           # than hồng lộ ra ở các khe
        for cr, cc, hr, hc in layout(rng, r, c, 3, 0.02):
            pts = sq_shape(cr, cc, hr, hc, 2.0, 12, rng, 0.08)
            base = rng.choice(ART["frag_slab"])
            self.poly(s, ctx, pts, base)
            self.poly(s, ctx, shift(scale_pts(pts, cr, cc, 0.6), -0.012, -0.01), shade(base, 1.18))
        for _ in range(2):                                               # vết nứt lớn xuyên qua các tảng
            sa = rng.randrange(4)
            a = edge_pt(rng, r, c, sa)
            if rng.random() < 0.35:
                b = (r + 0.5 + rng.uniform(-0.25, 0.25), c + 0.5 + rng.uniform(-0.25, 0.25))
            else:
                b = edge_pt(rng, r, c, (sa + rng.choice((1, 2, 3))) % 4)
            path = jag(rng, a, b, 5, 0.06)
            self._hot_line(s, ctx, path, 1.0)
            for _ in range(1):                                           # nhánh phụ
                p = path[rng.randint(1, len(path) - 2)]
                ang = rng.uniform(0, 2 * math.pi)
                ln = rng.uniform(0.15, 0.3)
                q = (min(max(p[0] + ln * math.sin(ang), r + 0.03), r + 0.97),
                     min(max(p[1] + ln * math.cos(ang), c + 0.03), c + 0.97))
                self._hot_line(s, ctx, jag(rng, p, q, 3, 0.04), 0.6)
        for _ in range(8):                                               # tàn lửa li ti
            pr, pc = r + rng.uniform(0.08, 0.92), c + rng.uniform(0.08, 0.92)
            self.poly(s, ctx, circ(pr, pc, rng.uniform(0.01, 0.022), 6), (255, 168, 70))

    def _hot_line(self, s, ctx, path, k: float) -> None:
        self.line(s, ctx, path, (104, 24, 14), 4.2 * k)
        self.line(s, ctx, path, (238, 92, 30), 2.4 * k)
        self.line(s, ctx, path, (255, 198, 104), 1.0 * k)

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