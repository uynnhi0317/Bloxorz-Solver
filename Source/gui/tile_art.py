"""tile_art.py - Họa tiết trên mặt từng loại ô (chỉ vẽ, không đổi luật).

Hàm nhận `paint` có 2 thao tác trong tọa độ (row, col), đơn vị ô:
    paint.poly(points, color, z)          # đa giác
    paint.line(a, b, color, width, z)     # đoạn thẳng
Màu có thể kèm alpha. Họa tiết sinh giả ngẫu nhiên theo (row, col) nên mỗi ô khác nhau nhưng
luôn ổn định giữa các frame và giữa các lần vào màn.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

from game import SwitchKind, Tile
from .theme import PALETTE, sakura_points

Point = Tuple[float, float]


def _rng(r: int, c: int) -> random.Random:
    return random.Random(r * 73856093 ^ c * 19349663)


def _blob(rng: random.Random, cr: float, cc: float, rad: float, lo: Point, hi: Point, n: int = 8) -> List[Point]:
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        d = rad * rng.uniform(0.6, 1.15)
        pts.append((min(hi[0], max(lo[0], cr + d * math.sin(a))), min(hi[1], max(lo[1], cc + d * math.cos(a)))))
    return pts


def _circle(cr: float, cc: float, rad: float, n: int = 28) -> List[Point]:
    return [(cr + rad * math.sin(2 * math.pi * i / n), cc + rad * math.cos(2 * math.pi * i / n)) for i in range(n)]


def floor(paint, r: int, c: int, z: float = 0.002) -> None:
    """Đá lát sân: ron chia phiến + vài mảng rêu sát mép."""
    rng = _rng(r, c)
    seam = PALETTE["seam"]
    # ron chia phiến đá, không kẻ kín hết ô để khỏi rối mắt
    if rng.random() < 0.5:
        m = rng.uniform(0.35, 0.65)
        paint.line((r + m, c + 0.08), (r + m, c + 0.92), seam, 1, z)
        k = rng.uniform(0.3, 0.7)
        paint.line((r + 0.08, c + k), (r + m, c + k), seam, 1, z)
    else:
        m = rng.uniform(0.35, 0.65)
        paint.line((r + 0.08, c + m), (r + 0.92, c + m), seam, 1, z)
        k = rng.uniform(0.3, 0.7)
        paint.line((r + k, c + m), (r + k, c + 0.92), seam, 1, z)
    lo, hi = (r + 0.05, c + 0.05), (r + 0.95, c + 0.95)
    moss = PALETTE["moss"]
    for i in range(rng.randint(2, 3)):
        corner = rng.choice(((0.14, 0.14), (0.14, 0.86), (0.86, 0.14), (0.86, 0.86), (0.5, 0.1), (0.1, 0.5)))
        rad = rng.uniform(0.07, 0.14)
        shade = rng.uniform(0.82, 1.08)
        col = tuple(min(255, int(v * shade)) for v in moss)
        paint.poly(_blob(rng, r + corner[0], c + corner[1], rad, lo, hi), col, z + 0.001)


def _scaled(col, k: float) -> Tuple[int, int, int]:
    return tuple(min(255, max(0, int(v * k))) for v in col)


def _leaf(paint, rng: random.Random, r: int, c: int, cr: float, cc: float, ln: float, col, z: float, detail: int) -> None:
    """Một chiếc lá khô: gân giữa hơi cong chia lá thành nửa sáng nửa tối, đầu lá nhọn, gốc lá rộng hơn.
    detail: 0 = chỉ hình lá, 1 = thêm bóng đổ và gân giữa, 2 = thêm cuống và gân phụ."""
    ang = rng.uniform(0, 2 * math.pi)
    ca, sa = math.cos(ang), math.sin(ang)
    wd = ln * rng.uniform(0.42, 0.56)
    bend = ln * rng.uniform(-0.28, 0.28)
    lit = 1.1 if math.sin(ang) > 0 else 0.84   # nửa lá hướng về phía đèn sáng hơn, như lá hơi gập theo gân

    def world(along: float, across: float) -> Point:
        pr, pc = along * ln, across + bend * (along * along - 0.33)
        return (min(r + 0.985, max(r + 0.015, cr + pr * ca - pc * sa)),
                min(c + 0.985, max(c + 0.015, cc + pr * sa + pc * ca)))

    ts = [-1 + 2 * i / 8 for i in range(9)]
    half_w = [wd * (1 - t * t) ** 0.8 * (1 - 0.3 * t) for t in ts]
    rib = [world(t, 0.0) for t in ts]
    left = [world(t, h) for t, h in zip(ts, half_w)]
    right = [world(t, -h) for t, h in zip(ts, half_w)]
    if detail:
        shadow = [(a + 0.012, b + 0.008) for a, b in left + right[::-1]]
        paint.poly(shadow, _scaled(PALETTE["fragile"], 0.48), z)
    paint.poly(left + rib[::-1], _scaled(col, lit), z + 0.0001)
    paint.poly(right + rib[::-1], _scaled(col, 2.0 - lit - 0.06), z + 0.0001)
    if not detail:
        return
    vein = _scaled(col, 0.62)
    for p, q in zip(rib[::2], rib[2::2]):
        paint.line(p, q, vein, 1, z + 0.0002)
    if detail < 2:
        return
    paint.line(world(-1.0, 0.0), world(-1.28, 0.0), _scaled(col, 0.5), 1, z + 0.0002)   # cuống lá
    for t in (-0.45, 0.0, 0.45):
        h = wd * (1 - t * t) ** 0.8 * (1 - 0.3 * t) * 0.78
        for side in (-1, 1):
            paint.line(world(t, 0.0), world(t + 0.3, side * h), _scaled(col, 0.76), 1, z + 0.0002)


def fragile(paint, r: int, c: int, z: float = 0.003) -> None:
    """Đống lá khô: lớp dưới tối và lớp trên có gân, bóng đổ; rất khác màu đá nên nhận ra ngay cả khi ô nhỏ."""
    rng = _rng(r, c)
    leaves = PALETTE["leaves"]
    earth = _scaled(PALETTE["fragile"], 0.6)
    paint.poly([(r + 0.02, c + 0.02), (r + 0.02, c + 0.98), (r + 0.98, c + 0.98), (r + 0.98, c + 0.02)], earth, z)
    i = 0
    for layer, (cells, dim, lo, hi) in enumerate(((5, 0.66, 0.13, 0.19), (4, 0.85, 0.13, 0.18), (3, 1.0, 0.14, 0.2))):
        for a in range(cells):
            for b in range(cells):
                if layer == 2 and rng.random() < 0.3:
                    continue   # chừa khe để thấy lớp lá bên dưới
                cr = r + (a + rng.uniform(0.1, 0.9)) / cells
                cc = c + (b + rng.uniform(0.1, 0.9)) / cells
                jitter = rng.uniform(0.9, 1.1)
                base = leaves[rng.randrange(len(leaves))]
                col = (base[0] * jitter, base[1] * jitter * rng.uniform(0.93, 1.05), base[2] * jitter)
                _leaf(paint, rng, r, c, cr, cc, rng.uniform(lo, hi), _scaled(col, dim), z + 0.001 + i * 0.0004, layer)
                i += 1


def goal(paint, r: int, c: int, z: float = 0.002) -> None:
    """Hố dịch chuyển trên mặt ô đích: vành gỗ + lòng hố xanh biển sâu (xoáy sáng vẽ động trong renderer)."""
    cr, cc = r + 0.5, c + 0.5
    paint.poly(_circle(cr, cc, 0.40), PALETTE["goal_rim"], z)
    paint.poly(_circle(cr, cc, 0.36), PALETTE["goal_hole"], z + 0.001)
    paint.poly(_circle(cr, cc, 0.20), PALETTE["goal_inner"], z + 0.002)


def switch(paint, tile: Tile, r: int, c: int, active: bool, z: float = 0.002) -> None:
    cr, cc = r + 0.5, c + 0.5
    base = PALETTE["switch_base"]
    if tile.switch_kind == SwitchKind.SOFT:
        # phiến đá tròn, khắc hoa anh đào năm cánh
        col = PALETTE["soft"] if active else PALETTE["soft_off"]
        paint.poly(_circle(cr, cc, 0.43), (124, 110, 94), z)
        paint.poly(_circle(cr, cc, 0.39), (172, 158, 136), z + 0.001)
        for i in range(5):
            paint.poly(sakura_points(cr, cc, 0.35, i), col, z + 0.002)
        paint.poly(_circle(cr, cc, 0.07, 12), PALETTE["torii"] if active else (136, 106, 98), z + 0.003)
    elif tile.switch_kind == SwitchKind.HEAVY:
        # hai thanh tre bắt chéo thành dấu X, có đốt tre
        col = PALETTE["heavy"] if active else PALETTE["heavy_off"]
        node = tuple(int(v * 0.6) for v in col)
        d = 0.33
        paint.poly([(cr - 0.42, cc - 0.42), (cr - 0.42, cc + 0.42), (cr + 0.42, cc + 0.42), (cr + 0.42, cc - 0.42)],
                   base, z)
        for a, b in (((cr - d, cc - d), (cr + d, cc + d)), ((cr - d, cc + d), (cr + d, cc - d))):
            paint.line(a, b, (72, 80, 56), 7, z + 0.001)
            paint.line(a, b, col, 5, z + 0.002)
            for t in (0.2, 0.8):
                pr, pc = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                dr, dc = (b[1] - a[1]) * 0.09, -(b[0] - a[0]) * 0.09
                paint.line((pr - dr, pc - dc), (pr + dr, pc + dc), node, 2, z + 0.003)
    else:
        # hai vầng bát quái tách rời, ghép thành dấu ( )
        col = PALETTE["split"] if active else PALETTE["split_off"]
        paint.poly(_circle(cr, cc, 0.44), (46, 92, 96), z)
        paint.poly(_circle(cr, cc, 0.40), (66, 122, 122), z + 0.001)
        for shift, a0, a1 in ((0.20, 140, 220), (-0.20, -40, 40)):
            for radius, width in ((0.46, 6), (0.33, 2)):  # vạch dày + vạch mảnh như quẻ bát quái
                arc = []
                for i in range(9):
                    a = math.radians(a0 + (a1 - a0) * i / 8)
                    arc.append((cr + radius * math.sin(a), cc + shift + radius * math.cos(a)))
                for p, q in zip(arc, arc[1:]):
                    paint.line(p, q, col, width, z + 0.002)


def bridge_ghost(paint, r: int, c: int, axis: int, z: float) -> None:
    """Cầu ĐÓNG: ván chìm dưới mặt nước. Tối, không lan can, không mặt đứng,
    nên khác hẳn cả ô đi được lẫn ô Void (Void không có gì)."""
    col = PALETTE["bridge_ghost"]
    n = 5
    for i in range(n):
        a, b = 0.06 + i * 0.885 / n, 0.06 + (i + 0.78) * 0.885 / n
        corners = ((a, 0.08), (a, 0.92), (b, 0.92), (b, 0.08))
        pts = [(r + q, c + p) if axis == 1 else (r + p, c + q) for p, q in corners]
        paint.poly(pts, (*col, 150), z)
