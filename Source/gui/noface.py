"""noface.py - Hình Vô Diện dựng từ lưới 3D nằm gọn trong khối của game.

Mọi điểm được tính theo hệ trục cục bộ của khối (rút từ 8 đỉnh `verts`), nên hình tự xoay cứng
theo khối trong mọi animation (lăn, rơi, chìm...). Module này CHỈ vẽ, không đọc luật chơi.

Ba dạng hình:
    whole: khối 1x1x2 nguyên vẹn (áo choàng đen, mặt nạ trắng)
    head:  cube 1x1x1 phần đầu (nhỏ hơn ô, mặt nạ to)
    body:  cube 1x1x1 phần thân (nhỏ hơn ô, đai tím + đồng vàng)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from .theme import PALETTE

Color = Tuple[int, int, int]

_STANDING_F = (0.96, -0.28, 0.0)
# pose -> (trục dọc thân hướng về phía đầu, hướng mặt) theo (row, col, z).
POSES = {
    "standing": ((0.0, 0.0, 1.0), _STANDING_F),
    "horizontal": ((0.0, -1.0, 0.0), (0.72, 0.0, 0.70)),  # nằm ngang: đầu bên trái, mặt hướng lên-trước
    "vertical": ((-1.0, 0.0, 0.0), (0.0, -0.18, 0.98)),   # nằm dọc: đầu quay vào trong, mặt ngửa lên
}
RIM = np.array((118, 104, 150), dtype=float)   # viền sáng để áo đen nổi trên nền tối
_SEGMENTS = 18
_SQUARE = 3.0                                  # 2 = tròn, càng lớn càng vuông


@dataclass(frozen=True)
class Shape:
    length: float                                # chiều dài thân (theo trục dọc)
    half_w: float                                # nửa bề rộng
    rings: Tuple[Tuple[float, float], ...]       # (u dọc thân 0..1, hệ số bán kính)
    bands: Tuple[Tuple[float, float, Color], ...] = ()  # (u0, u1, màu) dải màu trên thân
    feet: float = 0.0                            # khoảng lệch chân so với đáy khối


def _dome(u0: float, r0: float, n: int = 6) -> List[Tuple[float, float]]:
    """Vòm đầu: từ u0 (bán kính r0) khép lại tại u = 1."""
    return [(u0 + (1 - u0) * math.sin(i / n * math.pi / 2), r0 * math.cos(i / n * math.pi / 2)) for i in range(1, n + 1)]


SHAPES: Dict[str, Shape] = {
    "whole": Shape(1.96, 0.45, ((0.0, 0.0), (0.0, 0.98), (0.03, 1.0), (0.14, 0.94), (0.4, 0.86), (0.66, 0.84),
                                *_dome(0.70, 0.84))),
    "head": Shape(0.62, 0.34, ((0.0, 0.0), (0.0, 0.62), (0.08, 0.84), (0.3, 1.0), *_dome(0.46, 1.0))),
    "body": Shape(0.60, 0.33, ((0.0, 0.0), (0.0, 1.0), (0.1, 1.0), (0.5, 0.86), (0.82, 0.68), (0.94, 0.46), (0.98, 0.2),
                               (0.98, 0.0)),
                  bands=((0.44, 0.60, PALETTE["mask_mark"]),)),
}


@dataclass
class _Mesh:
    verts: np.ndarray      # (N, 3) tọa độ metric trong khối
    quads: np.ndarray      # (M, 4) chỉ số đỉnh
    normals: np.ndarray    # (M, 3)
    base: np.ndarray       # (M, 3) màu gốc của từng mảnh; 0 = theo màu Box
    tinted: np.ndarray     # (M,) True: dùng màu Box
    surf: Callable[[float, float], np.ndarray]
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray]


def _section_radius(phi: float) -> float:
    return 1.0 / (abs(math.cos(phi)) ** _SQUARE + abs(math.sin(phi)) ** _SQUARE) ** (1 / _SQUARE)


def _build(kind: str, pose: str, dims: Sequence[float]) -> _Mesh:
    sh = SHAPES[kind]
    up = np.array(POSES[pose][0], dtype=float)
    f = np.array(POSES[pose][1], dtype=float)
    f = f - up * float(f @ up)
    f /= np.linalg.norm(f)
    side = np.cross(up, f)
    centre = np.array(dims, dtype=float) / 2
    # chân chạm đáy theo trục dọc (đứng thẳng hay nằm đều nằm giữa ô theo hai chiều còn lại)
    centre = centre - up * (float(np.abs(np.array(dims) @ up)) / 2 - sh.length / 2)

    def ring_radius(u: float) -> float:
        rings = sh.rings
        for (u0, r0), (u1, r1) in zip(rings, rings[1:]):
            if u <= u1 and u1 > u0:
                return r0 + (r1 - r0) * (u - u0) / (u1 - u0)
        return rings[-1][1]

    def surf(u: float, phi: float) -> np.ndarray:
        rad = ring_radius(u) * sh.half_w * _section_radius(phi)
        return centre + up * (sh.length * (u - 0.5)) + (f * math.cos(phi) + side * math.sin(phi)) * rad

    phis = [2 * math.pi * i / _SEGMENTS for i in range(_SEGMENTS)]
    vs, quads, base, tinted = [], [], [], []
    for u, s in sh.rings:
        for ph in phis:
            rad = s * sh.half_w * _section_radius(ph)
            vs.append(centre + up * (sh.length * (u - 0.5)) + (f * math.cos(ph) + side * math.sin(ph)) * rad)
    verts = np.array(vs)
    nrings = len(sh.rings)
    for i in range(nrings - 1):
        u_mid = (sh.rings[i][0] + sh.rings[i + 1][0]) / 2
        band = next((c for u0, u1, c in sh.bands if u0 <= u_mid <= u1), None)
        for j in range(_SEGMENTS):
            a = i * _SEGMENTS + j
            b = i * _SEGMENTS + (j + 1) % _SEGMENTS
            quads.append((a, b, b + _SEGMENTS, a + _SEGMENTS))
            base.append(band if band is not None else (0, 0, 0))
            tinted.append(band is None)
    quads_arr = np.array(quads)
    # pháp tuyến hướng ra ngoài trục thân
    p0, p1, p2, p3 = (verts[quads_arr[:, k]] for k in range(4))
    n = np.cross(p2 - p0, p3 - p1)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(length, 1e-9)
    mid = (p0 + p1 + p2 + p3) / 4
    outward = mid - centre
    flip = (n * outward).sum(axis=1) < 0
    n[flip] *= -1
    keep = length[:, 0] > 1e-6  # bỏ mảnh suy biến (đỉnh vòm, tâm đáy)
    return _Mesh(verts, quads_arr[keep], n[keep], np.array(base, dtype=float)[keep], np.array(tinted)[keep], surf,
                 (up, f, side))


_MESHES: Dict[tuple, _Mesh] = {}
_FEATURES: Dict[tuple, list] = {}


def _mesh(kind: str, pose: str, dims: Tuple[int, int, int]) -> _Mesh:
    key = (kind, pose, dims)
    if key not in _MESHES:
        _MESHES[key] = _build(kind, pose, dims)
    return _MESHES[key]


def _oval(u0: float, du: float, ph0: float, dph: float, n: int = 16):
    return [(u0 + du * math.sin(t), ph0 + dph * math.cos(t)) for t in (2 * math.pi * i / n for i in range(n))]


def _face_features(kind: str) -> List[Tuple[list, str]]:
    """Các mảng trên mặt nạ trong tọa độ (u, phi): phi = 0 là chính giữa mặt."""
    if kind == "body":
        return [(_oval(0.34, 0.10, 0.0, 0.34), "gold"), (_oval(0.34, 0.05, 0.0, 0.16), "coin_hole")]
    if kind == "whole":
        u0, du, dph = 0.755, 0.165, 0.86
    else:
        u0, du, dph = 0.52, 0.34, 0.92
    k = du / 0.165   # cỡ chi tiết tỉ lệ theo mặt nạ
    feats: List[Tuple[list, str]] = [(_oval(u0, du, 0.0, dph, 24), "mask")]
    for s in (-1, 1):
        ex = s * dph * 0.42
        feats.append(([(u0 + 0.105 * k, ex), (u0 + 0.03 * k, ex + 0.10 * dph), (u0 + 0.03 * k, ex - 0.10 * dph)], "mark"))
        feats.append(([(u0 - 0.095 * k, ex), (u0 - 0.018 * k, ex + 0.11 * dph), (u0 - 0.018 * k, ex - 0.11 * dph)], "mark"))
        feats.append((_oval(u0 + 0.008 * k, 0.018 * k, ex, 0.13 * dph, 10), "eye"))
    feats.append((_oval(u0 - 0.10 * k, 0.011 * k, 0.0, 0.2 * dph, 10), "eye"))
    return feats


def _mix(a, b, t: float) -> Color:
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def figure_ops(verts: np.ndarray, kind: str, pose: str, color: Color, view_dir: np.ndarray, light: np.ndarray,
               project: Callable[[np.ndarray], np.ndarray], clip: Callable, below: Optional[bool] = None,
               fade: float = 1.0, active: bool = True) -> list:
    """Sinh danh sách lệnh vẽ (định dạng renderer._replay: (0, màu, đa giác)).
    fade: 0..1, độ hiện của mặt nạ (0 = hòa vào thân, dùng khi đổi tư thế giữa cú lăn)."""
    o = verts[0]
    axes = np.stack([verts[4] - o, verts[2] - o, verts[1] - o])  # 3 cạnh theo (row, col, z) của khối
    sizes = np.linalg.norm(axes, axis=1)
    unit = axes / sizes[:, None]
    dims = tuple(int(round(float(x))) for x in sizes)
    m = _mesh(kind, pose, dims)

    def to_world(p: np.ndarray) -> np.ndarray:
        return o + p @ unit

    world = to_world(m.verts)
    normals = m.normals @ unit
    facing = normals @ view_dir
    vis = np.nonzero(facing > 1e-3)[0]

    ops: list = []
    body = np.array(color, dtype=float)
    lit = 0.62 + 0.75 * np.maximum(0.0, normals @ light)
    rim = (1 - np.clip(facing, 0.0, 1.0)) ** 3
    shades = np.where(m.tinted[:, None], body[None, :], m.base) * lit[:, None] + RIM[None, :] * rim[:, None] * 0.75
    shades = shades.clip(0, 255).astype(int)

    quads = world[m.quads]
    order = vis[np.argsort((quads[vis].mean(axis=1) @ view_dir))]
    z_lo = quads[:, :, 2].min(axis=1)
    z_hi = quads[:, :, 2].max(axis=1)
    for i in order:
        col = tuple(int(v) for v in shades[i])
        pts = quads[i]
        if below is not None:
            if (below and z_lo[i] >= 0) or (not below and z_hi[i] <= 0):
                continue
            if z_lo[i] < 0 < z_hi[i]:
                poly, _ = clip(pts, below)
                if len(poly) < 3:
                    continue
                pts = np.array(poly)
        ops.append((0, col, project(pts).tolist()))

    if below is True:
        return ops  # phần chìm dưới sàn bị tile che, khỏi vẽ mặt nạ

    # mặt nạ: dán lên bề mặt thân, chỉ hiện khi mặt hướng về camera
    if fade <= 0.02:
        return ops
    up, f, _ = m.frame
    if float((f @ unit) @ view_dir) < -0.05:
        return ops
    feats = _FEATURES.get((kind, pose, dims))
    if feats is None:
        feats = _FEATURES[(kind, pose, dims)] = [(np.array([m.surf(u, ph) for u, ph in pts]), role)
                                                 for pts, role in _face_features(kind)]
    roles = {"mask": PALETTE["mask"], "mark": PALETTE["mask_mark"], "eye": PALETTE["mask_eye"],
             "gold": PALETTE["gold"], "coin_hole": PALETTE["block"]}
    dim = 1.0 if active else 0.72
    bodyc = tuple(int(c) for c in body)
    for pts, role in feats:
        col = _mix(bodyc, tuple(int(c * dim) for c in roles[role]), min(1.0, fade))
        p3 = to_world(pts)
        if below is False and p3[:, 2].min() < 0:
            if p3[:, 2].max() <= 0:
                continue
            poly, _ = clip(p3, False)
            if len(poly) < 3:
                continue
            p3 = np.array(poly)
        ops.append((0, col, project(p3).tolist()))
    return ops


_MASCOTS: Dict[int, pygame.Surface] = {}


def mascot_sprite(height: int) -> pygame.Surface:
    """Vô Diện nhìn thẳng (2D, nền trong suốt) dùng trang trí menu và màn thắng."""
    if height not in _MASCOTS:
        ss = 3
        h = height * ss
        w = int(h * 0.62)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        cx = w / 2
        cloak = [(cx - w * 0.47, h), (cx - w * 0.40, h * 0.62), (cx - w * 0.34, h * 0.30)]
        cloak += [(cx + w * 0.36 * math.cos(a), h * 0.26 - h * 0.25 * math.sin(a))
                  for a in np.linspace(math.pi, 0, 18)]
        cloak += [(cx + w * 0.34, h * 0.30), (cx + w * 0.40, h * 0.62), (cx + w * 0.47, h)]
        pygame.draw.polygon(s, (28, 24, 34), cloak)
        for dx in (-0.2, 0.0, 0.2):  # nếp áo
            pygame.draw.line(s, (52, 46, 62), (cx + w * dx * 0.7, h * 0.52), (cx + w * dx * 1.3, h),
                             max(1, ss))
        rect = pygame.Rect(0, 0, int(w * 0.48), int(h * 0.34))
        rect.center = (int(cx), int(h * 0.27))
        pygame.draw.ellipse(s, PALETTE["mask"], rect)
        pygame.draw.ellipse(s, (206, 196, 190), rect, max(1, ss))
        ey = rect.centery - rect.h * 0.04
        for side in (-1, 1):
            ex = cx + side * rect.w * 0.2
            mw, mh = rect.w * 0.055, rect.h * 0.20
            pygame.draw.polygon(s, PALETTE["mask_mark"], [(ex, ey - mh * 1.25), (ex - mw, ey - mh * 0.3), (ex + mw, ey - mh * 0.3)])
            pygame.draw.polygon(s, PALETTE["mask_mark"], [(ex, ey + mh * 1.25), (ex - mw, ey + mh * 0.3), (ex + mw, ey + mh * 0.3)])
            eye = pygame.Rect(0, 0, int(rect.w * 0.13), max(2, int(rect.h * 0.055)))
            eye.center = (int(ex), int(ey))
            pygame.draw.ellipse(s, PALETTE["mask_eye"], eye)
        mouth = pygame.Rect(0, 0, int(rect.w * 0.2), max(2, int(rect.h * 0.045)))
        mouth.center = (int(cx), int(rect.centery + rect.h * 0.27))
        pygame.draw.ellipse(s, PALETTE["mask_eye"], mouth)
        _MASCOTS[height] = pygame.transform.smoothscale(s, (w // ss, height))
    return _MASCOTS[height]
