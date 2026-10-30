"""noface.py - Hình Vô Diện dạng khối hộp vuông cạnh, mặt nạ hình chữ nhật.

Hình là một vật rắn duy nhất, dựng trong tọa độ cục bộ: gốc ở tâm khối, trục z chạy dọc thân về phía đầu,
mặt nạ dán trên mặt +row. Renderer đặt nó vào thế giới bằng một ma trận xoay (Box.rot), nên đầu luôn gắn
chặt với thân trong mọi animation. Module này CHỈ vẽ, không đọc luật chơi.

Ba dạng hình:
    whole: khối 1x1x2 nguyên vẹn (hộp đen, mặt nạ trắng)
    head:  cube phần đầu (nhỏ hơn ô, mặt nạ to)
    body:  cube phần thân (nhỏ hơn ô, cùng màu đầu nhưng không mặt nạ)

build_model() là nguồn để xuất ra models/noface_*.obj, đồng thời là phương án dự phòng khi thiếu file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pygame

from .mesh_render import DECAL_PART, Mesh, get_mesh
from .resources import LoadedModel, Material, ModelBuilder
from .theme import PALETTE

Color = Tuple[int, int, int]

_STRIPS = 6            # số dải dọc trên mỗi mặt bên, mỗi dải một màu lấy từ texture nếp vải
_EDGE = 0.02           # bề rộng viền sáng ở cạnh hộp: giúp thấy rõ khối vuông trên nền tối
_LIFT = 0.003          # khoảng nổi của chi tiết dán so với mặt hộp

FIGURE_MATERIALS = {
    "cloak": Material("cloak", (255, 255, 255), texture_ref="cloak.png"),   # nhận màu do renderer truyền vào
    "edge": Material("edge", (150, 128, 118)),
    "mask": Material("mask", PALETTE["mask"], illum=0),
    "mark": Material("mark", PALETTE["mask_mark"], illum=0),
    "eye": Material("eye", PALETTE["mask_eye"], illum=0),
}


@dataclass(frozen=True)
class Shape:
    half_w: float                                # nửa bề rộng (hai chiều ngang bằng nhau)
    length: float                                # chiều cao hộp
    base_z: float                                # z của đáy trong tọa độ cục bộ
    face_width: float                            # đổi góc phi của chi tiết mặt nạ ra bề ngang trên mặt phẳng
    drop: float = 0.0                            # hạ tâm hình so với tâm ô chiếm chỗ, để hình nhỏ vẫn chạm sàn


SHAPES: Dict[str, Shape] = {
    "whole": Shape(0.48, 1.96, -0.98, 0.42),
    # cube tách nhỏ hơn ô; tâm hình ở gốc tọa độ để lăn quanh tâm mà không lệch khỏi ô
    "head": Shape(0.34, 0.68, -0.34, 0.30, drop=0.16),
    "body": Shape(0.34, 0.68, -0.34, 0.30, drop=0.16),
}

X, Y, Z = np.eye(3)
# (pháp tuyến, trục ngang trong mặt, trục dọc trong mặt): ngang x dọc = pháp tuyến, nên mặt quay ra ngoài
_SIDES = ((X, Y, Z), (-X, -Y, Z), (Y, -X, Z), (-Y, X, Z))
_CAPS = ((Z, X, Y), (-Z, Y, X))


def _oval(u0: float, du: float, ph0: float, dph: float, n: int = 16):
    return [(u0 + du * math.sin(t), ph0 + dph * math.cos(t)) for t in (2 * math.pi * i / n for i in range(n))]


def _face_features(kind: str) -> List[Tuple[list, str]]:
    """Các mảng trên mặt nạ trong tọa độ (u, phi): phi = 0 là chính giữa mặt. Thứ tự = thứ tự dán."""
    if kind == "whole":
        u0, du, dph = 0.755, 0.165, 0.86
    else:
        u0, du, dph = 0.52, 0.34, 0.92
    k = du / 0.165   # cỡ chi tiết tỉ lệ theo mặt nạ
    mask = [(u0 - du, dph), (u0 + du, dph), (u0 + du, -dph), (u0 - du, -dph)]   # hình chữ nhật
    feats: List[Tuple[list, str]] = [(mask, "mask")]
    for s in (-1, 1):
        ex = s * dph * 0.42
        feats.append(([(u0 + 0.105 * k, ex), (u0 + 0.03 * k, ex - 0.10 * dph), (u0 + 0.03 * k, ex + 0.10 * dph)], "mark"))
        feats.append(([(u0 - 0.095 * k, ex), (u0 - 0.018 * k, ex + 0.11 * dph), (u0 - 0.018 * k, ex - 0.11 * dph)], "mark"))
        feats.append((_oval(u0 + 0.008 * k, 0.018 * k, ex, 0.13 * dph, 10), "eye"))
    feats.append((_oval(u0 - 0.10 * k, 0.011 * k, 0.0, 0.2 * dph, 10), "eye"))
    return feats


def _add_rect(b: ModelBuilder, centre, u_axis, v_axis, a0: float, a1: float, c0: float, c1: float, material: str,
              part: str, uvs=None) -> None:
    corners = [centre + u_axis * p + v_axis * q for p, q in ((a0, c0), (a1, c0), (a1, c1), (a0, c1))]
    start = b.add_vertices(corners)
    b.add_face(range(start, start + 4), material, part, uvs)


def build_model(kind: str) -> LoadedModel:
    sh = SHAPES[kind]
    b = ModelBuilder(f"noface_{kind}")
    hw, hl = sh.half_w, sh.length / 2
    box_centre = np.array((0.0, 0.0, sh.base_z + hl))

    # mặt bên: chia dải dọc để thấy nếp vải
    for side, (normal, u_axis, v_axis) in enumerate(_SIDES):
        centre = box_centre + normal * hw
        for i in range(_STRIPS):
            t0, t1 = i / _STRIPS, (i + 1) / _STRIPS
            uvs = [((side + t0) / 4, 0), ((side + t1) / 4, 0), ((side + t1) / 4, 1), ((side + t0) / 4, 1)]
            _add_rect(b, centre, u_axis, v_axis, -hw + 2 * hw * t0, -hw + 2 * hw * t1, -hl, hl, "cloak", "body", uvs)
    for normal, u_axis, v_axis in _CAPS:
        _add_rect(b, box_centre + normal * hl, u_axis, v_axis, -hw, hw, -hw, hw, "cloak", "body",
                  [(0, 0), (1, 0), (1, 1), (0, 1)])
    # viền sáng dọc các cạnh hộp
    e = _EDGE
    for normal, u_axis, v_axis in _SIDES:
        centre = box_centre + normal * (hw + _LIFT)
        for a0, a1, c0, c1 in ((-hw, hw, -hl, -hl + e), (-hw, hw, hl - e, hl), (-hw, -hw + e, -hl, hl), (hw - e, hw, -hl, hl)):
            _add_rect(b, centre, u_axis, v_axis, a0, a1, c0, c1, "edge", DECAL_PART)
    for normal, u_axis, v_axis in _CAPS:
        centre = box_centre + normal * (hl + _LIFT)
        for a0, a1, c0, c1 in ((-hw, hw, -hw, -hw + e), (-hw, hw, hw - e, hw), (-hw, -hw + e, -hw, hw), (hw - e, hw, -hw, hw)):
            _add_rect(b, centre, u_axis, v_axis, a0, a1, c0, c1, "edge", DECAL_PART)

    if kind != "body":
        # mặt nạ phẳng trên mặt +row
        x = hw + 2 * _LIFT
        for pts, role in _face_features(kind):
            b.add_polygon([(x, ph * sh.face_width, sh.base_z + sh.length * u) for u, ph in pts], role, DECAL_PART)
    return LoadedModel(b.build(), dict(FIGURE_MATERIALS), source="fallback")


def figure_mesh(kind: str) -> Mesh:
    return get_mesh(f"noface_{kind}", lambda: build_model(kind), tint_material="cloak")


_MASCOTS: Dict[int, pygame.Surface] = {}


def mascot_sprite(height: int) -> pygame.Surface:
    """Vô Diện dạng khối nhìn thẳng (2D, nền trong suốt) dùng trang trí menu và màn thắng."""
    if height not in _MASCOTS:
        ss = 3
        h = height * ss
        w = int(h * 0.56)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, PALETTE["block"], (0, 0, w, h))
        pygame.draw.rect(s, (150, 128, 118), (0, 0, w, h), max(2, ss * 2))
        for dx in (0.2, 0.4, 0.6, 0.8):  # nếp áo
            pygame.draw.line(s, (84, 74, 76), (w * dx, h * 0.56), (w * dx, h - ss * 3), max(1, ss))
        rect = pygame.Rect(0, 0, int(w * 0.72), int(h * 0.36))
        rect.center = (w // 2, int(h * 0.27))
        pygame.draw.rect(s, PALETTE["mask"], rect)
        pygame.draw.rect(s, (214, 202, 184), rect, max(1, ss))
        cx, ey = w / 2, rect.centery - rect.h * 0.04
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
