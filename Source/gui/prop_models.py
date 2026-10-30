"""prop_models.py - Hình học cổng torii, dựng bằng code.

Là nguồn để xuất ra models/*.obj (make_assets.py), đồng thời là phương án dự phòng khi thiếu file.
Tọa độ (row, col, z), gốc ở chân vật trên mặt sàn.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .resources import LoadedModel, Material, ModelBuilder
from .theme import PALETTE

PROP_MATERIALS = {
    "torii_red": Material("torii_red", PALETTE["torii"], texture_ref="torii_wood.png"),
    "torii_dark": Material("torii_dark", tuple(int(c * 0.62) for c in PALETTE["torii"]), texture_ref="torii_wood.png"),
    "torii_cap": Material("torii_cap", PALETTE["torii_cap"]),
}


def _prism(b: ModelBuilder, outline: List[Tuple[float, float]], r0: float, r1: float, material: str, part: str) -> None:
    """Lăng trụ: đa giác (col, z) kéo dài theo row từ r0 tới r1."""
    n = len(outline)
    front = b.add_vertices([(r1, c, z) for c, z in outline])
    back = b.add_vertices([(r0, c, z) for c, z in outline])
    b.add_face([front + i for i in reversed(range(n))], material, part)
    b.add_face([back + i for i in range(n)], material, part)
    for i in range(n):
        j = (i + 1) % n
        b.add_face([front + i, front + j, back + j, back + i], material, part)


def build_torii() -> LoadedModel:
    b = ModelBuilder("torii")
    half, top = 0.37, 1.62
    for name, dc in (("post_left", -half), ("post_right", half)):
        b.add_box((-0.05, dc - 0.05, 0.0), (0.05, dc + 0.05, top - 0.05), "torii_red", name)
        b.add_box((-0.065, dc - 0.065, 0.0), (0.065, dc + 0.065, 0.12), "torii_cap", name + "_foot")
    w = half + 0.1
    b.add_box((-0.035, -w, 1.20), (0.035, w, 1.30), "torii_dark", "nuki")
    b.add_box((-0.03, -0.06, 1.28), (0.03, 0.06, 1.50), "torii_cap", "plaque")
    wide = half + 0.24
    curve = [(wide * t, top + 0.09 * abs(t) ** 2.4) for t in np.linspace(-1, 1, 11)]
    # xà trên và mũ đen: chia thành từng đoạn lồi để sắp thứ tự vẽ đúng ở mọi góc nhìn
    for i, ((c0, z0), (c1, z1)) in enumerate(zip(curve, curve[1:])):
        _prism(b, [(c0, z0), (c1, z1), (c1, z1 - 0.11), (c0, z0 - 0.11)], -0.05, 0.05, "torii_red", f"kasagi{i}")
        _prism(b, [(c0, z0 + 0.06), (c1, z1 + 0.06), (c1, z1), (c0, z0)], -0.07, 0.07, "torii_cap", f"cap{i}")
    return LoadedModel(b.build(), dict(PROP_MATERIALS), source="fallback")
