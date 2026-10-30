"""raster.py - Dán texture lên một mặt phẳng. Camera trực giao nên mặt phẳng lên hình là hình bình hành,
ánh xạ afin là chính xác. Tương đối chậm nên chỉ dùng khi dựng ảnh sàn tĩnh (có cache), không dùng mỗi frame."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pygame

from .resources import Texture


def textured_quad(surface: pygame.Surface, p0: Sequence[float], pu: Sequence[float], pv: Sequence[float], tex: Texture,
                  uv0: Tuple[float, float], uv_span: Tuple[float, float], tint: Sequence[int],
                  corner_gain: Sequence[float] = (1.0, 1.0, 1.0, 1.0)) -> None:
    """Tô hình bình hành (p0, p0->pu, p0->pv) bằng texture, chuẩn hóa theo màu trung bình rồi nhân với tint:
    nơi không có texture thì mặt vẫn có màu tint, nên diện mạo thay đổi ít giữa hai trường hợp.
    corner_gain: hệ số sáng tại (p0, pu, p0+pu+pv-p0, pv), nội suy song tuyến nên liền mạch giữa các ô kề nhau.
    surface phải có kênh alpha (SRCALPHA)."""
    p0 = np.asarray(p0, dtype=float)
    a, b = np.asarray(pu, dtype=float) - p0, np.asarray(pv, dtype=float) - p0
    det = a[0] * b[1] - a[1] * b[0]
    if abs(det) < 1e-6:
        return
    w, h = surface.get_size()
    corners = np.array([p0, p0 + a, p0 + b, p0 + a + b])
    x0, y0 = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 1, (w, h))
    if x1 <= x0 or y1 <= y0:
        return
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5 - p0[0], np.arange(y0, y1) + 0.5 - p0[1], indexing="ij")
    s = (gx * b[1] - gy * b[0]) / det
    t = (gy * a[0] - gx * a[1]) / det
    inside = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    if not inside.any():
        return
    tw, th = tex.size
    tx = (((uv0[0] + s[inside] * uv_span[0]) % 1.0) * tw).astype(int) % tw
    ty = (((uv0[1] + t[inside] * uv_span[1]) % 1.0) * th).astype(int) % th
    rgb = tex.pixels[tx, ty] * (np.asarray(tint, dtype=float) / np.maximum(tex.mean, 1.0))
    si, ti = s[inside], t[inside]
    g00, g10, g11, g01 = corner_gain
    rgb *= (g00 * (1 - si) * (1 - ti) + g10 * si * (1 - ti) + g11 * si * ti + g01 * (1 - si) * ti)[:, None]
    colors = pygame.surfarray.pixels3d(surface)
    colors[x0:x1, y0:y1][inside] = rgb.clip(0, 255).astype(np.uint8)
    del colors
    alpha = pygame.surfarray.pixels_alpha(surface)
    alpha[x0:x1, y0:y1][inside] = 255
    del alpha
