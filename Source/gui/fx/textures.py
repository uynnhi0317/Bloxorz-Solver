"""fx/textures.py - Texture procedural (numpy) cho theme Hollow.

Tất cả texture được sinh bằng code, bake một lần (khi dựng sprite / static layer), không tốn chi phí mỗi frame.

API chính:
    tile_texture(kind, base_color, seed)        -> ndarray (S, S, 3) uint8
    map_top_face(tex, p00, p01, p10)            -> (pygame.Surface RGBA, (x0, y0))
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pygame

Color = Tuple[int, int, int]
TEX_SIZE = 48


# ------------------------------------------------------------------ noise
def value_noise(w: int, h: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Value noise 2D trong [0, 1], nội suy smoothstep giữa lưới ngẫu nhiên (cells x cells)."""
    grid = rng.random((cells + 2, cells + 2))
    ys = np.linspace(0, cells, h, endpoint=False)
    xs = np.linspace(0, cells, w, endpoint=False)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = ys - y0, xs - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    fy, fx = fy[:, None], fx[None, :]
    a = grid[y0][:, x0]
    b = grid[y0][:, x0 + 1]
    c = grid[y0 + 1][:, x0]
    d = grid[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm_noise(w: int, h: int, octaves: int = 4, seed: int = 0, base_cells: int = 3) -> np.ndarray:
    """Fractal noise: tổng các octave value noise, chuẩn hóa về [0, 1]."""
    rng = np.random.default_rng(seed)
    out, amp, total = np.zeros((h, w)), 1.0, 0.0
    for i in range(octaves):
        out += amp * value_noise(w, h, base_cells * 2 ** i, rng)
        total += amp
        amp *= 0.5
    out /= total
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-9)


def voronoi_cracks(w: int, h: int, seed: int = 0, n_points: int = 7, width: float = 1.6) -> np.ndarray:
    """Mạng vết nứt: (khoảng cách tới điểm gần thứ 2) - (gần nhất) nhỏ => nằm trên biên Voronoi.
    Trả về cường độ vết nứt trong [0, 1]."""
    rng = np.random.default_rng(seed + 991)
    pts = rng.random((n_points, 2)) * np.array([h, w])
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((yy[None] - pts[:, 0, None, None]) ** 2 + (xx[None] - pts[:, 1, None, None]) ** 2)
    part = np.partition(d, 1, axis=0)
    edge = part[1] - part[0]
    return np.clip(1.0 - edge / width, 0.0, 1.0)


def edge_factor(size: int, margin: float = 0.12) -> np.ndarray:
    """0 ở mép -> 1 ở trong (dùng làm ambient occlusion / chipped edge)."""
    i = np.arange(size) + 0.5
    e = np.minimum(i, size - i) / size
    m = np.minimum(e[:, None], e[None, :])
    return np.clip(m / margin, 0.0, 1.0)


# ------------------------------------------------------------------ texture từng loại tile
def stone_texture(base: Color, seed: int, size: int = TEX_SIZE, tint: float = 0.06,
                  contrast: float = 0.25, cracks: float = 0.0,
                  crack_glow: Optional[Color] = None) -> np.ndarray:
    """Đá: fBm grain + sai khác màu theo tile + (tùy chọn) vết nứt, có thể phát sáng.
    Độ sáng chỉ dao động trong khoảng ±contrast để mặt sàn vẫn dễ đọc (quy tắc readability)."""
    rng = np.random.default_rng(seed)
    n = fbm_noise(size, size, octaves=4, seed=seed)
    lum = 1.0 + contrast * (n * 2 - 1)
    lum *= 1.0 + rng.uniform(-tint, tint)
    rgb = np.array(base, dtype=float)[None, None, :] * lum[..., None]

    # hạt mịn
    grain = (rng.random((size, size)) - 0.5) * 0.05
    rgb *= (1.0 + grain)[..., None]

    if cracks > 0:
        c = voronoi_cracks(size, size, seed, n_points=6 + int(cracks * 4))
        c = c * cracks
        if crack_glow is not None:
            glow = np.array(crack_glow, dtype=float)[None, None, :]
            rgb = rgb * (1 - c[..., None] * 0.85) + glow * c[..., None]
        else:
            rgb *= (1 - 0.55 * c)[..., None]

    # AO nhẹ ở mép: tile tách bạch nhau
    ao = 0.82 + 0.18 * edge_factor(size)
    rgb *= ao[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def bridge_texture(base: Color, seed: int, size: int = TEX_SIZE,
                   glow: Color = (120, 235, 255)) -> np.ndarray:
    """Tấm soul-glass / kim loại nhạt: khung khắc chìm + đường rune phát sáng cyan ở giữa."""
    n = fbm_noise(size, size, octaves=3, seed=seed)
    rgb = np.array(base, dtype=float)[None, None, :] * (1.0 + 0.07 * (n * 2 - 1))[..., None]
    idx = np.arange(size) + 0.5
    yy, xx = np.meshgrid(idx, idx, indexing="ij")
    f, cx = 0.14 * size, size / 2
    inside = (xx > f - 1) & (xx < size - f + 1) & (yy > f - 1) & (yy < size - f + 1)
    d_frame = np.minimum(np.minimum(np.abs(xx - f), np.abs(xx - (size - f))),
                         np.minimum(np.abs(yy - f), np.abs(yy - (size - f))))
    frame = np.exp(-(d_frame / 0.9) ** 2) * inside
    d_seam = np.minimum(np.abs(xx - cx), np.abs(yy - cx))
    along = np.where(np.abs(xx - cx) < np.abs(yy - cx), yy, xx)       # tọa độ dọc theo đường seam
    gap = (np.floor(along / (size / 12)) % 3) == 0                    # đứt quãng như chữ rune
    seam = np.exp(-(d_seam / 0.9) ** 2) * inside * np.where(gap, 0.35, 1.0)
    halo = np.exp(-(d_seam / 3.5) ** 2) * inside * 0.18
    g = np.array(glow, dtype=float)[None, None, :]
    rgb = rgb * (1 - 0.35 * frame[..., None])
    rgb = rgb * (1 - halo[..., None]) + g * halo[..., None]
    rgb = rgb * (1 - seam[..., None]) + g * seam[..., None]
    rgb *= (0.84 + 0.16 * edge_factor(size))[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def tile_texture(kind: str, base: Color, seed: int, size: int = TEX_SIZE) -> np.ndarray:
    """kind: 'floor' | 'fragile' | 'bridge' (goal và switch dùng 'floor')."""
    if kind == "fragile":
        # khác sàn thường cả về hoa văn (nứt dày) lẫn màu -> không chỉ dựa vào hue
        return stone_texture(base, seed, size, tint=0.04, contrast=0.12, cracks=1.0,
                             crack_glow=(255, 176, 96))
    if kind == "bridge":
        pale = tuple(int(c * 0.45 + v * 0.55) for c, v in zip(base, (190, 205, 215)))
        return bridge_texture(pale, seed, size)
    return stone_texture(base, seed, size, cracks=0.18 if (seed % 3 == 0) else 0.0)


# ------------------------------------------------------------------ ánh xạ lên mặt chiếu
def map_top_face(tex: np.ndarray, p00: Sequence[float], p01: Sequence[float], p10: Sequence[float],
                 inset: float = 0.03) -> Optional[Tuple[pygame.Surface, Tuple[int, int]]]:
    """Dán texture vuông lên hình bình hành trên màn hình (inverse mapping, numpy).

    p00 = chiếu của góc (r, c), p01 = góc (r, c+1), p10 = góc (r+1, c).
    inset chừa viền nhỏ để các đường viền của _box_ops vẫn nhìn thấy.
    Trả về (surface RGBA, (x0, y0)) để blit, hoặc None nếu mặt không hướng lên camera.
    """
    p00, p01, p10 = (np.asarray(p, dtype=float) for p in (p00, p01, p10))
    p11 = p01 + p10 - p00
    quad = np.stack([p00, p01, p10, p11])
    x0, y0 = np.floor(quad.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + 1
    w, h = int(x1 - x0), int(y1 - y0)

    a, b = p01 - p00, p10 - p00                      # a: hướng cột, b: hướng hàng
    A = np.array([[a[0], b[0]], [a[1], b[1]]])
    if np.linalg.det(A) < 1e-6:        # mặt úp xuống / suy biến (cầu đang gập dưới tile kề)
        return None
    inv = np.linalg.inv(A)

    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    dx, dy = gx - p00[0], gy - p00[1]
    uc = inv[0, 0] * dx + inv[0, 1] * dy
    ur = inv[1, 0] * dx + inv[1, 1] * dy

    inside = (uc >= inset) & (uc <= 1 - inset) & (ur >= inset) & (ur <= 1 - inset)
    S = tex.shape[0]
    ix = np.clip((uc * S).astype(int), 0, S - 1)
    iy = np.clip((ur * S).astype(int), 0, S - 1)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = tex[iy, ix, :3]
    rgba[..., 3] = np.where(inside, 255, 0)
    surf = pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), (w, h), "RGBA").convert_alpha()
    return surf, (int(x0), int(y0))