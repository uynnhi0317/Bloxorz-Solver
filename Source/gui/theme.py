"""theme.py - Theme "Hollow" (cảm hứng từ Hollow Knight, không dùng asset gốc): bảng màu + nền động."""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
import pygame

# ---- màu thế giới (renderer dùng)
PALETTE = {
    "floor": (112, 128, 164),          # đá xanh xám
    "fragile": (44, 36, 38),         # hổ phách: nổi bật trên nền xanh
    "bridge": (226, 182, 78),          # xanh ngọc phát sáng
    "goal_hole": (6, 8, 16),
    "goal_inner": (28, 36, 64),
    "soft": (110, 224, 164),
    "heavy": (240, 104, 104),
    "split": (160, 150, 255),
    "block": (226, 230, 240),          # vỏ trắng ngà
    "cube_active": (236, 240, 250),
    "cube_idle": (40, 44, 66),
}
WATER_Z = -0.2   # mặt nước, thấp hơn mặt tile một chút

# ---- màu giao diện (controls/app dùng)
UI = {
    "bg": (6, 10, 24), "bar": (20, 29, 64), "panel": (20, 30, 62), "border": (78, 98, 150),
    "text": (234, 240, 252), "muted": (146, 167, 209), "accent": (176, 208, 255),
    "accent_dark": (54, 83, 153), "win": (196, 209, 255), "lose": (255, 146, 56),
}
SERIF = "palatinolinotype,palatino,garamond,georgia,timesnewroman,dejavuserif,liberationserif,freeserif"


def _glow_sprite(radius: int, color: Tuple[int, int, int], gain: float = 1.0) -> pygame.Surface:
    """Chấm sáng mềm (nền đen) để vẽ cộng (BLEND_RGB_ADD)."""
    d = 2 * radius + 1
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    a = np.clip(1 - np.sqrt(x * x + y * y) / radius, 0, 1) ** 2 * gain
    arr = (a[..., None] * np.array(color)).clip(0, 255).astype(np.uint8)
    return pygame.surfarray.make_surface(arr.swapaxes(0, 1))


def _backdrop(w: int, h: int) -> pygame.Surface:
    """Gradient dọc (tối trên, xanh sáng dưới) + vignette tối hai mép."""
    y = np.linspace(0, 1, h)[None, :, None]
    x = np.linspace(-1, 1, w)[:, None, None]
    top, bot = np.array((3, 5, 12)), np.array((24, 52, 120))
    col = top + (bot - top) * y ** 1.6
    vign = np.clip(1 - 0.9 * np.abs(x) ** 1.8, 0.0, 1.0)
    return pygame.surfarray.make_surface((col * vign).clip(0, 255).astype(np.uint8))


def _shaft(w: int, h: int, color: Tuple[int, int, int], peak: int) -> pygame.Surface:
    """Tia sáng dọc: mờ dần hai bên (gauss) và mờ dần về phía trên."""
    x = np.linspace(-1, 1, w)[:, None]
    y = np.linspace(0, 1, h)[None, :]
    alpha = (peak * np.exp(-2 * (x / 0.6) ** 2) * y ** 1.3).clip(0, 255).astype(np.uint8)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    rgb = pygame.surfarray.pixels3d(s)
    rgb[...] = color
    del rgb
    a = pygame.surfarray.pixels_alpha(s)
    a[...] = alpha
    del a
    return s


class Background:
    """Nền động: gradient + vài tia sáng xanh/trắng nhấp nháy nhẹ + hạt bụi sáng trôi lên."""
    # (vị trí x theo tỉ lệ bề rộng, độ rộng tia theo tỉ lệ, màu, độ sáng tối đa)
    SHAFTS = [(0.33, 0.30, (120, 170, 255), 40), (0.45, 0.50, (110, 160, 255), 60), 
              (0.55, 0.30, (235, 244, 255), 40), (0.70, 0.60, (235, 244, 255), 70),
              (0.90, 0.70, (120, 170, 255), 65)]
    N_PARTICLES = 46

    def __init__(self, size: Tuple[int, int], seed: int = 7):
        self.rng = random.Random(seed)
        self.t = 0.0
        self.sprites: List[List[pygame.Surface]] = [
            [_glow_sprite(r, (150, 200, 255), g) for g in (0.45, 0.65, 1.0)] for r in (2, 3, 5)]
        self.resize(size)

    def resize(self, size: Tuple[int, int]) -> None:
        w, h = size
        self.size = size
        self.backdrop = _backdrop(w, h)
        self.shafts = [(_shaft(max(8, int(w * wr)), h, col, pk), int(w * xr - w * wr / 2), pk,
                        self.rng.uniform(0, 6.28), self.rng.uniform(0.25, 0.6))
                       for xr, wr, col, pk in self.SHAFTS]
        # mỗi hạt: x, y (tỉ lệ 0..1), tốc độ bay lên, biên độ lắc, pha, cỡ sprite
        self.particles = [[self.rng.random(), self.rng.random(), self.rng.uniform(0.01, 0.04),
                           self.rng.uniform(0.004, 0.015), self.rng.uniform(0, 6.28),
                           self.rng.choice((0, 0, 1, 1, 2))] for _ in range(self.N_PARTICLES)]

    def update(self, dt_ms: float) -> None:
        dt = dt_ms / 1000
        self.t += dt
        for p in self.particles:
            p[1] -= p[2] * dt
            if p[1] < -0.05:
                p[0], p[1] = self.rng.random(), 1.05

    def draw(self, surf: pygame.Surface) -> None:
        w, h = self.size
        surf.blit(self.backdrop, (0, 0))
        for s, x, peak, phase, speed in self.shafts:
            s.set_alpha(int(255 * (0.72 + 0.28 * math.sin(self.t * speed + phase))))
            surf.blit(s, (x, 0))
        for px, py, _, sway, phase, size in self.particles:
            level = int((math.sin(self.t * 1.3 + phase) * 0.5 + 0.5) * 2.99)
            spr = self.sprites[size][level]
            x = (px + sway * math.sin(self.t * 0.7 + phase)) * w
            surf.blit(spr, (x - spr.get_width() // 2, py * h - spr.get_height() // 2),
                      special_flags=pygame.BLEND_RGB_ADD)