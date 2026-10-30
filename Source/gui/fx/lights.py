"""fx/lights.py - Sprite phát sáng (vẽ cộng BLEND_RGB_ADD) + hàm làm mượt cho ánh sáng tương tác."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pygame

Color = Tuple[int, int, int]
_GLOWS: Dict[tuple, pygame.Surface] = {}
_BEAMS: Dict[tuple, pygame.Surface] = {}
_CAP = 300


def approach(value: float, target: float, rate: float, dt: float) -> float:
    """Tiến dần về target theo hàm mũ (độc lập frame-rate). rate ~ 1/giây: càng lớn càng nhanh."""
    return target + (value - target) * math.exp(-rate * dt)


def glow_sprite(radius: float, color: Color, gain: float = 1.0) -> pygame.Surface:
    """Chấm sáng tròn mềm trên nền đen (blit với BLEND_RGB_ADD).
    radius lượng tử hóa theo 6px và gain theo 1/24 để cache không phình ra khi giá trị thay đổi liên tục."""
    radius = max(6, int(round(radius / 6)) * 6)
    gain = max(0.0, min(2.0, round(gain * 24) / 24))
    key = (radius, tuple(color), gain)
    if key not in _GLOWS:
        if len(_GLOWS) > _CAP:
            _GLOWS.clear()
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        a = np.clip(1 - np.sqrt(x * x + y * y) / radius, 0, 1) ** 2 * gain
        arr = (a[..., None] * np.array(color)).clip(0, 255).astype(np.uint8)
        _GLOWS[key] = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
    return _GLOWS[key]


def beam_sprite(width: float, height: float, color: Color, peak: float = 0.5) -> pygame.Surface:
    """Tia sáng dọc: sáng nhất ở ĐÁY (chỗ phát sáng), mờ dần lên trên về 0, mờ dần hai bên."""
    width, height = max(4, int(width)), max(4, int(height))
    peak = max(0.0, min(1.5, round(peak * 24) / 24))
    key = (width, height, tuple(color), peak)
    if key not in _BEAMS:
        if len(_BEAMS) > _CAP:
            _BEAMS.clear()
        x = np.linspace(-1, 1, width)[:, None]
        t = np.linspace(0, 1, height)[None, :]            # index 0 = trên cùng, index cuối = đáy
        a = peak * np.exp(-(x / 0.55) ** 2 * 2) * t ** 1.6
        arr = (a[..., None] * np.array(color)).clip(0, 255).astype(np.uint8)
        _BEAMS[key] = pygame.surfarray.make_surface(arr)
    return _BEAMS[key]