"""texture_loader.py - Đọc ảnh bề mặt thành mảng điểm ảnh. Không cần cửa sổ đồ họa đang mở."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pygame

from .model_loader import AssetError

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp"}


@dataclass
class Texture:
    path: Path
    pixels: np.ndarray                      # (W, H, 3) uint8, trục x trước theo quy ước surfarray
    mean: Tuple[float, float, float]        # màu trung bình

    @property
    def size(self) -> Tuple[int, int]:
        return self.pixels.shape[0], self.pixels.shape[1]

    def sample_mean(self, uvs: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
        """Màu trung bình tại các điểm uv (lặp theo chu kỳ, v tính từ dưới lên như chuẩn OBJ)."""
        w, h = self.size
        total = np.zeros(3)
        for u, v in uvs:
            total += self.pixels[int((u % 1.0) * w) % w, int((1.0 - v % 1.0) * h) % h]
        return tuple(total / max(1, len(uvs)))  # type: ignore[return-value]


def load_texture(path: Path) -> Texture:
    """Raise AssetError nếu định dạng không hỗ trợ hoặc ảnh hỏng."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise AssetError(f"{path}: định dạng ảnh không hỗ trợ")
    try:
        img = pygame.image.load(str(path))
    except (pygame.error, OSError, FileNotFoundError) as exc:
        raise AssetError(f"{path}: không đọc được ảnh ({exc})") from exc
    pixels = pygame.surfarray.array3d(img)
    if pixels.size == 0:
        raise AssetError(f"{path}: ảnh rỗng")
    return Texture(path, pixels, tuple(float(v) for v in pixels.reshape(-1, 3).mean(axis=0)))  # type: ignore[arg-type]
