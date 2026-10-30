"""camera.py - Camera góc nhìn cố định, chỉ có zoom. Không đọc hay ghi trạng thái game."""
from __future__ import annotations

import math

import numpy as np

YAW = 20.0     # độ xoay mặt bảng (góc phải trên cùng chếch lên)
PITCH = 45.0   # độ dốc khi nhìn xuống
MIN_ZOOM, MAX_ZOOM = 0.45, 2.6
ZOOM_STEP = 1.1


def view_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Ma trận chiếu trực giao 2x3: (row, col, z) -> (x, y) màn hình, đơn vị bề rộng ô."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    s, c = math.sin(yaw), math.cos(yaw)
    k, h = math.sin(pitch), math.cos(pitch)
    return np.array(((s, c, 0.0), (k * c, -k * s, -h)), dtype=float)


class Camera:
    def __init__(self) -> None:
        self.version = 0     # tăng mỗi khi zoom đổi, renderer dùng để biết phải dựng lại
        self.zoom = 1.0

    def reset(self) -> None:
        self.zoom = 1.0
        self.version += 1

    def zoom_by(self, steps: float) -> None:
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * ZOOM_STEP ** steps))
        self.version += 1

    def matrix(self) -> np.ndarray:
        return view_matrix(YAW, PITCH)
