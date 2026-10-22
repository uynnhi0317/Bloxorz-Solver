"""Animations cho renderer 2.5D.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence as Seq, Tuple

import numpy as np

from game import Action, State, SwitchKind, TileType
from .renderer import PALETTE, Box, Renderer, make_box

# action -> (axis, sign); axis 0 = row, 1 = col
DIR_AXIS = {Action.MOVE_UP: (0, -1), Action.MOVE_DOWN: (0, 1),
            Action.MOVE_LEFT: (1, -1), Action.MOVE_RIGHT: (1, 1)}
DIR_DELTA = {Action.MOVE_UP: (-1, 0), Action.MOVE_DOWN: (1, 0),
             Action.MOVE_LEFT: (0, -1), Action.MOVE_RIGHT: (0, 1)}


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def rotate(verts: np.ndarray, axis: int, sign: int, theta: float, pivot: float = None,
           slide: float = 0.0, drop: float = 0.0) -> np.ndarray:
    """Xoay các đỉnh `verts` một góc `theta` quanh một đường nằm ngang trên sàn.
    """
    out = verts.copy()
    if pivot is None:
        pivot = verts[:, axis].max() if sign > 0 else verts[:, axis].min()
        pz = verts[:, 2].min()          # độ cao trục = đáy box
    else:
        pz = 0.0                        # độ cao trục = mặt sàn
    u = sign * (verts[:, axis] - pivot)  # khoảng cách tới trục; dương = phía đang lăn tới
    w = verts[:, 2] - pz                 # độ cao so với trục

    out[:, axis] = pivot + sign * (u * math.cos(theta) + w * math.sin(theta) + slide)
    out[:, 2] = pz - u * math.sin(theta) + w * math.cos(theta) - drop
    return out


class Animation:
    done: bool = False

    def update(self, dt_ms: float) -> None:
        raise NotImplementedError

    def draw(self, surface) -> None:
        raise NotImplementedError


class RollAnimation(Animation):
    """Animation lăn cho cả khối, hay phần nửa khối đang active nếu đang trong trạng thái tách."""

    def __init__(self, renderer: Renderer, before: State, action: Action,
                 moving: int = 0, duration_ms: float = 160):
        self.r = renderer
        self.mask = before.bridge_mask
        self.axis, self.sign = DIR_AXIS[action]
        boxes = renderer.state_boxes(before)
        self.static = [b for i, b in enumerate(boxes) if i != moving]
        self.mover = boxes[moving]
        end = np.round(rotate(self.mover.verts, self.axis, self.sign, math.pi / 2), 6) + 0.0
        self.final_boxes: List[Box] = self.static + [Box(end, self.mover.color)]
        self.t, self.duration = 0.0, duration_ms

    def update(self, dt_ms: float) -> None:
        self.t = min(1.0, self.t + dt_ms / self.duration)
        self.done = self.t >= 1.0

    def draw(self, surface) -> None:
        theta = ease(self.t) * math.pi / 2
        v = rotate(self.mover.verts, self.axis, self.sign, theta)
        self.r.draw_scene(surface, self.mask, self.static + [Box(v, self.mover.color)])


class TumbleAnimation(Animation):
    """Thua: khối vừa rơi vừa lăn.

    Dựa trên góc lăn phi (0..180), chia thành các giai đoạn:
      * rotation about the pivot edge,
      * Khi phi > 90, trượt khỏi cạnh bảng, rơi nhanh dần, thêm hiệu ứng fade-out
    """
    PHI_END = math.pi
    GRAVITY = 1.6

    def __init__(self, renderer: Renderer, bridge_mask: int, static: Seq[Box], mover: Box,
                 axis: int, sign: int, pivot: float, duration_ms: float = 650):
        self.r, self.mask, self.static, self.mover = renderer, bridge_mask, list(static), mover
        self.axis, self.sign, self.pivot = axis, sign, pivot

        u = sign * (mover.verts[:, axis] - pivot)
        self.reach = max(0.0, float(u.max()))
        self.duration, self.t = duration_ms, 0.0

    def pose(self, t: float) -> np.ndarray:
        phi = self.PHI_END * t ** 1.5
        slide = self.reach * max(0.0, -math.cos(phi))
        drop = self.GRAVITY * max(0.0, phi - math.pi / 2) ** 2
        return rotate(self.mover.verts, self.axis, self.sign, phi, self.pivot, slide, drop)

    def update(self, dt_ms: float) -> None:
        self.t = min(1.0, self.t + dt_ms / self.duration)
        self.done = self.t >= 1.0

    def draw(self, surface) -> None:
        alpha = int(255 * (1 - ease(max(0.0, (self.t - 0.4) / 0.6))))
        self.r.draw_scene(surface, self.mask,
                          self.static + [Box(self.pose(self.t), self.mover.color)], alpha)


class DropAnimation(Animation):
    """Khối chìm thẳng (gạch vỡ, lăn khỏi mặt bảng, thắng)."""

    def __init__(self, renderer: Renderer, bridge_mask: int, boxes: Seq[Box],
                 duration_ms: float = 450, depth: float = 4.0, fade: bool = True):
        self.r, self.mask, self.boxes = renderer, bridge_mask, list(boxes)
        self.duration, self.depth, self.fade = duration_ms, depth, fade
        self.t = 0.0

    def update(self, dt_ms: float) -> None:
        self.t = min(1.0, self.t + dt_ms / self.duration)
        self.done = self.t >= 1.0

    def draw(self, surface) -> None:
        dz = -self.depth * self.t * self.t
        moved = []
        for b in self.boxes:
            v = b.verts.copy()
            v[:, 2] += dz
            moved.append(Box(v, b.color))
        alpha = int(255 * (1 - self.t)) if self.fade else 255
        self.r.draw_scene(surface, self.mask, moved, alpha)


class SpawnAnimation(Animation):
    """Animation 'tách': hai khối con nhảy khỏi công tắc đến destination."""

    def __init__(self, renderer: Renderer, bridge_mask: int, origin: Tuple[int, int],
                 targets: Seq[Tuple[int, int]], active: Optional[Tuple[int, int]] = None,
                 duration_ms: float = 320):
        self.r, self.mask, self.origin, self.targets = renderer, bridge_mask, origin, targets
        self.active = active
        self.duration, self.t = duration_ms, 0.0

    def update(self, dt_ms: float) -> None:
        self.t = min(1.0, self.t + dt_ms / self.duration)
        self.done = self.t >= 1.0

    def draw(self, surface) -> None:
        k, hop = ease(self.t), math.sin(math.pi * self.t)
        boxes = []
        for i, (tr, tc) in enumerate(self.targets):
            r = self.origin[0] + (tr - self.origin[0]) * k
            c = self.origin[1] + (tc - self.origin[1]) * k
            color = PALETTE["cube_active"] if (tr, tc) == self.active else PALETTE["cube_idle"]
            boxes.append(Box(make_box(r, c, hop, 1, 1, 1), color))
        self.r.draw_scene(surface, self.mask, boxes)


class Sequence(Animation):
    """Vẽ từng frame"""

    def __init__(self, *anims: Animation):
        self.anims = list(anims)
        self.done = not self.anims

    def update(self, dt_ms: float) -> None:
        if self.anims:
            self.anims[0].update(dt_ms)
            if self.anims[0].done:
                self.anims.pop(0)
        self.done = not self.anims

    def draw(self, surface) -> None:
        if self.anims:
            self.anims[0].draw(surface)


# --------------------------------------------------------------- factories
def _moving_index(before: State) -> int:
    """Index (trong before.split_positions) của cube đang được điều khiển.
    Nếu chưa tách --> chỉ có một khối, trả về 0."""
    if before.split_positions is None:
        return 0
    return before.split_positions.index(before.selected_cube)


def is_bump(before: State, action: Action) -> bool:
    """Cube đang điều khiển đi thẳng vào ô của cube còn lại: nước đi bị chặn (get_next_state
    trả None) nhưng KHÔNG phải thua --> caller nên bỏ qua, không gọi animate_fall."""
    if before.split_positions is None or action not in DIR_DELTA:
        return False
    dr, dc = DIR_DELTA[action]
    r, c = before.selected_cube
    return (r + dr, c + dc) in before.split_positions


def _split_origin(renderer: Renderer, landed: np.ndarray) -> Tuple[int, int]:
    """Ô split switch mà khối vừa lăn lên (dùng làm điểm xuất phát của 2 cube)."""
    box = Box(landed, PALETTE["block"])
    cells = Renderer._cells_under(box)
    for cell in cells:
        tile = renderer.board.tile_at(cell)
        if tile.kind == TileType.SWITCH and tile.switch_kind == SwitchKind.SPLIT:
            return cell
    return cells[0]


def animate_move(renderer: Renderer, before: State, action: Action, after: State,
                 duration_ms: float = 160) -> Animation:
    """Animation của nước đi hợp lệ (action phải là MOVE_*)."""
    roll = RollAnimation(renderer, before, action,
                         moving=_moving_index(before), duration_ms=duration_ms)
    if after.split_positions is not None and before.split_positions is None:
        origin = _split_origin(renderer, roll.final_boxes[0].verts)
        return Sequence(roll, SpawnAnimation(renderer, after.bridge_mask, origin,
                                             after.split_positions, after.selected_cube))
    return roll


def _is_floor(board, cell, mask: int) -> bool:
    """Kiểm tra cell có đứng được?"""
    if not board.in_bounds(cell):
        return False
    tile = board.tile_at(cell)
    if tile.kind == TileType.VOID:
        return False
    if tile.kind == TileType.BRIDGE and mask & board.bridge_bit[tile.tile_id]:
        return False
    return True


def animate_fall(renderer: Renderer, before: State, action: Optional[Action] = None,
                 moving: Optional[int] = None) -> Animation:
    """Animation 'thua' sau nước đi không hợp lệ.

    * action=None: rơi thẳng tại chỗ (e.g. standing on a fragile tile).
    * Nước đi kết thúc với box nằm hẳn ngoài bảng: lăn theo hướng cũ
    * Nước đi kết thúc với box nằm một nửa trên bảng, nửa trên void: lăn về phía cell void
    """
    mask = before.bridge_mask
    if action is None:
        return DropAnimation(renderer, mask, renderer.state_boxes(before))
    if moving is None:
        moving = _moving_index(before)
    board = renderer.board
    roll = RollAnimation(renderer, before, action, moving=moving)
    end = roll.final_boxes[-1]
    cells = Renderer._cells_under(end)
    solid = [c for c in cells if _is_floor(board, c, mask)]
    void = [c for c in cells if c not in solid]

    if not solid:  # lăn khỏi bảng, giữ hướng lăn cũ
        axis, sign = DIR_AXIS[action]
        v = roll.mover.verts
        pivot = v[:, axis].max() if sign > 0 else v[:, axis].min()
        return TumbleAnimation(renderer, mask, roll.static, roll.mover, axis, sign, float(pivot))
    if not void:  # không đứng được, nhưng cũng không ra khỏi bảng (vd như gạch dễ vỡ)
        return Sequence(roll, DropAnimation(renderer, mask, roll.final_boxes))

    # nửa trên bảng, nửa trên void --> lăn về cell void
    (sr, sc), (vr, vc) = solid[0], void[0]
    dr, dc = vr - sr, vc - sc
    axis = 0 if dr else 1
    sign = dr if dr else dc
    pivot = (sr if axis == 0 else sc) + (1 if sign > 0 else 0)
    return Sequence(roll, TumbleAnimation(renderer, mask, roll.static, end, axis, sign, float(pivot)))


def animate_win(renderer: Renderer, state: State) -> Animation:
    """Block sinks into the goal hole."""
    return DropAnimation(renderer, state.bridge_mask, renderer.state_boxes(state),
                         duration_ms=600, depth=2.5)