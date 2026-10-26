"""Animations cho renderer 2.5D.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import numpy as np

from game import Action, Coord, State, SwitchKind, TileType
from .renderer import PALETTE, Box, Renderer, TileFx, make_box, rotate

# action -> (axis, sign); axis 0 = row, 1 = col
DIR_AXIS = {Action.MOVE_UP: (0, -1), Action.MOVE_DOWN: (0, 1),
            Action.MOVE_LEFT: (1, -1), Action.MOVE_RIGHT: (1, 1)}
DIR_DELTA = {Action.MOVE_UP: (-1, 0), Action.MOVE_DOWN: (1, 0),
             Action.MOVE_LEFT: (0, -1), Action.MOVE_RIGHT: (0, 1)}


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


class Animation:
    done: bool = False
    # Hiệu ứng tile còn lại sau khi animation kết thúc (vd tile vỡ/đã rơi hết phải tiếp tục ẩn
    # khi vẽ màn thắng/thua). None = sàn vẽ lại bình thường.
    end_fx: Optional[Dict[Coord, TileFx]] = None

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
    """Khối chìm thẳng (gạch vỡ, lăn khỏi mặt bảng, thắng).
    fall_tiles: các ô tile rơi theo khối (vd ô fragile bị đứng lên), nghiêng nhẹ; rơi xong thì ẩn hẳn."""
    TILT = 0.35  # rad (~20 độ)

    def __init__(self, renderer: Renderer, bridge_mask: int, boxes: Seq[Box],
                 duration_ms: float = 450, depth: float = 4.0, fade: bool = True,
                 fall_tiles: Seq[Coord] = ()):
        self.r, self.mask, self.boxes = renderer, bridge_mask, list(boxes)
        self.duration, self.depth, self.fade = duration_ms, depth, fade
        self.fall_tiles = list(fall_tiles)
        self.end_fx = {c: TileFx(hidden=True) for c in self.fall_tiles} or None
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
        fx = {c: TileFx(xform=self._tilt(c)) for c in self.fall_tiles}
        self.r.draw_scene(surface, self.mask, moved, alpha, fx)

    def _tilt(self, cell: Coord):
        """Ô fragile vừa rơi vừa nghiêng nhẹ quanh một cạnh."""
        r, c = cell
        sign = 1 if (r + c) % 2 == 0 else -1
        theta, drop = self.TILT * self.t, self.depth * self.t * self.t
        return lambda v: rotate(v, 1, sign, theta, c if sign > 0 else c + 1, 0.0, drop)


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

    @property
    def end_fx(self) -> Optional[Dict[Coord, TileFx]]:
        return self.anims[-1].end_fx if self.anims else None


def _stagger(n: int, per_ms: float = 35.0, span_ms: float = 800.0) -> float:
    """Độ trễ khi intro/outro giữa 2 tile liên tiếp; co lại khi màn có nhiều tile để tổng không quá span_ms."""
    return 0.0 if n <= 1 else min(per_ms, span_ms / (n - 1))


class IntroAnimation(Animation):
    """Vào màn: tile bay thẳng từ dưới lên lần lượt theo thứ tự row 0->n, col 0->n,
    sau khi tile cuối vào chỗ thì khối rơi từ trên xuống."""
    TILE_MS, CUBE_MS = 300.0, 150.0
    TILE_DEPTH, CUBE_HEIGHT = 12.0, 5.0

    def __init__(self, renderer: Renderer, state: State):
        self.r, self.state = renderer, state
        self.cells = renderer.drawn_tiles(state.bridge_mask)
        self.step = _stagger(len(self.cells))
        self.tiles_end = self.step * max(0, len(self.cells) - 1) + self.TILE_MS
        self.total = self.tiles_end + self.CUBE_MS
        self.boxes = renderer.state_boxes(state)
        self.t = 0.0

    def update(self, dt_ms: float) -> None:
        self.t = min(self.total, self.t + dt_ms)
        self.done = self.t >= self.total

    def draw(self, surface) -> None:
        fx: Dict[Coord, TileFx] = {}
        for i, cell in enumerate(self.cells):
            p = (self.t - i * self.step) / self.TILE_MS
            if p <= 0:
                fx[cell] = TileFx(hidden=True)
            elif p < 1:
                fx[cell] = TileFx(dz=-self.TILE_DEPTH * (1 - p) ** 3)  # nhanh rồi chậm dần
        boxes, alpha = [], 255
        cp = (self.t - self.tiles_end) / self.CUBE_MS
        if cp > 0:
            dz = self.CUBE_HEIGHT * (1 - min(cp, 1.0) ** 2)  # rơi nhanh dần
            for b in self.boxes:
                v = b.verts.copy()
                v[:, 2] += dz
                boxes.append(Box(v, b.color))
            alpha = int(255 * min(1.0, cp * 3))
        self.r.draw_scene(surface, self.state.bridge_mask, boxes, alpha, fx)


class OutroAnimation(Animation):
    """Kết màn: tile rơi thẳng xuống, theo đúng thứ tự lúc bay vào."""
    TILE_MS, TILE_DEPTH = 200.0, 12.0

    def __init__(self, renderer: Renderer, bridge_mask: int):
        self.r, self.mask = renderer, bridge_mask
        self.cells = renderer.drawn_tiles(bridge_mask)
        self.step = _stagger(len(self.cells))
        self.total = self.step * max(0, len(self.cells) - 1) + self.TILE_MS
        self.end_fx = {c: TileFx(hidden=True) for c in self.cells}
        self.t = 0.0

    def update(self, dt_ms: float) -> None:
        self.t = min(self.total, self.t + dt_ms)
        self.done = self.t >= self.total

    def draw(self, surface) -> None:
        fx: Dict[Coord, TileFx] = {}
        for i, cell in enumerate(self.cells):
            p = (self.t - i * self.step) / self.TILE_MS
            if p >= 1:
                fx[cell] = TileFx(hidden=True)
            elif p > 0:
                fx[cell] = TileFx(dz=-self.TILE_DEPTH * p * p)  # rơi nhanh dần
        self.r.draw_scene(surface, self.mask, [], 255, fx)


class BridgeFlips:
    """Cầu đổi trạng thái: lật quanh bản lề (cạnh chung với tile kề) giữa "gập dưới tile kề" (đóng)
    và "phẳng" (mở). KHÔNG chặn input: chạy nền, caller gọi update() mỗi frame và
    gán renderer.ambient_fx = flips.fx(), nên người chơi đi tiếp được ngay trong lúc cầu lật.
    Cầu không có bản lề (không có tile kề) thì trồi lên / chìm xuống."""
    DURATION_MS, STAGGER_MS = 170.0, 25.0

    def __init__(self, renderer: Renderer):
        self.r = renderer
        self.items: List[list] = []  # [cell, opening, thời gian đã chạy (âm = chưa tới lượt)]

    def clear(self) -> None:
        self.items = []

    def start(self, old_mask: int, new_mask: int) -> None:
        board, diff, i = self.r.board, old_mask ^ new_mask, 0
        self.items = [it for it in self.items if it[0] not in
                      {(r, c) for r, row in enumerate(board.tiles) for c, t in enumerate(row)
                       if t.kind == TileType.BRIDGE and diff & board.bridge_bit[t.tile_id]}]
        for r, row in enumerate(board.tiles):
            for c, tile in enumerate(row):
                if tile.kind == TileType.BRIDGE and diff & board.bridge_bit[tile.tile_id]:
                    opening = not (new_mask & board.bridge_bit[tile.tile_id])
                    self.items.append([(r, c), opening, -i * self.STAGGER_MS])
                    i += 1

    def update(self, dt_ms: float) -> None:
        for it in self.items:
            it[2] += dt_ms
        self.items = [it for it in self.items if it[2] < self.DURATION_MS]

    def fx(self) -> Dict[Coord, TileFx]:
        out: Dict[Coord, TileFx] = {}
        for cell, opening, t in self.items:
            q = ease(min(1.0, max(0.0, t / self.DURATION_MS)))
            if cell in self.r.hinges:
                theta = math.pi * ((1 - q) if opening else q)
                out[cell] = TileFx(theta=theta, under=theta > math.pi / 2)
            elif opening:   # không bản lề: trồi lên từ dưới
                out[cell] = TileFx(hidden=True) if t <= 0 else TileFx(dz=-6.0 * (1 - q))
            else:           # không bản lề: chìm xuống rồi biến mất (cầu đã đóng nên mask không vẽ nữa)
                out[cell] = TileFx(dz=-6.0 * q, theta=0.0)
        return out


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
    """Animation của nước đi hợp lệ (action phải là MOVE_*).
    Thứ tự: lăn -> (tách cube nếu dẫm split switch). Cầu đổi trạng thái do BridgeFlips lo (chạy nền)."""
    roll = RollAnimation(renderer, before, action,
                         moving=_moving_index(before), duration_ms=duration_ms)
    parts: List[Animation] = [roll]
    if after.split_positions is not None and before.split_positions is None:
        origin = _split_origin(renderer, roll.final_boxes[0].verts)
        parts.append(SpawnAnimation(renderer, before.bridge_mask, origin,
                                    after.split_positions, after.selected_cube))
    return parts[0] if len(parts) == 1 else Sequence(*parts)


def animate_intro(renderer: Renderer, state: State) -> Animation:
    """Vào màn / restart: tile bay lên rồi khối rơi xuống."""
    return IntroAnimation(renderer, state)


def animate_outro(renderer: Renderer, state: State) -> Animation:
    """Sau khi thắng: tile rơi hết xuống."""
    return OutroAnimation(renderer, state.bridge_mask)


def _fragile_cells(renderer: Renderer, boxes: Seq[Box]) -> List[Coord]:
    """Các ô fragile nằm dưới một khối đứng thẳng (khối 1x1x2 thẳng đứng) -> tile sẽ vỡ theo."""
    out: List[Coord] = []
    for bx in boxes:
        v = bx.verts
        if v[:, 2].max() - v[:, 2].min() < 1.5:   # chỉ khối cao 2 (đứng) mới làm vỡ
            continue
        for cell in Renderer._cells_under(bx):
            if renderer.board.tile_at(cell).kind == TileType.FRAGILE:
                out.append(cell)
    return out


def _drop_with_tiles(renderer: Renderer, mask: int, boxes: Seq[Box]) -> Animation:
    """Khối rơi; nếu đang đứng trên ô fragile thì ô đó rơi theo."""
    tiles = _fragile_cells(renderer, boxes)
    if tiles:
        return DropAnimation(renderer, mask, boxes, duration_ms=600, depth=12.0, fall_tiles=tiles)
    return DropAnimation(renderer, mask, boxes)


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
        return _drop_with_tiles(renderer, mask, renderer.state_boxes(before))
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
        return Sequence(roll, _drop_with_tiles(renderer, mask, roll.final_boxes))

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