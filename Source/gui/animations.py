"""Animations cho renderer 2.5D.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import numpy as np
import pygame

from game import Action, Board, Coord, Move, State, SwitchKind, TileType
from game.rules import _is_switch_triggered, roll as roll_state
from .renderer import PALETTE, Box, Renderer, TileFx, make_box, rotate
from .theme import glow_sprite, petal_sprite

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


def _aligned_box(box: Box, verts: np.ndarray) -> Box:
    """Box nằm thẳng theo lưới ở vị trí `verts`, với tư thế hình Vô Diện đúng cho trạng thái mới."""
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    d = hi - lo
    pose = "standing" if d[2] > 1.5 else "horizontal" if d[1] > 1.5 else "vertical" if d[0] > 1.5 else "standing"
    return replace(box, verts=make_box(lo[0], lo[1], lo[2], d[0], d[1], d[2]), pose=pose, fade=1.0)


class RollAnimation(Animation):
    """Animation lăn cho cả khối, hay phần nửa khối đang active nếu đang trong trạng thái tách.
    Nửa đầu cú lăn: hình theo tư thế cũ; nửa sau: theo tư thế mới. Mặt nạ mờ đi đúng lúc đổi tư thế."""

    def __init__(self, renderer: Renderer, before: State, action: Action,
                 moving: int = 0, duration_ms: float = 160):
        self.r = renderer
        self.mask = before.bridge_mask
        self.axis, self.sign = DIR_AXIS[action]
        boxes = renderer.state_boxes(before)
        self.static = [b for i, b in enumerate(boxes) if i != moving]
        self.mover = boxes[moving]
        v = self.mover.verts
        self.pivot = float(v[:, self.axis].max() if self.sign > 0 else v[:, self.axis].min())
        end = np.round(rotate(v, self.axis, self.sign, math.pi / 2), 6) + 0.0
        self.end_box = _aligned_box(self.mover, end)
        self.final_boxes: List[Box] = self.static + [self.end_box]
        self.t, self.duration = 0.0, duration_ms

    def update(self, dt_ms: float) -> None:
        self.t = min(1.0, self.t + dt_ms / self.duration)
        self.done = self.t >= 1.0

    def draw(self, surface) -> None:
        theta = ease(self.t) * math.pi / 2
        fade = min(1.0, abs(theta - math.pi / 4) / (math.pi / 8))
        if theta < math.pi / 4:
            box = replace(self.mover, verts=rotate(self.mover.verts, self.axis, self.sign, theta), fade=fade)
        else:
            box = replace(self.end_box, fade=fade,
                          verts=rotate(self.end_box.verts, self.axis, self.sign, theta - math.pi / 2, self.pivot))
        self.r.draw_scene(surface, self.mask, self.static + [box])


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
                          self.static + [replace(self.mover, verts=self.pose(self.t))], alpha)


class DropAnimation(Animation):
    """Khối chìm thẳng (gạch vỡ, lăn khỏi mặt bảng).
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
            moved.append(replace(b, verts=v))
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
            on = (tr, tc) == self.active
            color = PALETTE["cube_active"] if on else PALETTE["cube_idle"]
            boxes.append(Box(make_box(r, c, hop, 1, 1, 1), color, "head" if i == 0 else "body", active=on))
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
                boxes.append(replace(b, verts=v))
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


class BridgeTransitions:
    """Cầu đổi trạng thái: các ván gỗ trượt lên và rõ dần (mở) / trượt xuống và mờ dần (đóng), khoảng 0,3 giây.
    KHÔNG chặn input và KHÔNG giữ trạng thái logic: bridge_mask của State đã đổi ngay khi nước đi hoàn tất,
    lớp này chỉ dựng hình quá độ. Caller gọi update() mỗi frame và gán renderer.ambient_fx = fx()."""
    DURATION_MS, WAVE_MS = 260.0, 40.0  # tổng tối đa 0,3 giây kể cả độ lan giữa các ô cầu

    def __init__(self, renderer: Renderer):
        self.r = renderer
        self.items: List[list] = []  # [cell, opening, thời gian đã chạy (âm = chưa tới lượt)]

    def clear(self) -> None:
        self.items = []

    def start(self, old_mask: int, new_mask: int) -> None:
        board, diff = self.r.board, old_mask ^ new_mask
        cells = [(r, c, tile) for r, row in enumerate(board.tiles) for c, tile in enumerate(row)
                 if tile.kind == TileType.BRIDGE and diff & board.bridge_bit[tile.tile_id]]
        changed = {(r, c) for r, c, _ in cells}
        self.items = [it for it in self.items if it[0] not in changed]
        step = self.WAVE_MS / max(1, len(cells) - 1)
        for i, (r, c, tile) in enumerate(cells):
            opening = not (new_mask & board.bridge_bit[tile.tile_id])
            self.items.append([(r, c), opening, -i * step])

    def update(self, dt_ms: float) -> None:
        for it in self.items:
            it[2] += dt_ms
        self.items = [it for it in self.items if it[2] < self.DURATION_MS]

    def fx(self) -> Dict[Coord, TileFx]:
        out: Dict[Coord, TileFx] = {}
        for cell, opening, t in self.items:
            q = min(1.0, max(0.0, t / self.DURATION_MS))
            out[cell] = TileFx(plank=q if opening else 1 - q)
        return out


class SwitchPulses:
    """Công tắc vừa được kích hoạt: chớp sáng, vòng sóng lan tỏa, vài cánh hoa bung ra.
    Chỉ là hình ảnh, không chặn input; ô nào được kích hoạt do triggered_switches() xác định từ State."""
    DURATION_MS, N_PETALS = 700.0, 8

    def __init__(self, renderer: Renderer):
        self.r = renderer
        self.items: List[list] = []  # [cell, màu, thời gian đã chạy, góc lệch, số cánh, phạm vi]

    def clear(self) -> None:
        self.items = []

    def trigger(self, cell: Coord, color=None, petals: Optional[int] = None, reach: float = 1.0) -> None:
        tile = self.r.board.tile_at(cell)
        if color is None:
            color = PALETTE[tile.switch_kind.value] if tile.switch_kind is not None else PALETTE["soft"]
        self.items.append([cell, color, 0.0, (cell[0] * 37 + cell[1] * 61) % 360,
                           self.N_PETALS if petals is None else petals, reach])

    def update(self, dt_ms: float) -> None:
        for it in self.items:
            it[2] += dt_ms
        self.items = [it for it in self.items if it[2] < self.DURATION_MS]

    def draw_under(self, surface) -> None:
        r3 = self.r
        for (r, c), color, t, _, _, reach in self.items:
            k = t / self.DURATION_MS
            cr, cc = r + 0.5, c + 0.5
            rad = max(4, int(r3.scale * 0.8 * reach))
            x, y = r3.project([(cr, cc, 0.0)])[0]
            surface.blit(glow_sprite(rad, color, round((1 - k) ** 2, 1)), (x - rad, y - rad),
                         special_flags=pygame.BLEND_RGB_ADD)
            for delay in (0.0, 0.22):  # hai vòng sóng nối nhau
                q = (k - delay) / (1 - delay)
                if q <= 0:
                    continue
                ring = r3._ground_ring(cr, cc, (0.25 + 0.95 * ease(q)) * reach, 36)
                x0, y0 = np.floor(ring.min(axis=0)).astype(int) - 2
                x1, y1 = np.ceil(ring.max(axis=0)).astype(int) + 3
                layer = pygame.Surface((int(x1 - x0), int(y1 - y0)))
                col = tuple(int(v * (1 - q)) for v in color)
                pygame.draw.lines(layer, col, True, (ring - (x0, y0)).tolist(), max(1, int(3 * r3.detail)))
                surface.blit(layer, (int(x0), int(y0)), special_flags=pygame.BLEND_RGB_ADD)

    def draw_over(self, surface) -> None:
        r3 = self.r
        size = max(3, int(r3.scale * 0.1))
        for (r, c), _, t, seed, n, reach in self.items:
            k = t / self.DURATION_MS
            for i in range(n):
                a = math.radians(seed + i * 360 / n)
                d = (0.2 + 0.85 * ease(k)) * reach
                z = (0.15 + 1.1 * math.sin(math.pi * min(1.0, k * 1.1))) * reach
                x, y = r3.project([(r + 0.5 + d * math.sin(a), c + 0.5 + d * math.cos(a), z)])[0]
                spr = petal_sprite(size, i * 5 + int(k * 40), i % 3, PALETTE["soft"], 235)
                spr.set_alpha(int(255 * (1 - k) ** 0.7))
                surface.blit(spr, (x - spr.get_width() / 2, y - spr.get_height() / 2))


def triggered_switches(board: Board, before: State, after: State, move: Move) -> List[Coord]:
    """Các ô công tắc vừa được kích hoạt bởi nước đi before -> after. Dùng lại đúng điều kiện của
    game.rules (không tự định nghĩa lại luật), chỉ đọc, chỉ phục vụ hiệu ứng."""
    cells = []
    for cell in set(after.occupied_cells) - set(before.occupied_cells):
        tile = board.tile_at(cell)
        if tile.kind == TileType.SWITCH and tile.switch_kind != SwitchKind.SPLIT and _is_switch_triggered(after, tile):
            cells.append(cell)
    if after.split_positions is not None and before.split_positions is None:
        landed = roll_state(before, move.action).position   # ô công tắc ( ) khối vừa đứng lên
        if board.tile_at(landed).kind == TileType.SWITCH:
            cells.append(landed)
    return sorted(cells)


# --------------------------------------------------------------- factories
def _moving_index(before: State, cube: Optional[Coord] = None) -> int:
    """Index (trong before.split_positions) của cube đang được điều khiển.
    Nếu chưa tách --> chỉ có một khối, trả về 0."""
    if before.split_positions is None or cube is None:
        return 0
    return before.split_positions.index(cube)


def is_bump(before: State, move: Move) -> bool:
    """Cube đang điều khiển đi thẳng vào ô của cube còn lại: nước đi bị chặn (get_next_state
    trả None) nhưng KHÔNG phải thua --> caller nên bỏ qua, không gọi animate_fall."""
    if before.split_positions is None or move.cube is None or move.action not in DIR_DELTA:
        return False
    dr, dc = DIR_DELTA[move.action]
    r, c = move.cube
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


def animate_move(renderer: Renderer, before: State, move: Move, after: State,
                 duration_ms: float = 160) -> Animation:
    """Animation của nước đi hợp lệ (move.action phải là MOVE_*).
    Thứ tự: lăn -> (tách cube nếu dẫm split switch). Cầu đổi trạng thái do BridgeTransitions lo (chạy nền)."""
    roll = RollAnimation(renderer, before, move.action,
                         moving=_moving_index(before, move.cube), duration_ms=duration_ms)
    parts: List[Animation] = [roll]
    if after.split_positions is not None and before.split_positions is None:
        origin = _split_origin(renderer, roll.final_boxes[0].verts)
        parts.append(SpawnAnimation(renderer, before.bridge_mask, origin,
                                    after.split_positions, after.split_positions[0])) # cube đầu - mặc định chọn sau khi tách
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


class WinAnimation(Animation):
    """Thắng: cổng torii sáng dần, xoáy sáng bùng lên, Vô Diện chìm dần vào tâm cổng rồi tan đi.
    Chỉ là hình ảnh: điều kiện thắng vẫn do is_goal quyết định trước khi animation này được tạo."""

    def __init__(self, renderer: Renderer, state: State, duration_ms: float = 1250):
        self.r, self.state = renderer, state
        self.boxes = renderer.state_boxes(state)
        self.duration, self.t = duration_ms, 0.0
        self.burst = False

    def update(self, dt_ms: float) -> None:
        self.t = min(1.0, self.t + dt_ms / self.duration)
        self.done = self.t >= 1.0
        if not self.burst and self.t > 0.12:
            self.burst = True
            if self.r.pulses is not None:
                self.r.pulses.trigger(self.r.board.goal, PALETTE["soft"], petals=14, reach=1.6)

    def draw(self, surface) -> None:
        self.r.goal_glow = ease(min(1.0, self.t / 0.3))
        sink = ease(max(0.0, (self.t - 0.22) / 0.78))
        moved = []
        for b in self.boxes:
            v = b.verts.copy()
            v[:, 2] -= 2.05 * sink
            moved.append(replace(b, verts=v))
        alpha = int(255 * (1 - ease(max(0.0, (self.t - 0.72) / 0.28))))
        self.r.draw_scene(surface, self.state.bridge_mask, moved, alpha, hide_below=True)
        self.r.goal_glow = 0.0


def animate_win(renderer: Renderer, state: State) -> Animation:
    """Block sinks into the torii gate."""
    return WinAnimation(renderer, state)
