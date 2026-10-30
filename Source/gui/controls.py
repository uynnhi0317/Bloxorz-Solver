"""controls.py - Input (phím) và các widget UI nhỏ (Button, Menu) dùng chung cho gui/app.py."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

from game import Action
from .theme import CREAM, SAKURA_DEEP, SAKURA_LIGHT, SERIF, UI, flower_sprite, sakura_points

COLORS = UI

# Phím -> hành động trong game (mũi tên hoặc WASD; Space đổi cube khi block đang tách)
KEY_ACTIONS = {
    pygame.K_UP: Action.MOVE_UP, pygame.K_w: Action.MOVE_UP,
    pygame.K_DOWN: Action.MOVE_DOWN, pygame.K_s: Action.MOVE_DOWN,
    pygame.K_LEFT: Action.MOVE_LEFT, pygame.K_a: Action.MOVE_LEFT,
    pygame.K_RIGHT: Action.MOVE_RIGHT, pygame.K_d: Action.MOVE_RIGHT,
}
CONFIRM_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)

_FONTS: Dict[Tuple[int, bool], pygame.font.Font] = {}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    """Font có hỗ trợ tiếng Việt (nếu máy có), cache theo (size, bold)."""
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = pygame.font.SysFont(SERIF, size, bold=bold)
    return _FONTS[key]


def draw_text(surface: pygame.Surface, text: str, size: int, color, bold: bool = False,
              **anchor) -> pygame.Rect:
    """Vẽ text; anchor là kwargs của Rect, vd center=(x, y), midtop=(x, y), topleft=(x, y)."""
    img = font(size, bold).render(text, True, color)
    rect = img.get_rect(**anchor)
    surface.blit(img, rect)
    return rect


_CAPS: Dict[tuple, pygame.Surface] = {}


def caps_surface(text: str, size: int, color, bold: bool = False) -> pygame.Surface:
    """Chữ small caps (chữ thường thành HOA nhỏ hơn) cho tiêu đề và menu."""
    key = (text, size, tuple(color), bold)
    if key not in _CAPS:
        big, small = font(size, bold), font(max(8, int(size * 0.78)), bold)
        parts = []
        for ch in text:
            f = small if ch.islower() else big
            parts.append((f, f.render(ch.upper() if ch != " " else " ", True, color)))
        w = sum(img.get_width() + 1 for _, img in parts)
        surf = pygame.Surface((max(1, w), big.get_height()), pygame.SRCALPHA)
        x = 0
        for f, img in parts:
            surf.blit(img, (x, big.get_ascent() - f.get_ascent()))
            x += img.get_width() + 1
        _CAPS[key] = surf
    return _CAPS[key]


_GLOW: Dict[tuple, pygame.Surface] = {}


def draw_title(surface: pygame.Surface, text: str, size: int, color=COLORS["text"],
               glow_color=CREAM, **anchor) -> pygame.Rect:
    """Tiêu đề small caps có quầng sáng (làm mờ bằng scale xuống rồi lên)."""
    key = (text, size, tuple(color), tuple(glow_color))
    if key not in _GLOW:
        base = caps_surface(text, size, color)
        pad = size // 2
        canvas = pygame.Surface((base.get_width() + 2 * pad, base.get_height() + 2 * pad), pygame.SRCALPHA)
        canvas.blit(base, (pad, pad))
        w, h = canvas.get_size()
        glow = pygame.transform.smoothscale(pygame.transform.smoothscale(canvas, (max(1, w // 7), max(1, h // 7))), (w, h))
        glow.fill((255, 255, 255, 0), special_flags=pygame.BLEND_RGBA_ADD)   # chỉ giữ hình dạng, không giữ màu chữ
        glow.fill((*glow_color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        out = pygame.Surface((w, h), pygame.SRCALPHA)
        out.blit(glow, (0, 0))
        out.blit(glow, (0, 0))
        out.blit(canvas, (0, 0))
        _GLOW[key] = out
    img = _GLOW[key]
    rect = img.get_rect(**anchor)
    surface.blit(img, rect)
    return rect


def draw_caps(surface: pygame.Surface, text: str, size: int, color, **anchor) -> pygame.Rect:
    img = caps_surface(text, size, color)
    rect = img.get_rect(**anchor)
    surface.blit(img, rect)
    return rect


_ORN: Dict[tuple, Tuple[pygame.Surface, int]] = {}
_ORN_H, _ORN_LINE, _ORN_PAD = 34, 17, 8   # chiều cao, y của đường kẻ, lề cho quầng sáng


def _ornament(width: int, color) -> Tuple[pygame.Surface, int]:
    """Dải trang trí: dây đỏ mờ dần ra hai mép, giữa là bông anh đào kẹp hai nụ nhỏ.
    Vẽ ở độ phân giải gấp 3 rồi thu nhỏ để mượt. Trả về (surface, y của đường kẻ trong surface)."""
    key = (width, tuple(color))
    if key not in _ORN:
        S, h = 3, _ORN_H
        W, H, cy, cx = width * S, h * S, _ORN_LINE * S, width * S // 2
        layer = pygame.Surface((W, H), pygame.SRCALPHA)
        # đường kẻ mờ dần về hai mép, ngắt quãng quanh phần trang trí giữa
        xs = np.arange(W)
        alpha = np.clip(np.minimum(xs, W - 1 - xs) / (W * 0.40), 0, 1) ** 0.8
        alpha[np.abs(xs - cx) < 34 * S] = 0
        rgb = pygame.surfarray.pixels3d(layer)
        rgb[...] = COLORS["cord"]
        del rgb
        a = pygame.surfarray.pixels_alpha(layer)
        a[:, cy - S:cy + S] = (alpha * 255).astype(np.uint8)[:, None]
        del a
        for i in range(5):
            pygame.draw.polygon(layer, (*color, 255), sakura_points(cx, cy, 14 * S, i))
        pygame.draw.circle(layer, (*SAKURA_DEEP, 255), (cx, cy), 3 * S)
        for off in (-27, 27):
            pygame.draw.circle(layer, (*color, 255), (cx + off * S, cy), int(2.6 * S))
        base = pygame.transform.smoothscale(layer, (width, h))
        pad = _ORN_PAD
        canvas = pygame.Surface((width + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
        canvas.blit(base, (pad, pad))
        cw, ch = canvas.get_size()
        glow = pygame.transform.smoothscale(pygame.transform.smoothscale(canvas, (max(1, cw // 4), max(1, ch // 4))), (cw, ch))
        glow.fill((*SAKURA_DEEP, 255), special_flags=pygame.BLEND_RGBA_MULT)
        out = pygame.Surface((cw, ch), pygame.SRCALPHA)
        out.blit(glow, (0, 0))
        out.blit(canvas, (0, 0))
        _ORN[key] = (out, _ORN_LINE + pad)
    return _ORN[key]


def draw_divider(surface: pygame.Surface, center_x: int, line_y: int, width: int, color=SAKURA_LIGHT,
                 flip: bool = False) -> None:
    """Vẽ dải trang trí tại đường kẻ y=line_y. flip=True: lật dọc (dùng cho phía dưới tiêu đề)."""
    img, ly = _ornament(width, color)
    if flip:
        img, ly = pygame.transform.flip(img, False, True), img.get_height() - ly
    surface.blit(img, (center_x - img.get_width() // 2, line_y - ly))


# ------------------------------------------------------------------ thẻ gỗ Ema
_EMA: Dict[tuple, pygame.Surface] = {}
CORD_W = 2


def ema_board(w: int, h: int, fill, edge, accent=None, seed: int = 3) -> pygame.Surface:
    """Thẻ Ema: tấm gỗ có đỉnh vát như mái nhà, vân gỗ ngang, viền tối, lỗ treo và dải màu nhận diện."""
    key = (w, h, tuple(fill), tuple(edge), accent, seed)
    if key not in _EMA:
        S = 3
        W, H = w * S, h * S
        roof = min(H * 0.14, W * 0.05)   # vát nhẹ hai góc trên: vẫn là thẻ Ema nhưng vuông vức
        shape = [(0, roof), (roof, 0), (W - 1 - roof, 0), (W - 1, roof), (W - 1, H - 1), (0, H - 1)]
        mask = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), shape)
        wood = pygame.Surface((W, H), pygame.SRCALPHA)
        wood.fill((*fill, 255))
        rng = random.Random(seed + w * 7 + h)
        for _ in range(max(4, H // (6 * S))):   # vân gỗ
            y = rng.randrange(H)
            shade = rng.uniform(0.78, 1.12)
            col = tuple(min(255, int(v * shade)) for v in fill)
            pts = [(x, y + 2 * S * math.sin(x / W * rng.uniform(4, 9) + y)) for x in range(0, W + 30, 30)]
            pygame.draw.lines(wood, col, False, pts, max(1, S))
        if accent is not None:                  # dải màu dưới đáy thẻ để phân biệt thuật toán
            band = max(3 * S, int(H * 0.13))
            pygame.draw.rect(wood, accent, (0, H - band, W, band))
        wood.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        pygame.draw.polygon(wood, edge, shape, max(2, int(1.4 * S)))
        hole = (W // 2, int(roof * 0.5) + 2 * S)
        pygame.draw.circle(wood, (34, 22, 16), hole, max(2, int(1.6 * S)))
        _EMA[key] = pygame.transform.smoothscale(wood, (w, h))
    return _EMA[key]


def draw_ema(surface: pygame.Surface, rect: pygame.Rect, fill, edge, cord: int = 8, accent=None) -> None:
    """Vẽ dây đỏ treo và tấm thẻ trong rect. cord: độ dài dây phía trên thẻ."""
    if cord > 0:
        cx = rect.centerx
        pygame.draw.line(surface, COLORS["cord"], (cx, rect.top - cord), (cx, rect.top + max(2, rect.h // 7)), CORD_W)
    surface.blit(ema_board(rect.w, rect.h, fill, edge, accent), rect)


def _fit_size(text: str, base: int, max_w: int) -> int:
    size = base
    while size > 10 and font(size).size(text)[0] > max_w:
        size -= 1
    return size


def _status_icon(surface: pygame.Surface, status: str, x: int, y: int, r: int) -> None:
    """Biểu tượng trạng thái solver: vòng quay (đang chạy), dấu X (thất bại)."""
    if status == "running":
        spin = pygame.time.get_ticks() / 140
        for i in range(6):
            a = spin + i * math.pi / 3
            lvl = 0.35 + 0.65 * i / 5
            col = tuple(int(v * lvl) for v in COLORS["text"])
            pygame.draw.circle(surface, col, (x + r * math.cos(a), y + r * math.sin(a)), max(1, r // 4))
    elif status == "failed":
        pygame.draw.circle(surface, (56, 20, 22), (x, y), r + 2)
        d = r * 0.55
        pygame.draw.line(surface, COLORS["fail"], (x - d, y - d), (x + d, y + d), 2)
        pygame.draw.line(surface, COLORS["fail"], (x - d, y + d), (x + d, y - d), 2)


class Button:
    """Nút bấm dạng thẻ Ema treo dây đỏ. rect được caller gán lại mỗi frame (để layout theo kích thước cửa sổ).
    accent: dải màu nhận diện. status: None | "running" | "done" | "failed" (cho nút solver)."""

    def __init__(self, label: str, icon: Optional[str] = None, enabled: bool = True, accent=None, cord: int = 8):
        self.label, self.icon, self.enabled = label, icon, enabled
        self.accent, self.cord = accent, cord
        self.status: Optional[str] = None
        self.rect = pygame.Rect(0, 0, 0, 0)

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def draw(self, surface: pygame.Surface) -> None:
        hover = self.enabled and self.hit(pygame.mouse.get_pos())
        pressed = hover and pygame.mouse.get_pressed()[0]
        if not self.enabled:
            fill, fg = COLORS["wood_off"], COLORS["text_off"]
        else:
            fill, fg = (COLORS["wood_press"] if pressed else COLORS["wood_hover"] if hover else COLORS["wood"]), COLORS["text"]
        edge = COLORS["wood_edge"]
        if self.status == "done":
            edge = COLORS["ok"]
        elif self.status == "failed":
            edge = COLORS["fail"]
        elif self.status == "running":
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 160)
            edge = tuple(int(a + (b - a) * pulse) for a, b in zip(COLORS["wood_edge"], COLORS["accent"]))
        rect = self.rect.move(0, 2 if pressed else -1 if hover else 0)   # bấm: thẻ tụt xuống; hover: nhích lên
        draw_ema(surface, rect, fill, edge, self.cord + rect.y - self.rect.y, self.accent)
        if self.icon == "pause":  # biểu tượng menu trong stage: hai thanh dọc
            for dx in (-5, 3):
                pygame.draw.rect(surface, fg, (rect.centerx + dx, rect.centery - 6, 4, 14), border_radius=1)
            return
        pad = 22 if self.status else 8
        size = _fit_size(self.label, max(12, int(self.rect.h * 0.44)), self.rect.w - 2 * pad)
        band = (max(3, int(rect.h * 0.13)) if self.accent is not None else 0)
        cy = rect.centery + rect.h // 16 - band // 2
        draw_text(surface, self.label, size, fg, center=(rect.centerx, cy))
        if self.status:
            r = max(4, int(self.rect.h * 0.15))
            _status_icon(surface, self.status, rect.right - r - 6, rect.y + rect.h // 3, r)


class Menu:
    """Menu dọc dạng các thẻ Ema nối nhau bằng dây đỏ: điều khiển bằng chuột (hover/click)
    hoặc phím (lên/xuống + Enter/Space). handle() trả về nhãn của mục được chọn, hoặc None."""

    def __init__(self, items: List[str], size: int = 30, width: int = 300, gap: int = 15):
        self.size, self.width, self.gap = size, width, gap
        self.set_items(items)

    def set_items(self, items: List[str]) -> None:
        self.items, self.selected, self.rects = list(items), 0, []

    def height(self, scale: float = 1.0) -> int:
        h = int((self.size + 22) * scale)
        return len(self.items) * h + max(0, len(self.items) - 1) * int(self.gap * scale)

    def draw(self, surface: pygame.Surface, center_x: int, top: int, scale: float = 1.0) -> None:
        """Mục đang chọn sáng lên + hoa anh đào hai bên."""
        self.rects = []
        h, gap = int((self.size + 22) * scale), int(self.gap * scale)
        w = int(self.width * scale)
        size = max(12, int(self.size * scale))
        for i, label in enumerate(self.items):
            r = pygame.Rect(0, 0, w, h)
            r.midtop = (center_x, top + i * (h + gap))
            self.rects.append(r)
            sel = i == self.selected
            pressed = sel and pygame.mouse.get_pressed()[0] and r.collidepoint(pygame.mouse.get_pos())
            rect = r.move(0, 2 if pressed else 0)
            fill = COLORS["wood_press"] if pressed else COLORS["wood_hover"] if sel else COLORS["wood"]
            draw_ema(surface, rect, fill, COLORS["accent"] if sel else COLORS["wood_edge"],
                     gap + 2 if i else gap + 6)
            img = caps_surface(label, size, COLORS["text"] if sel else COLORS["muted"])
            surface.blit(img, img.get_rect(center=(rect.centerx, rect.centery + h // 16)))
            if sel:
                fl = flower_sprite(max(5, int(10 * scale)))
                for x in (rect.left - int(22 * scale), rect.right + int(22 * scale)):
                    surface.blit(fl, fl.get_rect(center=(x, rect.centery)))

    def handle(self, event: pygame.event.Event) -> Optional[str]:
        if not self.items:
            return None
        if event.type == pygame.MOUSEMOTION:
            for i, r in enumerate(self.rects):
                if r.collidepoint(event.pos):
                    self.selected = i
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, r in enumerate(self.rects):
                if r.collidepoint(event.pos):
                    self.selected = i
                    return self.items[i]
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.selected = (self.selected - 1) % len(self.items)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.selected = (self.selected + 1) % len(self.items)
            elif event.key in CONFIRM_KEYS:
                return self.items[self.selected]
        return None
