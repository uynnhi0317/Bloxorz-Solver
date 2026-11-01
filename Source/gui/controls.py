"""controls.py - Input (phím) và các widget UI nhỏ (Button, Menu) dùng chung cho gui/app.py."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
from pathlib import Path

from game import Action
from .theme import SERIF, UI

COLORS = UI

# Phím -> hành động trong game (mũi tên hoặc WASD; Space đổi cube khi block đang tách)
KEY_ACTIONS = {
    pygame.K_UP: Action.MOVE_UP, pygame.K_w: Action.MOVE_UP,
    pygame.K_DOWN: Action.MOVE_DOWN, pygame.K_s: Action.MOVE_DOWN,
    pygame.K_LEFT: Action.MOVE_LEFT, pygame.K_a: Action.MOVE_LEFT,
    pygame.K_RIGHT: Action.MOVE_RIGHT, pygame.K_d: Action.MOVE_RIGHT,
}
CONFIRM_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)


FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
FONT_FILES = {"title": "EmilysCandy-Regular.ttf", "ui": "FuzzyBubbles-Regular.ttf"}
_FONTS: Dict[Tuple[int, bool], pygame.font.Font] = {}

def font(size: int, bold: bool = False, role: str = "ui") -> pygame.font.Font:
    key = (size, bold, role)
    if key not in _FONTS:
        path = FONT_DIR / FONT_FILES.get(role, "")
        if path.is_file():
            f = pygame.font.Font(str(path), size)
            f.set_bold(bold)  # fake bold; pass bold=False for brush fonts
        else:
            f = pygame.font.SysFont(SERIF, size, bold=bold)  # fallback if file missing
        _FONTS[key] = f
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
    """Chữ small caps (chữ thường thành HOA nhỏ hơn) như menu Hollow Knight."""
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


def draw_title(surface: pygame.Surface, text: str, size: int, color=(238, 244, 255),
               glow_color=(130, 180, 255), **anchor) -> pygame.Rect:
    """Tiêu đề small caps có quầng sáng (làm mờ bằng scale xuống rồi lên)."""
    key = (text, size, tuple(color), tuple(glow_color))
    if key not in _GLOW:
        base = caps_surface(text, size, color)
        pad = size // 2
        canvas = pygame.Surface((base.get_width() + 2 * pad, base.get_height() + 2 * pad), pygame.SRCALPHA)
        canvas.blit(base, (pad, pad))
        w, h = canvas.get_size()
        glow = pygame.transform.smoothscale(pygame.transform.smoothscale(canvas, (max(1, w // 7), max(1, h // 7))), (w, h))
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
_ORN_H, _ORN_LINE, _ORN_PAD = 34, 22, 8   # chiều cao, y của đường kẻ, lề cho quầng sáng


def _ornament(width: int, color) -> Tuple[pygame.Surface, int]:
    """Đồ trang trí kiểu ___.oOOo.___ : đường kẻ mờ dần ra hai mép, ở giữa là hai vòng tròn
    lớn kẹp hai vòng nhỏ, hai chấm và một mũi nhọn hướng lên. Vẽ ở độ phân giải gấp 3 rồi thu nhỏ
    để mượt. Trả về (surface, y của đường kẻ trong surface)."""
    key = (width, tuple(color))
    if key not in _ORN:
        S, h = 3, _ORN_H
        W, H, cy, cx = width * S, h * S, _ORN_LINE * S, width * S // 2
        layer = pygame.Surface((W, H), pygame.SRCALPHA)
        # đường kẻ mờ dần về hai mép, ngắt quãng quanh phần trang trí giữa
        xs = np.arange(W)
        alpha = np.clip(np.minimum(xs, W - 1 - xs) / (W * 0.40), 0, 1) ** 0.8
        alpha[np.abs(xs - cx) < 37 * S] = 0
        rgb = pygame.surfarray.pixels3d(layer)
        rgb[...] = color
        del rgb
        a = pygame.surfarray.pixels_alpha(layer)
        a[:, cy - S:cy + S] = (alpha * 255).astype(np.uint8)[:, None]
        del a
        col = (*color, 255)
        for off, rad in ((-20, 3), (-9, 5), (9, 5), (20, 3)):  # o O O o
            pygame.draw.circle(layer, col, (cx + off * S, cy), rad * S, int(1.6 * S))
        for off in (-31, 31):                                 # hai chấm
            pygame.draw.circle(layer, col, (cx + off * S, cy), int(1.6 * S))
        pygame.draw.polygon(layer, col, [(cx, cy - 13 * S), (cx + 4 * S, cy - 5 * S),   # mũi nhọn hướng lên
                                         (cx, cy - 1 * S), (cx - 4 * S, cy - 5 * S)])
        base = pygame.transform.smoothscale(layer, (width, h))
        pad = _ORN_PAD
        canvas = pygame.Surface((width + 2 * pad, h + 2 * pad), pygame.SRCALPHA)
        canvas.blit(base, (pad, pad))
        cw, ch = canvas.get_size()
        glow = pygame.transform.smoothscale(pygame.transform.smoothscale(canvas, (max(1, cw // 4), max(1, ch // 4))), (cw, ch))
        glow.fill((130, 180, 255, 255), special_flags=pygame.BLEND_RGBA_MULT)
        out = pygame.Surface((cw, ch), pygame.SRCALPHA)
        out.blit(glow, (0, 0))
        out.blit(canvas, (0, 0))
        _ORN[key] = (out, _ORN_LINE + pad)
    return _ORN[key]


def draw_divider(surface: pygame.Surface, center_x: int, line_y: int, width: int, color=(238, 244, 255),
                 flip: bool = False) -> None:
    """Vẽ đồ trang trí tại đường kẻ y=line_y. flip=True: lật dọc (mũi nhọn hướng xuống), dùng cho
    phía dưới tiêu đề; phía trên dùng flip=False như tiêu đề Hollow Knight."""
    img, ly = _ornament(width, color)
    if flip:
        img, ly = pygame.transform.flip(img, False, True), img.get_height() - ly
    surface.blit(img, (center_x - img.get_width() // 2, line_y - ly))


def draw_ornament(surface: pygame.Surface, x: int, y: int, d: int, color) -> None:
    """Mũi tên trang trí hai bên mục đang chọn (d=+1 hướng sang phải, -1 sang trái)."""
    for off, half in ((0, 9), (-9, 6)):
        pygame.draw.lines(surface, color, False,
                          [(x + d * (off - 8), y - half), (x + d * (off + 4), y), (x + d * (off - 8), y + half)], 2)
    pygame.draw.circle(surface, color, (x + d * 14, y), 2)


class Button:
    """Nút bấm chữ nhật. rect được caller gán lại mỗi frame (để layout theo kích thước cửa sổ)."""

    def __init__(self, label: str, icon: Optional[str] = None, enabled: bool = True):
        self.label, self.icon, self.enabled = label, icon, enabled
        self.rect = pygame.Rect(0, 0, 0, 0)

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def draw(self, surface: pygame.Surface) -> None:
        hover = self.hit(pygame.mouse.get_pos())
        if not self.enabled:
            fill, fg = (36, 39, 50), (90, 96, 112)
        else:
            fill, fg = (COLORS["accent_dark"] if hover else COLORS["panel"]), COLORS["text"]
        pygame.draw.rect(surface, fill, self.rect, border_radius=8)
        pygame.draw.rect(surface, COLORS["accent"] if hover and self.enabled else COLORS["border"],
                         self.rect, 1, border_radius=8)
        if self.icon == "pause":  # biểu tượng menu trong stage: hai thanh dọc
            r = self.rect
            for dx in (-5, 3):
                pygame.draw.rect(surface, fg, (r.centerx + dx, r.centery - 8, 4, 16), border_radius=1)
        else:
            draw_text(surface, self.label, 18, fg, center=(self.rect.centerx, self.rect.centery + 3))


class Menu:
    """Menu dọc: điều khiển bằng chuột (hover/click) hoặc phím (lên/xuống + Enter/Space).
    handle() trả về nhãn của mục được chọn, hoặc None."""

    def __init__(self, items: List[str], size: int = 35, width: int = 300, gap: int = 15):
        self.size, self.width, self.gap = size, width, gap
        self.set_items(items)

    def set_items(self, items: List[str]) -> None:
        self.items, self.selected, self.rects = list(items), 0, []

    def draw(self, surface: pygame.Surface, center_x: int, top: int, scale: float = 1.0) -> None:
        """Không khung, chữ small caps; mục đang chọn sáng + mũi tên hai bên."""
        self.rects = []
        size = max(10, int(self.size * scale))
        width = int(self.width * scale)
        gap = int(self.gap * scale)
        h = size + int(18 * scale)
        for i, label in enumerate(self.items):
            r = pygame.Rect(0, 0, width, h)
            r.midtop = (center_x, top + i * (h + gap))
            self.rects.append(r)
            sel = i == self.selected
            col = COLORS["text"] if sel else COLORS["muted"]
            img = caps_surface(label, size, col)
            rect = img.get_rect(center=r.center)
            surface.blit(img, rect)
            if sel:
                off = int(34 * scale)
                draw_ornament(surface, rect.left - off, r.centery, -1, col)
                draw_ornament(surface, rect.right + off, r.centery, 1, col)

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