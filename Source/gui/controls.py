"""controls.py - Input (phím) và các widget UI nhỏ (Button, Menu) dùng chung cho gui/app.py."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pygame

from game import Action

COLORS = {
    "bg": (24, 26, 34), "bar": (32, 35, 46), "panel": (44, 48, 62), "border": (70, 76, 96),
    "text": (230, 232, 240), "muted": (120, 128, 146), "accent": (86, 164, 236),
    "accent_dark": (44, 84, 124), "win": (110, 200, 120), "lose": (226, 96, 96),
}

# Phím -> hành động trong game (mũi tên hoặc WASD; Space đổi cube khi block đang tách)
KEY_ACTIONS = {
    pygame.K_UP: Action.MOVE_UP, pygame.K_w: Action.MOVE_UP,
    pygame.K_DOWN: Action.MOVE_DOWN, pygame.K_s: Action.MOVE_DOWN,
    pygame.K_LEFT: Action.MOVE_LEFT, pygame.K_a: Action.MOVE_LEFT,
    pygame.K_RIGHT: Action.MOVE_RIGHT, pygame.K_d: Action.MOVE_RIGHT,
    pygame.K_SPACE: Action.SWITCH_CONTROL,
}
CONFIRM_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)

_FONTS: Dict[Tuple[int, bool], pygame.font.Font] = {}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    """Font được cache theo (size, bold)."""
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = pygame.font.SysFont("segoeui,arial,dejavusans,liberationsans,helvetica,freesans",
                                          size, bold=bold)
    return _FONTS[key]


def draw_text(surface: pygame.Surface, text: str, size: int, color, bold: bool = False,
              **anchor) -> pygame.Rect:
    """Vẽ text; anchor là kwargs của Rect, vd center=(x, y), midtop=(x, y), topleft=(x, y)."""
    img = font(size, bold).render(text, True, color)
    rect = img.get_rect(**anchor)
    surface.blit(img, rect)
    return rect


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
        if self.icon == "pause":
            r = self.rect
            for dx in (-5, 3):
                pygame.draw.rect(surface, fg, (r.centerx + dx, r.centery - 8, 4, 16), border_radius=1)
        else:
            draw_text(surface, self.label, 18, fg, center=self.rect.center)


class Menu:
    """Menu dọc: điều khiển bằng chuột (hover/click) hoặc phím (lên/xuống + Enter/Space).
    handle() trả về nhãn của mục được chọn, hoặc None."""

    def __init__(self, items: List[str], size: int = 30, width: int = 300, gap: int = 12):
        self.size, self.width, self.gap = size, width, gap
        self.set_items(items)

    def set_items(self, items: List[str]) -> None:
        self.items, self.selected, self.rects = list(items), 0, []

    def draw(self, surface: pygame.Surface, center_x: int, top: int) -> None:
        self.rects = []
        h = self.size + 18
        for i, label in enumerate(self.items):
            r = pygame.Rect(0, 0, self.width, h)
            r.midtop = (center_x, top + i * (h + self.gap))
            self.rects.append(r)
            sel = i == self.selected
            pygame.draw.rect(surface, COLORS["accent_dark"] if sel else COLORS["panel"], r, border_radius=10)
            pygame.draw.rect(surface, COLORS["accent"] if sel else COLORS["border"], r, 2, border_radius=10)
            draw_text(surface, label, self.size, COLORS["text"] if sel else COLORS["muted"], center=r.center)

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