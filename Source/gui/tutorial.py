"""tutorial.py - Scene Tutorial: nhiều trang (chữ + ảnh minh hoạ), nội dung đọc từ assets/tutorial/tutorial.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame

from .controls import COLORS, Button, draw_divider, draw_text, draw_title, font

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "tutorial"
JSON_PATH = ASSET_DIR / "tutorial.json"


def _wrap(text: str, size: int, max_w: int) -> List[str]:
    f = font(size)
    lines: List[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and f.size(trial)[0] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


class Tutorial:
    def __init__(self, path: Path = JSON_PATH) -> None:
        self.title = "Tutorial"
        self.pages: List[dict] = []
        self.error = ""
        self.index = 0
        self.btn_skip = Button("Skip tutorial")
        self.btn_prev = Button("< Prev")
        self.btn_next = Button("Next >")
        self.last_label = "Finish"
        self._imgs: Dict[tuple, Optional[pygame.Surface]] = {}
        self._layouts: Dict[tuple, tuple] = {}
        self._dim: Optional[pygame.Surface] = None
        self._load(path)

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.title = str(data.get("title", "Tutorial"))
            for p in data["pages"]:
                text = p.get("text", [])
                paras = [text] if isinstance(text, str) else [str(t) for t in text]
                self.pages.append({"text": paras, "image": p.get("image")})
            if not self.pages:
                raise ValueError("pages[] rỗng")
        except Exception as exc:  # file thiếu/hỏng: không crash app
            self.pages, self.error = [], f"Không nạp được tutorial: {exc}"

    def open(self, mode: str = "menu") -> None:
        """mode: "newgame" | "menu" | "play" - quyết định nhãn nút."""
        self.index = 0
        self.btn_skip.label = {"newgame": "Skip tutorial", "play": "Back to game"}.get(mode, "Back to menu")
        self.last_label = "Start game" if mode == "newgame" else "Finish"
        
    # ------------------------------------------------------------------ event
    def _go(self, d: int) -> None:
        self.index = max(0, min(len(self.pages) - 1, self.index + d))

    def _advance(self) -> Optional[str]:
        if self.index == len(self.pages) - 1:
            return "exit"
        self._go(1)
        return None

    def handle(self, e: pygame.event.Event) -> Optional[str]:
        """Trả về "exit" khi cần đóng scene."""
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                return "exit"
            if e.key in (pygame.K_LEFT, pygame.K_a):
                self._go(-1)
            elif e.key in (pygame.K_RIGHT, pygame.K_d, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return self._advance()
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.btn_skip.hit(e.pos):
                return "exit"
            if self.btn_prev.enabled and self.btn_prev.hit(e.pos):
                self._go(-1)
            elif self.btn_next.hit(e.pos):
                return self._advance()
        return None

    # ------------------------------------------------------------------ assets
    def _image(self, name: str, box: Tuple[int, int]) -> Optional[pygame.Surface]:
        key = (name, box)
        if key not in self._imgs:
            if len(self._imgs) > 32:
                self._imgs.clear()
            img, path = None, ASSET_DIR / name
            if path.is_file():
                try:
                    raw = pygame.image.load(str(path)).convert_alpha()
                    k = min(1.0, box[0] / raw.get_width(), box[1] / raw.get_height())
                    img = pygame.transform.smoothscale(
                        raw, (max(1, int(raw.get_width() * k)), max(1, int(raw.get_height() * k))))
                except pygame.error:
                    img = None
            self._imgs[key] = img
        return self._imgs[key]

    def _layout(self, paras: List[str], width: int, height: int, base: int):
        """Chọn cỡ chữ lớn nhất (<= base) để toàn bộ đoạn vừa chiều cao cho phép."""
        key = (self.index, width, height, base)
        if key not in self._layouts:
            if len(self._layouts) > 32:
                self._layouts.clear()
            for size in range(base, 13, -2):
                blocks = [_wrap(p, size, width) for p in paras]
                lh = font(size).get_linesize()
                gap = lh // 2
                total = sum(len(b) * lh for b in blocks) + gap * max(0, len(blocks) - 1)
                if total <= height:
                    break
            self._layouts[key] = (size, blocks, lh, gap)
        return self._layouts[key]

    # ------------------------------------------------------------------ draw
    def draw(self, surf: pygame.Surface) -> None:
        w, h = surf.get_size()
        s = max(0.6, min(1.5, min(w / 1536, h / 864)))
        if self._dim is None or self._dim.get_size() != (w, h):
            self._dim = pygame.Surface((w, h), pygame.SRCALPHA)
            self._dim.fill((*COLORS["dim"], 200))
        surf.blit(self._dim, (0, 0))

        n, page = len(self.pages), self.pages[self.index]
        mx, gap = int(90 * s), int(50 * s)
        draw_divider(surf, w // 2, int(60 * s), int(620 * s) // 20 * 20)
        draw_title(surf, self.title, max(20, int(64 * s)), center=(w // 2, int(130 * s)))

        top, bottom = int(220 * s), h - int(130 * s)
        box = (int(w * 0.32), bottom - top)
        img = self._image(page["image"], box) if page["image"] else None
        text_w = w - 2 * mx - ((box[0] + gap) if img else 0)

        draw_text(surf, f"{self.index + 1}/{n}", max(14, int(28 * s)), COLORS["accent"],
                  bold=True, topleft=(mx, top))
        y = top + int(56 * s)
        size, blocks, lh, par_gap = self._layout(page["text"], text_w, bottom - y, max(14, int(26 * s)))
        for block in blocks:
            for line in block:
                draw_text(surf, line, size, COLORS["text"], topleft=(mx, y))
                y += lh
            y += par_gap

        if img:
            r = img.get_rect(midright=(w - mx, (top + bottom) // 2))
            surf.blit(img, r)
            pygame.draw.rect(surf, COLORS["border"], r.inflate(8, 8), 3)

        bw, bh = int(170 * s), max(30, int(42 * s))
        by = h - int(50 * s) - bh
        self.btn_skip.rect = pygame.Rect(mx, by, int(190 * s), bh)
        self.btn_next.rect = pygame.Rect(w - mx - bw, by, bw, bh)
        self.btn_prev.rect = pygame.Rect(w - mx - 2 * bw - int(16 * s), by, bw, bh)
        self.btn_prev.enabled = self.index > 0
        self.btn_next.label = self.last_label if self.index == n - 1 else "Next >"
        for b in (self.btn_skip, self.btn_prev, self.btn_next):
            b.draw(surf)