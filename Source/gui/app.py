"""app.py - Vòng lặp chính: menu game, chọn màn, credits, chơi màn (HUD, pause, thắng/thua).
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import pygame

from game import Action, Board, State, get_next_state, is_goal, load_stage
from .animations import Animation, Sequence as AnimSequence, animate_fall, animate_move, animate_win, is_bump
from .controls import COLORS, KEY_ACTIONS, Button, Menu, draw_text
from .renderer import Renderer

SOURCE_DIR = Path(__file__).resolve().parent.parent
STAGE_GROUPS = ("simple", "fragile", "switch", "combined")  # thứ tự tăng dần độ khó
PAGE_COLS, PAGE_ROWS = 4, 2
PAGE_SIZE = PAGE_COLS * PAGE_ROWS
HUD_H = 56

# TODO: thay bằng tên + vai trò thật của các thành viên
CREDITS: List[Tuple[str, str]] = [
    ("Thành viên 1", "Vai trò A"),
    ("Thành viên 2", "Vai trò B"),
    ("Thành viên 3", "Vai trò C"),
    ("Thành viên 4", "Vai trò D"),
    ("Thành viên 5", "Vai trò E"),
]

Solver = Callable[[Board, State], Optional[List[Action]]]


def discover_stages(root: Path = SOURCE_DIR) -> List[Path]:
    """Gom stage theo thứ tự nhóm rồi theo tên file (NOTE: cần đặt tên 01_..., 02_... để sắp đúng thứ tự).
    Nếu chưa có stage nào thì dùng tạm tests/data/mock_data.json."""
    found: List[Path] = []
    for group in STAGE_GROUPS:
        found += sorted((root / "stages" / group).glob("*.json"))
    if not found:
        mock = root / "tests" / "data" / "mock_data.json"
        if mock.is_file():
            found.append(mock)
    return found


class PlaySession:
    """Một lượt chơi một màn: state hiện tại, số bước, trạng thái thắng/thua, animation đang chạy."""

    def __init__(self, board: Board, renderer: Renderer, index: int):
        self.board, self.renderer, self.index = board, renderer, index
        self.reset()

    def reset(self) -> None:
        self.state: State = self.board.initial_state
        self.moves = 0
        self.status = "playing"  # playing | won | lost
        self.anim: Optional[Animation] = None
        self.pending: Optional[Tuple[State, str]] = None
        self.replay: Deque[Action] = deque()

    def try_action(self, action: Action) -> bool:
        """Thực hiện một hành động; trả về False nếu bị bỏ qua (đang animation, đã kết thúc, bị chặn)."""
        if self.anim is not None or self.status != "playing":
            return False
        nxt = get_next_state(self.board, self.state, action)
        if action == Action.SWITCH_CONTROL:  # chỉ đổi cube active, không tính bước
            if nxt is None:
                return False
            self.state = nxt
            return True
        if nxt is None:
            if is_bump(self.state, action):  # cube đâm vào cube kia: bị chặn, không phải thua
                return False
            self.anim = animate_fall(self.renderer, self.state, action)
            self.pending = (self.state, "lost")
            return True
        self.moves += 1
        move = animate_move(self.renderer, self.state, action, nxt)
        if is_goal(self.board, nxt):
            self.anim, self.pending = AnimSequence(move, animate_win(self.renderer, nxt)), (nxt, "won")
        else:
            self.anim, self.pending = move, (nxt, "playing")
        return True

    def start_replay(self, actions: List[Action]) -> None:
        self.replay = deque(actions)

    def update(self, dt_ms: float) -> None:
        if self.anim is not None:
            self.anim.update(dt_ms)
            if self.anim.done:
                (self.state, self.status), self.anim, self.pending = self.pending, None, None
                if self.status != "playing":
                    self.replay.clear()
        if self.anim is None and self.status == "playing" and self.replay:
            if not self.try_action(self.replay.popleft()):
                self.replay.clear()

    def draw(self, surface: pygame.Surface) -> None:
        if self.anim is not None:
            self.anim.draw(surface)
        elif self.status == "playing":
            self.renderer.draw(surface, self.state)
        else:  # thắng/thua: block đã rơi, chỉ còn sàn
            self.renderer.draw_scene(surface, self.state.bridge_mask, [])


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Bloxorz Solver")
        self.screen = pygame.display.set_mode((960, 640), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.stages = discover_stages()
        self.scene = "menu"  # menu | select | credits | play
        self.main_menu = Menu(["New Game", "Load Stage", "Credits", "Quit"])
        self.pause_menu = Menu(["Resume", "Quit to Menu"])
        self.end_menu = Menu([])
        self._end_status = "playing"
        self.paused = False
        self.session: Optional[PlaySession] = None
        self.select_index = 0
        self.credits_y = 0.0
        self.toast_text, self.toast_ms = "", 0.0
        self._thumbs: Dict[Tuple[int, Tuple[int, int]], Optional[pygame.Surface]] = {}
        self._size = self.screen.get_size()

        self.solvers: Dict[str, Solver] = {}
        self.btn_pause = Button("", icon="pause")
        self.btn_restart = Button("Restart")
        self.solve_buttons = {name: Button(name, enabled=False) for name in ("BFS", "DFS", "UCS", "A*")}
        self.hud_buttons = [self.btn_restart, *self.solve_buttons.values()]

    # ------------------------------------------------------------------ API cho solver
    def register_solver(self, name: str, fn: Solver) -> None:
        self.solvers[name] = fn
        self.solve_buttons[name].enabled = True

    # ------------------------------------------------------------------ điều hướng
    def toast(self, text: str, ms: float = 2200) -> None:
        self.toast_text, self.toast_ms = text, ms

    def board_area(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(0, HUD_H, w, h - HUD_H)

    def start_stage(self, index: int) -> None:
        try:
            board = load_stage(self.stages[index])
        except Exception as exc:  # file stage hỏng: không crash cả app
            self.toast(f"Không nạp được stage {index + 1}: {exc}", 4000)
            self.scene = "menu"
            return
        self.session = PlaySession(board, Renderer(board, self.board_area()), index)
        self.select_index = index
        self.scene, self.paused, self._end_status = "play", False, "playing"

    def restart(self) -> None:
        if self.session:
            self.session.reset()
            self.paused, self._end_status = False, "playing"

    def run_solver(self, name: str) -> None:
        s = self.session
        if s is None or s.status != "playing" or s.anim is not None or s.replay:
            return
        fn = self.solvers.get(name)
        if fn is None:
            self.toast(f"Solver {name} chưa được tích hợp")
            return
        actions = fn(s.board, s.state)
        if actions is None:
            self.toast(f"{name}: không tìm thấy lời giải")
        else:
            s.start_replay(actions)

    # ------------------------------------------------------------------ event
    def handle_event(self, e: pygame.event.Event) -> None:
        if e.type == pygame.QUIT:
            self.running = False
        elif self.scene == "menu":
            self._event_menu(e)
        elif self.scene == "select":
            self._event_select(e)
        elif self.scene == "credits":
            if e.type == pygame.MOUSEBUTTONDOWN or (e.type == pygame.KEYDOWN and e.key in (
                    pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_BACKSPACE)):
                self.scene = "menu"
        elif self.scene == "play":
            self._event_play(e)

    def _event_menu(self, e) -> None:
        choice = self.main_menu.handle(e)
        if choice == "New Game":
            if self.stages:
                self.start_stage(0)
            else:
                self.toast("Chưa có stage nào trong stages/")
        elif choice == "Load Stage":
            self.scene = "select"
        elif choice == "Credits":
            self.scene, self.credits_y = "credits", float(self.screen.get_height())
        elif choice == "Quit":
            self.running = False

    def _event_select(self, e) -> None:
        n = len(self.stages)
        rects = self._select_rects()
        left, right = self._arrow_rects()
        if e.type == pygame.KEYDOWN:
            step = {pygame.K_LEFT: -1, pygame.K_RIGHT: 1, pygame.K_UP: -PAGE_COLS, pygame.K_DOWN: PAGE_COLS}.get(e.key)
            if step and n:
                self.select_index = max(0, min(n - 1, self.select_index + step))
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and n:
                self.start_stage(self.select_index)
            elif e.key == pygame.K_ESCAPE:
                self.scene = "menu"
        elif e.type == pygame.MOUSEMOTION:
            page = self.select_index // PAGE_SIZE
            for slot, r in enumerate(rects):
                if r.collidepoint(e.pos) and page * PAGE_SIZE + slot < n:
                    self.select_index = page * PAGE_SIZE + slot
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            page, pages = self.select_index // PAGE_SIZE, max(1, math.ceil(n / PAGE_SIZE))
            if left.collidepoint(e.pos) and page > 0:
                self.select_index = (page - 1) * PAGE_SIZE
            elif right.collidepoint(e.pos) and page < pages - 1:
                self.select_index = (page + 1) * PAGE_SIZE
            else:
                for slot, r in enumerate(rects):
                    if r.collidepoint(e.pos) and page * PAGE_SIZE + slot < n:
                        self.start_stage(page * PAGE_SIZE + slot)

    def _event_play(self, e) -> None:
        s = self.session
        if self.paused:
            choice = self.pause_menu.handle(e)
            if choice == "Resume" or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                self.paused = False
            elif choice == "Quit to Menu":
                self.scene = "menu"
            return
        if s.status != "playing":  # overlay thắng/thua
            choice = self.end_menu.handle(e)
            if choice == "Next Stage":
                self.start_stage(s.index + 1)
            elif choice in ("Restart", "Retry"):
                self.restart()
            elif choice == "Quit to Menu":
                self.scene = "menu"
            return
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                self.paused = True
            elif e.key == pygame.K_r:
                self.restart()
            elif e.key in KEY_ACTIONS and not s.replay:  # đang replay lời giải thì khóa input
                s.try_action(KEY_ACTIONS[e.key])
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.btn_pause.hit(e.pos):
                self.paused = True
            elif self.btn_restart.hit(e.pos):
                self.restart()
            else:
                for name, b in self.solve_buttons.items():
                    if b.hit(e.pos):
                        self.run_solver(name)

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        if self.screen.get_size() != self._size:  # cửa sổ đổi kích thước
            self._size = self.screen.get_size()
            if self.session:
                self.session.renderer.set_area(self.board_area())
        self.toast_ms = max(0.0, self.toast_ms - dt)
        if self.scene == "credits":
            self.credits_y -= 45 * dt / 1000
            if self.credits_y < -(len(CREDITS) * 70 + 260):
                self.credits_y = float(self.screen.get_height())
        if self.scene != "play" or self.paused:
            return
        s = self.session
        s.update(dt)
        if s.status != self._end_status:
            self._end_status = s.status
            if s.status == "won":
                has_next = s.index + 1 < len(self.stages)
                self.end_menu.set_items((["Next Stage"] if has_next else []) + ["Restart", "Quit to Menu"])
            elif s.status == "lost":
                self.end_menu.set_items(["Retry", "Quit to Menu"])

    # ------------------------------------------------------------------ draw
    def draw(self) -> None:
        surf = self.screen
        surf.fill(COLORS["bg"])
        w, h = surf.get_size()
        if self.scene == "menu":
            draw_text(surf, "BLOXORZ", 84, COLORS["accent"], bold=True, center=(w // 2, h // 2 - 190))
            draw_text(surf, "SOLVER", 40, COLORS["text"], bold=True, center=(w // 2, h // 2 - 130))
            self.main_menu.draw(surf, w // 2, h // 2 - 70)
        elif self.scene == "select":
            self._draw_select(surf)
        elif self.scene == "credits":
            self._draw_credits(surf)
        elif self.scene == "play":
            self.session.draw(surf)
            self._draw_hud(surf)
            if self.paused:
                self._draw_overlay(surf, "Paused", COLORS["text"], "", self.pause_menu)
            elif self.session.status == "won":
                last = self.session.index + 1 >= len(self.stages)
                sub = f"Moves: {self.session.moves}" + ("  -  All stages cleared!" if last else "")
                self._draw_overlay(surf, "YOU WIN!", COLORS["win"], sub, self.end_menu)
            elif self.session.status == "lost":
                self._draw_overlay(surf, "YOU LOSE", COLORS["lose"], f"Moves: {self.session.moves}", self.end_menu)
        if self.toast_ms > 0:
            r = draw_text(surf, self.toast_text, 20, COLORS["text"], midbottom=(w // 2, h - 24))
            pygame.draw.rect(surf, COLORS["panel"], r.inflate(28, 16), border_radius=8)
            draw_text(surf, self.toast_text, 20, COLORS["text"], center=r.center)

    def _draw_hud(self, surf: pygame.Surface) -> None:
        w = surf.get_width()
        pygame.draw.rect(surf, COLORS["bar"], (0, 0, w, HUD_H))
        pygame.draw.line(surf, COLORS["border"], (0, HUD_H), (w, HUD_H))
        self.btn_pause.rect = pygame.Rect(12, 10, 36, 36)
        x = w - 12
        for b in reversed(self.hud_buttons):
            width = 92 if b is self.btn_restart else 64
            x -= width
            b.rect = pygame.Rect(x, 10, width, 36)
            x -= 8
        for b in (self.btn_pause, *self.hud_buttons):
            b.draw(surf)
        s = self.session
        r = draw_text(surf, f"Stage {s.index + 1}", 22, COLORS["text"], bold=True, midleft=(66, HUD_H // 2))
        draw_text(surf, f"Moves: {s.moves}", 20, COLORS["muted"], midleft=(r.right + 24, HUD_H // 2))

    def _draw_overlay(self, surf, title, color, subtitle, menu: Menu) -> None:
        w, h = surf.get_size()
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((10, 12, 18, 175))
        surf.blit(dim, (0, 0))
        draw_text(surf, title, 56, color, bold=True, center=(w // 2, h // 2 - 130))
        if subtitle:
            draw_text(surf, subtitle, 22, COLORS["muted"], center=(w // 2, h // 2 - 80))
        menu.draw(surf, w // 2, h // 2 - 40)

    def _draw_credits(self, surf: pygame.Surface) -> None:
        w, h = surf.get_size()
        y = self.credits_y
        draw_text(surf, "Bloxorz Solver", 48, COLORS["accent"], bold=True, midtop=(w // 2, y))
        draw_text(surf, "CSC14003 - Introduction to Artificial Intelligence", 20, COLORS["muted"],
                  midtop=(w // 2, y + 64))
        y += 150
        for name, role in CREDITS:
            draw_text(surf, name, 28, COLORS["text"], bold=True, midtop=(w // 2, y))
            draw_text(surf, role, 20, COLORS["muted"], midtop=(w // 2, y + 36))
            y += 70
        draw_text(surf, "Esc: back", 18, COLORS["muted"], bottomright=(w - 16, h - 12))

    # --- chọn màn
    def _select_rects(self) -> List[pygame.Rect]:
        w, h = self.screen.get_size()
        mx, top, bottom, gap = 90, 110, h - 70, 20
        cw, ch = (w - 2 * mx - (PAGE_COLS - 1) * gap) / PAGE_COLS, (bottom - top - (PAGE_ROWS - 1) * gap) / PAGE_ROWS
        return [pygame.Rect(mx + (i % PAGE_COLS) * (cw + gap), top + (i // PAGE_COLS) * (ch + gap), cw, ch)
                for i in range(PAGE_SIZE)]

    def _arrow_rects(self) -> Tuple[pygame.Rect, pygame.Rect]:
        w, h = self.screen.get_size()
        mid = (110 + h - 70) // 2
        return pygame.Rect(18, mid - 40, 54, 80), pygame.Rect(w - 72, mid - 40, 54, 80)

    def _thumb(self, idx: int, size: Tuple[int, int]) -> Optional[pygame.Surface]:
        """Thumbnail gui/assets/thumbs/stage_01.png (nếu có); chưa có thì trả None."""
        key = (idx, size)
        if key not in self._thumbs:
            path, img = SOURCE_DIR / "gui" / "assets" / "thumbs" / f"stage_{idx + 1:02d}.png", None
            if path.is_file():
                try:
                    img = pygame.transform.smoothscale(pygame.image.load(str(path)).convert(), size)
                except pygame.error:
                    img = None
            self._thumbs[key] = img
        return self._thumbs[key]

    def _draw_select(self, surf: pygame.Surface) -> None:
        w, h = surf.get_size()
        n = len(self.stages)
        page, pages = self.select_index // PAGE_SIZE, max(1, math.ceil(n / PAGE_SIZE))
        draw_text(surf, "Select Stage", 40, COLORS["text"], bold=True, center=(w // 2, 56))
        for slot, r in enumerate(self._select_rects()):
            idx = page * PAGE_SIZE + slot
            if idx >= n:  # ô trống giữ nguyên bố cục 2x4
                pygame.draw.rect(surf, (34, 37, 48), r, 1, border_radius=10)
                continue
            sel = idx == self.select_index
            pygame.draw.rect(surf, COLORS["accent_dark"] if sel else COLORS["panel"], r, border_radius=10)
            thumb = self._thumb(idx, (max(1, r.w - 12), max(1, r.h - 52)))
            if thumb:
                surf.blit(thumb, (r.x + 6, r.y + 6))
                draw_text(surf, f"Stage {idx + 1}", 22, COLORS["accent"] if sel else COLORS["muted"],
                          bold=sel, midbottom=(r.centerx, r.bottom - 10))
            else:
                draw_text(surf, f"Stage {idx + 1}", 26, COLORS["accent"] if sel else COLORS["muted"],
                          bold=sel, center=r.center)
            pygame.draw.rect(surf, COLORS["accent"] if sel else COLORS["border"], r, 2, border_radius=10)
        for rect, glyph, ok in zip(self._arrow_rects(), ("left", "right"), (page > 0, page < pages - 1)):
            col = COLORS["accent"] if ok else (50, 54, 68)
            cx, cy, d = rect.centerx, rect.centery, 1 if glyph == "right" else -1
            pygame.draw.lines(surf, col, False, [(cx - 12 * d, cy - 26), (cx + 12 * d, cy), (cx - 12 * d, cy + 26)], 6)
        draw_text(surf, f"Page {page + 1}/{pages}", 20, COLORS["muted"], midbottom=(w // 2, h - 40))
        draw_text(surf, "Arrow keys: move   Enter: play   Esc: back", 16, COLORS["muted"],
                  midbottom=(w // 2, h - 14))

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(60)
            self.screen = pygame.display.get_surface()
            for e in pygame.event.get():
                self.handle_event(e)
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()