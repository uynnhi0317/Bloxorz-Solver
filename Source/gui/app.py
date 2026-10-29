"""app.py - Vòng lặp chính: menu game, chọn màn, credits, chơi màn (HUD, pause, thắng/thua).

Cách T14 gắn solver vào:
    app.register_solver("BFS", fn)   # fn(board, state) -> Optional[List[Action]]
Nút Solve tương ứng tự bật; kết quả được replay bằng animation. Số liệu (time, memory...)
do T14 tự hiển thị / lưu.
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple
import threading

import pygame

from game import Action, Board, State, get_next_state, is_goal, load_stage
from .animations import (Animation, BridgeFlips, Sequence as AnimSequence, animate_fall, animate_intro, animate_move,
                         animate_outro, animate_win, is_bump)
from .controls import COLORS, KEY_ACTIONS, Button, Menu, draw_caps, draw_divider, draw_text, draw_title
from .theme import Background
from .renderer import Renderer
from solvers import SolveResult

SOURCE_DIR = Path(__file__).resolve().parent.parent
STAGE_GROUPS = ("simple", "fragile", "switch", "combined")  # thứ tự tăng dần độ khó
PAGE_COLS, PAGE_ROWS = 4, 2
PAGE_SIZE = PAGE_COLS * PAGE_ROWS
HUD_H = 56
PANEL_BASE_W = 340          # bề rộng panel ở hệ số s = 1
PANEL_MARGIN = 12

CREDITS: List[Tuple[str, Tuple[str, str]]] = [
    ("Trần Tuấn Kiệt", ("Lead Systems Programmer · Core Architecture & Stage",
                      "Gameplay Programmer · Data Visualization")),
    ("Trần Ngô Uyên Nhi", ("Gameplay Programmer · Tile, Bridge & Switch Systems",
                      "Level Designer · Performance & QA Engineer")),
    ("Dương Ngọc Minh Thư", ("AI Programmer · Pathfinding & Search",
                      "Gameplay QA · Heuristic & Cost Design")),
    ("Hoàng Diễm Phương", ("Graphics Programmer · Rendering & Animation",
                      "UI Programmer · Tools & Benchmark Engineer")),
]

Solver = Callable[[Board, State], "SolveResult"]  # có .moves/.time/.peak_memory/.expanded_nodes


def discover_stages(root: Path = SOURCE_DIR) -> List[Path]:
    """Gom stage theo thứ tự nhóm rồi theo tên file (đặt tên 01_..., 02_... để sắp đúng thứ tự).
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
        self.flips = BridgeFlips(renderer)  # cầu lật chạy nền, không chặn input
        self.reset()

    def reset(self) -> None:
        self.state: State = self.board.initial_state
        self.moves = 0
        self.status = "playing"  # playing | won | lost
        self.anim: Optional[Animation] = None
        self.pending: Optional[Tuple[State, str]] = None
        self.replay: Deque[Action] = self.clear_solution()
        self.end_fx = None  # hiệu ứng tile còn lại sau animation cuối (vd tile fragile đã vỡ)
        self.flips.clear()
        self.renderer.prepare(self.state.bridge_mask)  # dựng sẵn ảnh sàn khử răng cưa

    def start_intro(self) -> None:
        """Vào màn / restart: tile bay lên, khối rơi xuống; input bị khóa tới khi xong."""
        self.anim, self.pending = animate_intro(self.renderer, self.state), (self.state, "playing")

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
            self.anim = AnimSequence(move, animate_win(self.renderer, nxt), animate_outro(self.renderer, nxt))
            self.pending = (nxt, "won")
        else:
            self.anim, self.pending = move, (nxt, "playing")
        return True

    def clear_solution(self) -> None:
        self.sol_actions: Optional[List[Action]] = None
        self.sol_states: List[State] = []   # sol_states[i] = state after i steps (0 = start)
        self.sol_moves: List[int] = []      # move counter after i steps
        self.sol_idx = 0
        self.sol_paused = False

    @property
    def has_solution(self) -> bool:
        return self.sol_actions is not None

    @property
    def replaying(self) -> bool:
        """Replay is actively running (not paused, steps left) -> player input is locked."""
        return self.has_solution and not self.sol_paused and self.sol_idx < len(self.sol_actions)

    def start_replay(self, actions: List[Action]) -> None:
        states, moves = [self.state], [self.moves]
        for a in actions:
            nxt = get_next_state(self.board, states[-1], a)
            if nxt is None:
                break
            states.append(nxt)
            moves.append(moves[-1] + int(a != Action.SWITCH_CONTROL))
        self.clear_solution()
        if len(states) > 1:
            self.sol_actions = list(actions[:len(states) - 1])
            self.sol_states, self.sol_moves = states, moves

    def toggle_replay_pause(self) -> None:
        self.sol_paused = not self.sol_paused

    def step_forward(self) -> bool:
        if (self.anim is not None or self.status != "playing" or not self.has_solution
                or self.sol_idx >= len(self.sol_actions)):
            return False
        if self.try_action(self.sol_actions[self.sol_idx]):
            self.sol_idx += 1
            return True
        self.clear_solution()  # solution no longer valid
        return False

    def step_back(self) -> bool:
        if (self.anim is not None or self.status != "playing" or not self.has_solution
                or self.sol_idx == 0):
            return False
        old = self.state
        self.sol_idx -= 1
        self.state, self.moves = self.sol_states[self.sol_idx], self.sol_moves[self.sol_idx]
        if self.state.bridge_mask != old.bridge_mask:  # same handling as in update()
            self.renderer.prepare(self.state.bridge_mask)
            self.flips.start(old.bridge_mask, self.state.bridge_mask)
        return True

    def manual_action(self, action: Action) -> bool:
        """Player input: if it goes through, we've left the solver's path."""
        if self.try_action(action):
            self.sol_actions = None
            return True
        return False
    
    def update(self, dt_ms: float) -> None:
        self.flips.update(dt_ms)
        if self.anim is not None:
            self.anim.update(dt_ms)
            if self.anim.done:
                self.end_fx = self.anim.end_fx
                new_state, self.status = self.pending
                if new_state.bridge_mask != self.state.bridge_mask:  # công tắc vừa đổi cầu -> lật nền
                    self.renderer.prepare(new_state.bridge_mask)
                    self.flips.start(self.state.bridge_mask, new_state.bridge_mask)
                self.state, self.anim, self.pending = new_state, None, None
                if self.status != "playing":
                    self.clear_solution()
        if self.anim is None and self.status == "playing" and self.replaying:
            self.step_forward()

    def draw(self, surface: pygame.Surface) -> None:
        self.renderer.ambient_fx = self.flips.fx()
        if self.anim is not None:
            self.anim.draw(surface)
        elif self.status == "playing":
            self.renderer.draw(surface, self.state)
        else:  # thắng/thua: block đã rơi, chỉ còn sàn
            self.renderer.draw_scene(surface, self.state.bridge_mask, [], tile_fx=self.end_fx)


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Bloxorz Solver")
        info = pygame.display.Info()

        # Set rescale = 1 to make fullscreen (instead of pygame.FULLSCREEN, since it might cause lag (?) ), the default is 0.8
        width, height, rescale = info.current_w, info.current_h, 0.8
        self.screen = pygame.display.set_mode((width*rescale, height*rescale), pygame.RESIZABLE)

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
        self.background = Background(self._size)

        self.solvers: Dict[str, Solver] = {}
        self.btn_pause = Button("", icon="pause")
        self.btn_restart = Button("Restart")
        self.btn_replay = Button("Pause")
        self.btn_prev = Button("< Prev")
        self.btn_next = Button("Next >")
        self.replay_buttons = (self.btn_replay, self.btn_prev, self.btn_next)
        self.solve_buttons = {name: Button(name, enabled=False) for name in ("BFS", "DFS", "UCS", "A*")}
        self.hud_buttons = [self.btn_restart, *self.solve_buttons.values()]

        self.solving: Optional[str] = None
        self._solve_out = None          # (name, start_state, result, error)
        self.last_stats = None          # (name, result)
        self._gen = 0
        self.panel_on = False

    #-------------------------------------------------------------------
    def _set_panel(self, on: bool) -> None:
        """Mở/đóng cột metrics; board được dựng lại cho vùng mới."""
        if self.panel_on == on:
            return
        self.panel_on = on
        if self.session:
            self.session.renderer.set_area(self.board_area())
            self.session.renderer.prepare(self.session.state.bridge_mask)  # dựng sẵn, đỡ khựng

    def _panel_scale(self) -> float:
        """Hệ số theo cửa sổ: =1 ở ~1536x864 (80% màn 1920x1080), kẹp trong [0.6, 1.5]."""
        w, h = self.screen.get_size()
        return max(0.6, min(1.5, min(w / 1536, h / 864)))

    def _panel_w(self) -> int:
        return int(PANEL_BASE_W * self._panel_scale())

    def _reset_solver_ui(self) -> None:
        """Xoá số liệu, bỏ kết quả solver đang chạy dở."""
        self._gen += 1
        self.solving = None
        self._solve_out = None
        self.last_stats = None
        self.toast_ms = 0.0
        self._set_panel(False)

    # ------------------------------------------------------------------ API cho T14
    def register_solver(self, name: str, fn: Solver) -> None:
        self.solvers[name] = fn
        self.solve_buttons[name].enabled = True

    # ------------------------------------------------------------------ điều hướng
    def toast(self, text: str, ms: float = 2200) -> None:
        self.toast_text, self.toast_ms = text, ms

    def board_area(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        side = self._panel_w() + 2 * PANEL_MARGIN if self.panel_on else 0
        return pygame.Rect(0, HUD_H, max(1, w - side), h - HUD_H)

    def start_stage(self, index: int) -> None:
        self._reset_solver_ui()
        try:
            board = load_stage(self.stages[index])
        except Exception as exc:  # file stage hỏng: không crash cả app
            self.toast(f"Không nạp được stage {index + 1}: {exc}", 4000)
            self.scene = "menu"
            return
        self.session = PlaySession(board, Renderer(board, self.board_area()), index)
        self.session.start_intro()
        self.select_index = index
        self.scene, self.paused, self._end_status = "play", False, "playing"

    def restart(self) -> None:
        if self.session:
            self._reset_solver_ui()
            self.session.reset()
            self.session.start_intro()
            self.paused, self._end_status = False, "playing"

    def run_solver(self, name: str) -> None:
        s = self.session
        if s is None or s.status != "playing" or s.anim is not None or s.replaying or self.solving:
            return
        fn = self.solvers.get(name)
        if fn is None:
            self.toast(f"Solver {name} chưa được tích hợp")
            return
        self._set_panel(True)
        s.clear_solution()
        self.solving = name
        self.toast(f"{name}: đang giải...", 600000)
        board, start = s.board, s.state

        def work() -> None:
            try:
                self._solve_out = (name, start, fn(board, start), None)
            except Exception as exc:
                self._solve_out = (name, start, None, exc)

        threading.Thread(target=work, daemon=True).start()

    def _poll_solver(self) -> None:
        out, self._solve_out = self._solve_out, None
        if out is None:
            return
        name, start, res, err = out
        self.solving, self.toast_ms = None, 0.0
        s = self.session
        if self.scene != "play" or s is None or s.status != "playing" or s.state != start:
            return  # người chơi đã thoát/restart/đi tiếp: kết quả cũ, bỏ
        if err is not None:
            self._set_panel(False)
            self.toast(f"{name} lỗi: {err}", 4000)
            return
        self.last_stats = (name, res)
        if res.moves is None:
            self.toast(f"{name}: không tìm thấy lời giải")
        else:
            s.start_replay(list(res.moves))

    def _replay_click(self, pos) -> bool:
        s = self.session
        if not (self.panel_on and s.has_solution):
            return False
        for b, act in ((self.btn_replay, s.toggle_replay_pause),
                    (self.btn_prev, s.step_back), (self.btn_next, s.step_forward)):
            if b.enabled and b.hit(pos):
                act()
                return True
        return False

    # ------------------------------------------------------------------ event
    def handle_event(self, e: pygame.event.Event) -> None:
        """Chọn handle event phù hợp với từng scene"""
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
        """Handle event khi ở menu chính"""
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
        """Handle event khi ở menu Load Stage"""
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
        """Handle event khi đang chơi"""
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
            elif e.key in KEY_ACTIONS and not s.replaying and not self.solving:
                s.manual_action(KEY_ACTIONS[e.key])
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.btn_pause.hit(e.pos):
                self.paused = True
            elif self.btn_restart.hit(e.pos):
                self.restart()
            elif self.btn_restart.hit(e.pos):
                self.restart()
            elif self._replay_click(e.pos):
                pass
            else:
                for name, b in self.solve_buttons.items():
                    if b.hit(e.pos):
                        self.run_solver(name)

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        # cửa sổ đổi kích thước, về sau tuy game đã chỉnh thành FULLSCREEN nhưng vẫn để code ở đây phòng khi đổi ý
        if self.screen.get_size() != self._size:
            self._size = self.screen.get_size()
            self.background.resize(self._size)
            if self.session:
                self.session.renderer.set_area(self.board_area())
        self.toast_ms = max(0.0, self.toast_ms - dt)
        self.background.update(dt)
        self._poll_solver()

        if self.scene == "credits":
            self.credits_y -= dt / 10
            if self.credits_y < -(len(CREDITS) * 260 + 210):
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
        """Chọn cảnh để vẽ"""
        surf = self.screen
        self.background.draw(surf)
        w, h = surf.get_size()
        if self.scene == "menu":
            draw_divider(surf, w // 2, h // 2 - 300, 720)
            draw_title(surf, "Bloxorz", 140, center=(w // 2, h // 2 - 190))
            draw_title(surf, "Solver", 85, color=COLORS["accent"], center=(w // 2, h // 2 - 80))
            draw_divider(surf, w // 2, h // 2 - 10, 620, flip=True)
            self.main_menu.draw(surf, w // 2, h // 2 + 40)  # thấp hơn một dòng so với trước (-70)
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
                self._draw_overlay(surf, "You Win!", COLORS["win"], sub, self.end_menu)
            elif self.session.status == "lost":
                self._draw_overlay(surf, "You Lose", COLORS["lose"], f"Moves: {self.session.moves}", self.end_menu)
            self._draw_metrics(surf)
        if self.toast_ms > 0:
            r = draw_text(surf, self.toast_text, 20, COLORS["text"], midbottom=(w // 2, h - 24))
            pygame.draw.rect(surf, COLORS["panel"], r.inflate(28, 16), border_radius=8)
            draw_text(surf, self.toast_text, 20, COLORS["text"], center=r.center)

    def _draw_hud(self, surf: pygame.Surface) -> None:
        """HUD trong game"""
        w = surf.get_width()
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
        cx = self.board_area().centerx
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((10, 12, 18, 175))
        surf.blit(dim, (0, 0))
        glow = tuple((c + 130) // 2 for c in color)
        draw_divider(surf, cx, h // 2 - 190, 340)
        draw_title(surf, title, 56, color=color, glow_color=glow, center=(cx, h // 2 - 130))
        if subtitle:
            draw_caps(surf, subtitle, 22, COLORS["muted"], center=(cx, h // 2 - 78))
        menu.draw(surf, cx, h // 2 - 20)

    def _draw_credits(self, surf: pygame.Surface) -> None:
        """Vẽ credits"""
        w, h = surf.get_size()
        y = self.credits_y
        draw_divider(surf, w // 2, y, 600)
        draw_text(surf, "Bloxorz Solver", 60, COLORS["accent"], bold=True, midtop=(w // 2, y + 40))
        draw_text(surf, "CSC14003 - Introduction to Artificial Intelligence", 35, COLORS["muted"],
                  midtop=(w // 2, y + 104))
        draw_divider(surf, w // 2, y + 180, 350, flip=True)

        y += 240
        for name, (line1, line2) in CREDITS:
            draw_text(surf, name, 40, COLORS["text"], bold=True, midtop=(w // 2, y))
            draw_text(surf, line1, 25, COLORS["muted"], midtop=(w // 2, y + 70))
            draw_text(surf, line2, 25, COLORS["muted"], midtop=(w // 2, y + 130))
            y += 260
        draw_text(surf, "Esc: back", 18, COLORS["muted"], bottomright=(w - 16, h - 12))

    def _draw_metrics(self, surf: pygame.Surface) -> None:
        """Cột SEARCH METRICS bên phải, chỉ có khi đã bấm solver."""
        if not self.panel_on:
            return
        s = self._panel_scale()
        w, h = surf.get_size()
        pw = self._panel_w()
        col = pygame.Rect(w - pw - 2 * PANEL_MARGIN, HUD_H, pw + 2 * PANEL_MARGIN, h - HUD_H)

        bg = pygame.Surface(col.size, pygame.SRCALPHA)
        bg.fill((*COLORS["bar"], 235))
        surf.blit(bg, col)
        pygame.draw.line(surf, COLORS["border"], col.topleft, col.bottomleft, 1)

        if self.last_stats:
            name, r = self.last_stats
            rows = [
                ("Search time", f"{r.time * 1000:.2f} ms"),
                ("Nodes expanded", f"{r.expanded_nodes:,}"),
                ("Memory (peak)", f"{r.peak_memory / 1024:.1f} KB"),
                ("Solution length", f"{r.solution_length} moves" if r.is_solved else "no solution"),
            ]
        else:  # đang giải
            name = self.solving or ""
            rows = [(l, "...") for l in ("Search time", "Nodes expanded", "Memory (peak)", "Solution length")]

        x, y = col.x + PANEL_MARGIN, col.y + PANEL_MARGIN
        pygame.draw.rect(surf, COLORS["accent"], (x, y + int(4 * s), max(2, int(3 * s)), int(18 * s)))
        draw_text(surf, f"SEARCH METRICS - {name}", max(10, int(18 * s)), COLORS["text"], bold=True,
                midleft=(x + int(10 * s), y + int(13 * s)))
        y += int(46 * s)

        row_h, gap = int(31 * s), int(8 * s)
        for label, value in rows:
            row = pygame.Rect(x, y, pw, row_h)
            pygame.draw.rect(surf, COLORS["panel"], row, border_radius=int(6 * s))
            draw_text(surf, label, max(9, int(17 * s)), COLORS["muted"],
                    midleft=(row.x + int(10 * s), row.centery))
            draw_text(surf, value, max(10, int(18 * s)), COLORS["text"], bold=True,
                    midright=(row.right - int(10 * s), row.centery))
            y += row_h + gap
        sess = self.session
        if sess is not None and sess.has_solution:
            n = len(sess.sol_actions)
            y += int(10 * s)
            draw_text(surf, f"Step {sess.sol_idx} / {n}", max(10, int(17 * s)), COLORS["accent"],
                    bold=True, midleft=(x, y + int(9 * s)))
            y += int(24 * s)
            bar_h = max(4, int(6 * s))
            pygame.draw.rect(surf, COLORS["panel"], (x, y, pw, bar_h), border_radius=bar_h // 2)
            pygame.draw.rect(surf, COLORS["accent"], (x, y, int(pw * sess.sol_idx / n), bar_h),
                            border_radius=bar_h // 2)
            y += bar_h + int(14 * s)

            bh, g = max(28, int(36 * s)), int(8 * s)
            half = (pw - g) // 2
            paused = sess.sol_paused
            self.btn_replay.label = "Resume" if paused else "Pause"
            self.btn_replay.enabled = True
            self.btn_prev.enabled = paused and sess.sol_idx > 0
            self.btn_next.enabled = paused and sess.sol_idx < n
            self.btn_replay.rect = pygame.Rect(x, y, pw, bh)
            self.btn_prev.rect = pygame.Rect(x, y + bh + g, half, bh)
            self.btn_next.rect = pygame.Rect(x + half + g, y + bh + g, half, bh)
            for b in self.replay_buttons:
                b.draw(surf)

    # --- chọn màn / Load Stage
    def _select_rects(self) -> List[pygame.Rect]:
        """Tạo bộ khung chữ nhật cho các slot trong menu Load Stage"""
        w, h = self.screen.get_size()
        mx, top, bottom, gap = 90, 150, h - 110, 20
        cw, ch = (w - 2 * mx - (PAGE_COLS - 1) * gap) / PAGE_COLS, (bottom - top - (PAGE_ROWS - 1) * gap) / PAGE_ROWS
        return [pygame.Rect(mx + (i % PAGE_COLS) * (cw + gap), top + (i // PAGE_COLS) * (ch + gap), cw, ch)
                for i in range(PAGE_SIZE)]

    def _arrow_rects(self) -> Tuple[pygame.Rect, pygame.Rect]:
        """Tạo bộ mũi tên trong menu Load Stage"""
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
                    img = pygame.image.load(str(path)).convert()
                    k = min(size[0] / img.get_width(), size[1] / img.get_height())  # giữ tỉ lệ, không méo ảnh
                    img = pygame.transform.smoothscale(img, (max(1, int(img.get_width() * k)),
                                                             max(1, int(img.get_height() * k))))
                except pygame.error:
                    img = None
            self._thumbs[key] = img
        return self._thumbs[key]

    def _draw_select(self, surf: pygame.Surface) -> None:
        """Vẽ menu Load Stage"""
        w, h = surf.get_size()
        n = len(self.stages)
        page, pages = self.select_index // PAGE_SIZE, max(1, math.ceil(n / PAGE_SIZE))
        draw_text(surf, "Select Stage", 40, COLORS["text"], bold=True, center=(w // 2, 80))
        draw_divider(surf, w // 2, 120, 340, flip=True)

        # các slot hiện stage
        rects = self._select_rects()
        # Kích thước card chuẩn: lấy từ thumbnail đầu tiên có trên trang này
        ref = None
        for i in range(page * PAGE_SIZE, min(n, (page + 1) * PAGE_SIZE)):
            t = self._thumb(i, (max(1, rects[0].w - 12), max(1, rects[0].h - 12)))
            if t:
                ref = (max(t.get_width() + 12, 110), t.get_height() + 12)
                break

        for slot, r in enumerate(rects):
            idx = page * PAGE_SIZE + slot
            if idx >= n:  # ô trống: cùng kích thước với card thật
                empty = pygame.Rect(0, 0, *ref) if ref else r.copy()
                empty.center = r.center
                pygame.draw.rect(surf, (34, 37, 48), empty, 1, border_radius=10)
                continue
            sel = idx == self.select_index
            thumb = self._thumb(idx, (max(1, r.w - 12), max(1, r.h - 12)))
            if thumb:
                card = pygame.Rect(0, 0, max(thumb.get_width() + 6, 110), thumb.get_height() + 6)
                card.center = r.center
                surf.blit(thumb, thumb.get_rect(midtop=(card.centerx, card.y + 3)))
                draw_text(surf, f"Stage {idx + 1}", 28, COLORS["accent"] if sel else COLORS["muted"],
                          bold=sel, midbottom=(card.centerx, card.bottom + 50))
            else:
                card = r
                draw_text(surf, f"Stage {idx + 1}", 26, COLORS["accent"] if sel else COLORS["muted"],
                          bold=sel, center=card.center)
            pygame.draw.rect(surf, COLORS["accent"] if sel else COLORS["border"], card, 2, border_radius=10)

        # mũi tên chuyển page, text hướng dẫn manuver trong menu
        for rect, glyph, ok in zip(self._arrow_rects(), ("left", "right"), (page > 0, page < pages - 1)):
            col = COLORS["accent"] if ok else (50, 54, 68)
            cx, cy, d = rect.centerx, rect.centery, 1 if glyph == "right" else -1
            pygame.draw.lines(surf, col, False, [(cx - 12 * d, cy - 26), (cx + 12 * d, cy), (cx - 12 * d, cy + 26)], 6)
        draw_text(surf, f"Page {page + 1}/{pages}", 25, COLORS["muted"], midbottom=(w // 2, h - 35))
        draw_text(surf, "Arrow keys: move   Enter: play   Esc: back", 20, COLORS["muted"],
                  midbottom=(w // 2, h - 8))

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