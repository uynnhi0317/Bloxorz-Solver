"""app.py - Vòng lặp chính: menu game, chọn màn, credits, chơi màn (HUD, pause, thắng/thua).

Cách T14 gắn solver vào:
    app.register_solver("BFS", fn)   # fn(board, state) -> Optional[List[Move]]
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

from game import Board, Move, State, Action, get_next_state, is_goal, load_stage
from .animations import (Animation, BridgeFlips, Sequence as AnimSequence, animate_fall, animate_intro, animate_move,
                         animate_outro, animate_win, is_bump, DIR_DELTA)
from .controls import COLORS, KEY_ACTIONS, Button, Menu, draw_caps, draw_divider, draw_text, draw_title
from .theme import Background
from .renderer import Renderer
from solvers import SolveResult

SOURCE_DIR = Path(__file__).resolve().parent.parent
STAGE_GROUPS = ("simple", "fragile", "switch", "combined")  # thứ tự tăng dần độ khó
PAGE_COLS, PAGE_ROWS = 4, 2
PAGE_SIZE = PAGE_COLS * PAGE_ROWS
HUD_H = 56

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
        self.selected: Optional[Tuple[int, int]] = None  # cube đang điều khiển khi block tách
        self.renderer.selected = None
        self.moves = 0
        self.status = "playing"  # playing | won | lost
        self.anim: Optional[Animation] = None
        self.pending: Optional[Tuple[State, str]] = None
        self.replay: Deque[Move] = deque()
        self.end_fx = None  # hiệu ứng tile còn lại sau animation cuối (vd tile fragile đã vỡ)
        self.flips.clear()
        self.renderer.prepare(self.state.bridge_mask)  # dựng sẵn ảnh sàn khử răng cưa

    def start_intro(self) -> None:
        """Vào màn / restart: tile bay lên, khối rơi xuống; input bị khóa tới khi xong."""
        self.anim, self.pending = animate_intro(self.renderer, self.state), (self.state, "playing")

    def toggle_cube(self) -> None:
        """Đổi cube đang điều khiển (phím Space). Chỉ đổi trạng thái UI, không tính bước."""
        if self.anim is not None or self.status != "playing" or self.state.split_positions is None:
            return
        a, b = self.state.split_positions
        self.selected = b if self.selected == a else a
        self.renderer.selected = self.selected

    def _follow_selection(self, move: Move, nxt: State) -> None:
        """Cập nhật cube đang chọn sau một nước đi hợp lệ"""
        if nxt.split_positions is None:  # đã ghép lại
            self.selected = None
        elif self.state.split_positions is None:  # vừa tách: mặc định chọn cube đầu
            self.selected = nxt.split_positions[0]
        else:  # cube đang chọn đi theo vị trí mới
            dr, dc = DIR_DELTA[move.action]
            if move.cube is not None:
                self.selected = (move.cube[0] + dr, move.cube[1] + dc)
        self.renderer.selected = self.selected

    def try_action(self, act: Action | Move) -> bool:
        """Thực hiện một hành động; trả về False nếu bị bỏ qua (đang animation, đã kết thúc, bị chặn)."""
        if self.anim is not None or self.status != "playing":
            return False
        split = self.state.split_positions is not None
        if split and self.selected not in self.state.split_positions: # luôn có một cube được chọn
            self.selected = self.renderer.selected = self.state.split_positions[0]
        move = act if isinstance(act, Move) else Move(act, self.selected if split else None)
        if split and move.cube is not None:  # replay: tự đổi cube theo lời giải
            self.selected = self.renderer.selected = move.cube
        nxt = get_next_state(self.board, self.state, move)
        if nxt is None:
            if is_bump(self.state, move):  # cube đâm vào cube kia: bị chặn, không phải thua
                return False
            moving = self.state.split_positions.index(move.cube) if split and move.cube is not None else None
            self.anim = animate_fall(self.renderer, self.state, move.action, moving=moving)
            self.pending = (self.state, "lost")
            return True
        self.moves += 1
        move_anim = animate_move(self.renderer, self.state, move, nxt)
        self._follow_selection(move, nxt)
        if is_goal(self.board, nxt):
            self.anim = AnimSequence(move_anim, animate_win(self.renderer, nxt), animate_outro(self.renderer, nxt))
            self.pending = (nxt, "won")
        else:
            self.anim, self.pending = move_anim, (nxt, "playing")
        return True

    def start_replay(self, moves: List[Move]) -> None:
        self.replay = deque(moves)

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
                    self.replay.clear()
        if self.anim is None and self.status == "playing" and self.replay:
            if not self.try_action(self.replay.popleft()):
                self.replay.clear()

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
        self.solve_buttons = {name: Button(name, enabled=False) for name in ("BFS", "DFS", "UCS", "A*")}
        self.hud_buttons = [self.btn_restart, *self.solve_buttons.values()]

        self.solving: Optional[str] = None
        self._solve_out = None  # (name, start_state, result, error)
        self.last_stats = None  # (name, result)

    # ------------------------------------------------------------------ API cho T14
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
        self.session.start_intro()
        self.select_index = index
        self.scene, self.paused, self._end_status = "play", False, "playing"

    def restart(self) -> None:
        if self.session:
            self.session.reset()
            self.session.start_intro()
            self.paused, self._end_status = False, "playing"

    def run_solver(self, name: str) -> None:
        s = self.session
        if s is None or s.status != "playing" or s.anim is not None or s.replay or self.solving:
            return
        fn = self.solvers.get(name)
        if fn is None:
            self.toast(f"Solver {name} chưa được tích hợp")
            return
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
            self.toast(f"{name} lỗi: {err}", 4000)
            return
        self.last_stats = (name, res)
        if res.moves is None:
            self.toast(f"{name}: không tìm thấy lời giải")
        else:
            s.start_replay(list(res.moves))

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
            elif e.key == pygame.K_SPACE and not s.replay:  # đổi cube đang điều khiển
                s.toggle_cube()
            elif e.key in KEY_ACTIONS and not s.replay and not self.solving:  # đang replay lời giải thì khóa input
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
        if self.last_stats:
            name, r = self.last_stats
            txt = (f"{name} | {r.time * 1000:.1f} ms | {r.peak_memory / 1024:.0f} KiB | "
                f"{r.expanded_nodes} nodes | moves: {r.solution_length}")
            draw_text(surf, txt, 25, COLORS["text"], bottomleft=(16, surf.get_height() - 12))

    def _draw_overlay(self, surf, title, color, subtitle, menu: Menu) -> None:
        """Overlay như Paused, Win, Lose"""
        w, h = surf.get_size()
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((10, 12, 18, 175))
        surf.blit(dim, (0, 0))
        glow = tuple((c + 130) // 2 for c in color)  # quầng sáng hơi ngả theo màu tiêu đề
        draw_divider(surf, w // 2, h // 2 - 190, 340)
        draw_title(surf, title, 56, color=color, glow_color=glow, center=(w // 2, h // 2 - 130))
        if subtitle:
            draw_caps(surf, subtitle, 22, COLORS["muted"], center=(w // 2, h // 2 - 78))
        menu.draw(surf, w // 2, h // 2 - 20)

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