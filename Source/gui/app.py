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

from game import Action, Board, Coord, Move, State, get_next_state, is_goal, load_stage
from .animations import (Animation, BridgeFlips, DIR_DELTA, Sequence as AnimSequence, animate_fall,
                         animate_intro, animate_move, animate_outro, animate_win, is_bump, DropAnimation, RollAnimation)
from .controls import COLORS, KEY_ACTIONS, Button, Menu, draw_caps, draw_divider, draw_text, draw_title
from .theme import Background
from .renderer import Renderer
from solvers import SolveResult
from .tutorial import Tutorial

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
        self.selected: Optional[Coord] = None
        self.renderer.selected = None
        self.moves = 0
        self.status = "playing"  # playing | won | lost
        self.anim: Optional[Animation] = None
        self.pending: Optional[Tuple[State, str]] = None
        self.replay: Deque[Action] = self.clear_solution()
        self.end_fx = None  # hiệu ứng tile còn lại sau animation cuối (vd tile fragile đã vỡ)
        self.flips.clear()
        self.renderer.prepare(self.state.bridge_mask)  # dựng sẵn ảnh sàn khử răng cưa
        self.by_solver = False  # flag, is the finishing move by solver?
        

    def start_intro(self) -> None:
        """Vào màn / restart: tile bay lên, khối rơi xuống; input bị khóa tới khi xong."""
        self.anim, self.pending = animate_intro(self.renderer, self.state), (self.state, "playing")

    def try_action(self, move: Move, by_solver: bool = False) -> bool:
        if self.anim is not None or self.status != "playing":
            return False
        self.by_solver = by_solver
        nxt = get_next_state(self.board, self.state, move)
        if nxt is None:
            if is_bump(self.state, move):    # cube hit its twin: blocked, not a loss
                return False
            self.anim = animate_fall(self.renderer, self.state, move.action,
                                    self._cube_index(move.cube))
            self.pending = (self.state, "lost")
            return True
        self.moves += 1
        roll = animate_move(self.renderer, self.state, move, nxt)
        self._sync_selected(nxt, move)       # AFTER building the animation, which captures colours
        if is_goal(self.board, nxt):
            parts = [roll, animate_win(self.renderer, nxt)]
            if not by_solver:
                parts.append(animate_outro(self.renderer, nxt))
            self.anim = AnimSequence(*parts)
            self.pending = (nxt, "won")
        else:
            self.anim, self.pending = roll, (nxt, "playing")
        return True

    def _set_selected(self, cell: Optional[Coord]) -> None:
        self.selected = cell
        self.renderer.selected = cell

    def _active_cube(self) -> Optional[Coord]:
        sp = self.state.split_positions
        if sp is None:
            return None
        return self.selected if self.selected in sp else sp[0]

    def _cube_index(self, cube: Optional[Coord]) -> int:
        sp = self.state.split_positions
        return sp.index(cube) if sp is not None and cube in sp else 0

    def toggle_cube(self) -> None:
        """UI only: swap the controlled cube. No move, no state change."""
        if self.anim is None and self.status == "playing" and self.state.split_positions is not None:
            a, b = self.state.split_positions
            self._set_selected(b if self._active_cube() == a else a)

    def _sync_selected(self, nxt: State, move: Move) -> None:
        if nxt.split_positions is None:      # whole block, or just merged
            self._set_selected(None)
        elif move.cube is None:              # just split on a switch
            self._set_selected(nxt.split_positions[0])
        else:                                # follow the cube that moved
            dr, dc = DIR_DELTA[move.action]
            self._set_selected((move.cube[0] + dr, move.cube[1] + dc))

    def clear_solution(self) -> None:
        self.sol_actions: Optional[List[Action]] = None
        self.sol_states: List[State] = []   # sol_states[i] = state after i steps (0 = start)
        self.sol_moves: List[int] = []      # move counter after i steps
        self.sol_idx = 0
        self.sol_paused = False

    @property
    def winning_drop(self) -> bool:
        """Đang ở pha rơi vào goal của bước cuối (đã lăn xong)."""
        return (self.pending is not None and self.pending[1] == "won"
                and isinstance(self.anim, AnimSequence) and bool(self.anim.anims)
                and isinstance(self.anim.anims[0], DropAnimation))

    @property
    def finished(self) -> bool:
        """Solver đã đi hết, và không còn pha lăn nào: lúc này nút là Replay."""
        return (self.has_solution and self.sol_idx >= len(self.sol_actions)
                and (self.anim is None or self.winning_drop))
    
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
            moves.append(moves[-1] + 1)
        self.clear_solution()
        if len(states) > 1:
            self.sol_actions = list(actions[:len(states) - 1])
            self.sol_states, self.sol_moves = states, moves

    def toggle_replay_pause(self) -> None:
        if not self.has_solution:
            return
        if self.finished:
            self.anim, self.pending, self.end_fx = None, None, None
            self.status = "playing"
            self.sol_idx = 0
            self.state = self.sol_states[0]
            self.moves = self.sol_moves[0]
            self.flips.clear()
            self.renderer.prepare(self.state.bridge_mask)
            self._set_selected(None)
            self.sol_paused = False
            return
        self.sol_paused = not self.sol_paused

    def step_forward(self) -> bool:
        if (self.anim is not None or self.status != "playing" or not self.has_solution
                or self.sol_idx >= len(self.sol_actions)):
            return False
        if self.try_action(self.sol_actions[self.sol_idx], by_solver=True):
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
        mv = self.sol_actions[self.sol_idx]      # the move we just undid
        self._set_selected(mv.cube if self.state.split_positions is not None else None)
        if self.state.bridge_mask != old.bridge_mask:  # same handling as in update()
            self.renderer.prepare(self.state.bridge_mask)
            self.flips.start(old.bridge_mask, self.state.bridge_mask)
        return True

    def manual_action(self, action: Action) -> bool:
        """Player input: if it goes through, we've left the solver's path."""
        if self.try_action(Move(action=action, cube=self._active_cube())):
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
                if self.status != "playing" and not self.by_solver:
                    self.clear_solution()
        if self.anim is None and self.status == "playing" and self.replaying:
            self.step_forward()

    def _winning(self) -> bool:
        """Khối đang rơi vào goal (hoặc đã thắng): goal sáng mạnh hơn."""
        if self.status == "won":
            return True
        a = self.anim
        return (isinstance(a, AnimSequence) and self.pending is not None and self.pending[1] == "won"
                and bool(a.anims) and not isinstance(a.anims[0], RollAnimation))

    def draw(self, surface: pygame.Surface) -> None:
        self.renderer.goal_boost = 1.0 if (self.status == "won" or self.winning_drop) else 0.0
        self.renderer.ambient_fx = self.flips.fx()
        if self.anim is not None:
            self.anim.draw(surface)
        elif self.status == "playing":
            if self.has_solution and self.sol_idx >= len(self.sol_actions):
                self.renderer.draw_scene(surface, self.state.bridge_mask, [], tile_fx=self.end_fx)
            else:
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
        self.main_menu = Menu(["New Game", "Load Stage", "Tutorial", "Credits", "Quit"])       
        self.pause_menu = Menu(["Resume", "Tutorial", "Quit to Menu"])
        self.tutorial_return = "menu"      # "newgame" | "menu" | "play"
        self.tutorial_seen = False         # auto-show only once per launch
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
        self.tutorial = Tutorial() 

        self.solvers: Dict[str, Solver] = {}
        self.btn_pause = Button("", icon="pause")
        self.btn_restart = Button("Restart")
        self.btn_replay = Button("Pause")
        self.btn_prev = Button("< Prev")
        self.btn_next = Button("Next >")
        self.replay_buttons = (self.btn_replay, self.btn_prev, self.btn_next)
        self.solve_buttons = {name: Button(name, enabled=False) for name in ("BFS", "DFS", "UCS", "A*")}
        self.btn_stage_prev = Button("< Stage")
        self.btn_stage_next = Button("Stage >")
        self.btn_solver = Button("Solver")
        self.hud_buttons = [self.btn_stage_prev, self.btn_stage_next, self.btn_restart, self.btn_solver]


        
        self.solve_done = False                     # last stage was finished by a solver
        self.solve_start: Optional[State] = None    # state the last solver run started from

        self.solving: Optional[str] = None
        self._solve_out = None          # (name, start_state, result, error)
        self.last_stats = None          # (name, result)
        self._gen = 0
        self.panel_on = False
        self.last_label = "Finish"

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

    def _menu_scale(self) -> float:
        """Hệ số scale cho menu chính (=1 ở ~1536x864). Làm tròn theo bước 0.05 để
        cache chữ/glow/ornament không phình ra khi kéo resize cửa sổ."""
        w, h = self.screen.get_size()
        s = max(0.4, min(1.5, min(w / 1536, h / 864)))
        return round(s * 20) / 20

    def _panel_w(self) -> int:
        return int(PANEL_BASE_W * self._panel_scale())

    def _reset_solver_ui(self) -> None:
        self._gen += 1
        self.solving = None
        self._solve_out = None
        self.last_stats = None
        self.solve_done = False
        self.solve_start = None
        self.toast_ms = 0.0

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
            self.toast(f"Could not load stage {index + 1}: {exc}", 4000)
            self.scene = "menu"
            return
        self.session = PlaySession(board, Renderer(board, self.board_area()), index)
        self.session.start_intro()
        self.select_index = index
        self.scene, self.paused, self._end_status = "play", False, "playing"

    def restart(self, keep_solver_ui: bool = False) -> None:
        if self.session:
            if not keep_solver_ui:
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
            self.toast(f"Solver {name} has not been integrated")
            return
        s.clear_solution()
        self.last_stats = None        # otherwise the old numbers stay on screen while solving
        self.solve_done = False
        self.solve_start = s.state
        self.solving = name
        self.toast(f"{name}: solving...", 600000)
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
        s = self.session
        in_game = self.scene == "play" or (self.scene == "tutorial" and self.tutorial_return == "play")
        if not in_game or s is None or s.status != "playing" or s.state != start:
            return
        if err is not None:
            self.toast(f"{name} error: {err}", 4000)
            return
        self.last_stats = (name, res)
        if res.moves is None:
            self.toast(f"{name}: could not find a solution")
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

    def _on_game_over(self) -> None:
        """Player thua: thử lại. Solver thắng: giữ số liệu, chạy lại intro của stage.
        Player thắng: tự động next stage, nếu là final stage thì có overlay 'You Win!'."""
        s = self.session
        index, moves, by_solver = s.index, s.moves, s.by_solver
        if s.status == "lost":
            self.restart()
            self.toast("You fell, try again!")
        elif by_solver:
            s.status = "playing"
            s.sol_paused = True
            self.solve_done = True
        elif index + 1 < len(self.stages):
            self.start_stage(index + 1)
            self.toast(f"You won in {index + 1} ({moves} moves)")
        else:
            self.end_menu.set_items(["Restart", "Quit to Menu"])  # final stage: overlay stays

    def _open_tutorial(self, mode: str) -> bool:
        if not self.tutorial.pages:
            return False
        self.tutorial_return = mode
        self.tutorial.open(mode)
        self.scene = "tutorial"
        return True

    def _close_tutorial(self) -> None:
        mode = self.tutorial_return
        if mode == "newgame":
            self.start_stage(0)
        elif mode == "play":
            self.scene = "play"          # self.paused stays True -> back to the pause menu
        else:
            self.scene = "menu"

    def _input_locked(self) -> bool:
        s = self.session
        return bool(self.solving) or s.replaying or (self.panel_on and s.has_solution)

    # ------------------------------------------------------------------ event
    def handle_event(self, e: pygame.event.Event) -> None:
        """Chọn handle event phù hợp với từng scene"""
        if e.type == pygame.QUIT:
            self.running = False
        elif self.scene == "menu":
            self._event_menu(e)
        elif self.scene == "tutorial":
            if self.tutorial.handle(e) == "exit":
                self._close_tutorial()
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
            if not self.stages:
                self.toast("No stage in folder stages/")
            elif not self.tutorial_seen and self._open_tutorial("newgame"):
                self.tutorial_seen = True
            else:
                self.start_stage(0)
        elif choice == "Tutorial":
            if not self._open_tutorial("menu"):
                self.toast(self.tutorial.error, 4000)
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
            elif choice == "Tutorial":
                if not self._open_tutorial("play"):
                    self.toast(self.tutorial.error, 4000)
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
            elif e.key == pygame.K_SPACE and not self._input_locked():
                    s.toggle_cube()
            elif e.key in KEY_ACTIONS and not self._input_locked():
                s.manual_action(KEY_ACTIONS[e.key])
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.btn_pause.hit(e.pos):
                self.paused = True
            elif self.btn_restart.hit(e.pos):
                self.restart()
            elif self.btn_stage_prev.enabled and self.btn_stage_prev.hit(e.pos):
                self.start_stage(s.index - 1)
            elif self.btn_stage_next.enabled and self.btn_stage_next.hit(e.pos):
                self.start_stage(s.index + 1)
            elif self.btn_solver.hit(e.pos):
                self._set_panel(not self.panel_on)
            elif self._replay_click(e.pos):
                pass
            elif self.panel_on:
                for name, b in self.solve_buttons.items():
                    if b.hit(e.pos):
                        if s.has_solution:
                            s.reset()
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
            if s.status != "playing":
                self._on_game_over()

    # ------------------------------------------------------------------ draw
    def draw(self) -> None:
        """Chọn cảnh để vẽ"""
        surf = self.screen
        self.background.draw(surf)
        w, h = surf.get_size()
        if self.scene == "menu":
            s = self._menu_scale()
            cx, cy = w // 2, h // 2
            draw_divider(surf, cx, cy - int(320 * s), int(720 * s) // 20 * 20)
            draw_title(surf, "Bloxorz", max(16, int(140 * s)), center=(cx, cy - int(220 * s)))
            draw_title(surf, "Solver", max(10, int(85 * s)), color=COLORS["accent"],
                       center=(cx, cy - int(110 * s)))
            draw_divider(surf, cx, cy - int(40 * s), int(620 * s) // 20 * 20, flip=True)
            self.main_menu.draw(surf, cx, cy + int(10 * s), s)
        elif self.scene == "tutorial":
            self.tutorial.draw(surf)
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
                sub = f"Moves: {self.session.moves}  -  All stages cleared!"
                self._draw_overlay(surf, "You Win!", COLORS["win"], sub, self.end_menu)
            self._draw_metrics(surf)
        if self.toast_ms > 0:
            r = draw_text(surf, self.toast_text, 20, COLORS["text"], midbottom=(w // 2, h - 24))
            pygame.draw.rect(surf, COLORS["panel"], r.inflate(28, 16), border_radius=8)
            draw_text(surf, self.toast_text, 20, COLORS["text"], center=r.center)

    def _draw_hud(self, surf: pygame.Surface) -> None:
        """HUD trong game"""
        w = surf.get_width()
        self.btn_pause.rect = pygame.Rect(12, 10, 36, 36)
        x = w - 12 - (self._panel_w() + 2 * PANEL_MARGIN if self.panel_on else 0)
        for b in reversed(self.hud_buttons):
            x -= 92
            b.rect = pygame.Rect(x, 10, 92, 36)
            x -= 8
        for b in (self.btn_pause, *self.hud_buttons):
            b.draw(surf)
        s = self.session
        
        self.btn_stage_prev.enabled = s.index > 0
        self.btn_stage_next.enabled = s.index + 1 < len(self.stages)
        r = draw_text(surf, f"Stage {s.index + 1}", 22, COLORS["text"], bold=True, midleft=(66, HUD_H // 2))
        draw_text(surf, f"Moves: {s.moves}", 20, COLORS["muted"], midleft=(r.right + 24, HUD_H // 2))

    def _draw_overlay(self, surf, title, color, subtitle, menu: Menu) -> None:
        w, h = surf.get_size()
        cx = self.board_area().centerx
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((10, 12, 18, 175))
        surf.blit(dim, (0, 0))
        glow = tuple((c + 130) // 2 for c in color)
        s = self._menu_scale()
        draw_divider(surf, cx, h // 2 - int(190 * s), int(340 * s) // 20 * 20)
        draw_title(surf, title, max(16, int(56 * s)), color=color, glow_color=glow,
                   center=(cx, h // 2 - int(130 * s)))
        if subtitle:
            draw_caps(surf, subtitle, max(10, int(22 * s)), COLORS["muted"],
                      center=(cx, h // 2 - int(78 * s)))
        menu.draw(surf, cx, h // 2 - int(20 * s), s)

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

    @staticmethod
    def _describe_state(st: Optional[State]) -> str:
        if st is None:
            return "-"
        cells = "+".join(f"({r},{c})" for r, c in st.occupied_cells)
        if st.split_positions is not None:
            return f"split {cells}"
        return f"{cells} standing" if len(st.occupied_cells) == 1 else f"{cells} lying"

    def _panel_title(self, surf, x, y, text, s) -> None:
        pygame.draw.rect(surf, COLORS["accent"], (x, y, max(2, int(3 * s)), int(20 * s)))
        draw_text(surf, text, max(10, int(22 * s)), COLORS["text"], bold=True,
                midleft=(x + int(10 * s), y + int(9 * s)))

    def _panel_row(self, surf, x, y, pw, s, label, value) -> None:
        row = pygame.Rect(x, y, pw, int(30 * s))
        pygame.draw.rect(surf, COLORS["panel"], row, border_radius=int(6 * s))
        draw_text(surf, label, max(9, int(18 * s)), COLORS["accent"],
                midleft=(row.x + int(10 * s), row.centery))
        draw_text(surf, value, max(10, int(19 * s)), COLORS["text"], bold=True,
                midright=(row.right - int(10 * s), row.centery))

    def _draw_metrics(self, surf: pygame.Surface) -> None:
        """Cột SOLVER bên phải: nút solver, thông tin màn, số liệu, điều khiển replay / sau khi giải."""
        if not self.panel_on:
            return
        s = self._panel_scale()
        w, h = surf.get_size()
        pw = self._panel_w()
        col = pygame.Rect(w - pw - 2 * PANEL_MARGIN, 0, pw + 2 * PANEL_MARGIN, h)

        bg = pygame.Surface(col.size, pygame.SRCALPHA)
        bg.fill((*COLORS["bar"], 235))
        surf.blit(bg, col)
        pygame.draw.line(surf, COLORS["border"], col.topleft, col.bottomleft, 1)

        x, y = col.x + PANEL_MARGIN, PANEL_MARGIN + int(30 * s)
        row_step = int(38 * s)                      # row height + gap
        bh, g = max(28, int(38 * s)), int(8 * s)
        half = (pw - g) // 2
        sess = self.session
        board = sess.board

        draw_divider(surf, x + pw // 2, y, pw)
        draw_divider(surf, x + pw // 2, h - y, pw, flip=True)
        y += int(35 * s)

        # --- solver buttons 2x2
        self._panel_title(surf, x, y, "SOLVER", s)
        y += int(40 * s)
        for i, b in enumerate(self.solve_buttons.values()):
            b.rect = pygame.Rect(x + (i % 2) * (half + g), y + (i // 2) * (bh + g), half, bh)
            b.draw(surf)
        y += 2 * bh + g + int(24 * s)

        # --- stage info (always)
        self._panel_title(surf, x, y, "STAGE INFO", s)
        y += int(38 * s)
        for label, value in (("Map size", f"{board.height} x {board.width}"),
                            ("Goal", f"({board.goal[0]}, {board.goal[1]})")):
            self._panel_row(surf, x, y, pw, s, label, value)
            y += row_step
        y += int(16 * s)

        # --- metrics
        if self.last_stats:
            name, r = self.last_stats
            values = [f"{r.time * 1000:.2f} ms", f"{r.expanded_nodes:,}", f"{r.peak_memory / 1024:.1f} KB",
                    f"{r.solution_length} moves" if r.is_solved else "no solution"]
        elif self.solving:
            name, values = self.solving, ["..."] * 4
        else:
            draw_text(surf, "Choose a solver to run", max(9, int(18 * s)), COLORS["muted"],
                    midleft=(x, y + int(8 * s)))
            return

        self._panel_title(surf, x, y, f"SEARCH METRICS - {name}", s)
        y += int(38 * s)
        labels = ("Search time", "Nodes expanded", "Memory (peak)", "Solution length")
        rows = [("Start state", self._describe_state(self.solve_start))] + list(zip(labels, values))
        for label, value in rows:
            self._panel_row(surf, x, y, pw, s, label, value)
            y += row_step
        y += int(8 * s)

        # --- after the metrics: replay controls, or Solve again / Next stage
        if sess.has_solution:
            n = len(sess.sol_actions)
            draw_text(surf, f"Step {sess.sol_idx} / {n}", max(10, int(22 * s)), COLORS["accent"],
                    bold=True, midleft=(x, y + int(9 * s)))
            y += int(26 * s)
            bar_h = max(4, int(6 * s))
            pygame.draw.rect(surf, COLORS["accent_dark"], (x, y, pw, bar_h), border_radius=bar_h // 2)
            pygame.draw.rect(surf, COLORS["accent"], (x, y, int(pw * sess.sol_idx / n), bar_h),
                            border_radius=bar_h // 2)
            y += bar_h + int(16 * s)

            paused = sess.sol_paused or sess.finished
            if sess.finished:
                self.btn_replay.label = "Replay"
            else:
                self.btn_replay.label = "Resume" if paused else "Pause"
            self.btn_replay.enabled = True
            self.btn_prev.enabled = paused and sess.sol_idx > 0
            self.btn_next.enabled = paused and sess.sol_idx < n
            self.btn_replay.rect = pygame.Rect(x, y, pw, bh)
            self.btn_prev.rect = pygame.Rect(x, y + bh + g, half, bh)
            self.btn_next.rect = pygame.Rect(x + half + g, y + bh + g, half, bh)
            for b in self.replay_buttons:
                b.draw(surf)
        elif self.solve_done:
            self.btn_replay.label = "Replay"
            self.btn_replay.enabled = False
            self.btn_prev.enabled = False
            self.btn_next.enabled = False
            self.btn_replay.rect = pygame.Rect(x, y, pw, bh)
            self.btn_prev.rect = pygame.Rect(x, y + bh + g, half, bh)
            self.btn_next.rect = pygame.Rect(x + half + g, y + bh + g, half, bh)
            for b in self.replay_buttons:
                b.draw(surf)

    # --- chọn màn / Load Stage
    def _select_rects(self) -> List[pygame.Rect]:
        """Tạo bộ khung chữ nhật cho các slot trong menu Load Stage (mỗi slot gồm thumbnail + nhãn)."""
        w, h = self.screen.get_size()
        s = self._panel_scale()
        mx, top, bottom, gap = int(90 * s), int(150 * s), h - int(110 * s), int(20 * s)
        cw = (w - 2 * mx - (PAGE_COLS - 1) * gap) / PAGE_COLS
        ch = (bottom - top - (PAGE_ROWS - 1) * gap) / PAGE_ROWS
        return [pygame.Rect(mx + (i % PAGE_COLS) * (cw + gap), top + (i // PAGE_COLS) * (ch + gap), cw, ch)
                for i in range(PAGE_SIZE)]

    def _arrow_rects(self) -> Tuple[pygame.Rect, pygame.Rect]:
        """Tạo bộ mũi tên trong menu Load Stage"""
        w, h = self.screen.get_size()
        s = self._panel_scale()
        mid = (int(110 * s) + h - int(70 * s)) // 2
        aw, ah, m = int(54 * s), int(80 * s), int(18 * s)
        return pygame.Rect(m, mid - ah // 2, aw, ah), pygame.Rect(w - m - aw, mid - ah // 2, aw, ah)

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
        """Vẽ menu Load Stage (mọi kích thước chữ / khoảng cách đều scale theo cửa sổ)"""
        w, h = surf.get_size()
        s = self._panel_scale()
        n = len(self.stages)
        page, pages = self.select_index // PAGE_SIZE, max(1, math.ceil(n / PAGE_SIZE))
        draw_text(surf, "Select Stage", max(18, int(40 * s)), COLORS["text"], bold=True,
                  center=(w // 2, int(80 * s)))
        draw_divider(surf, w // 2, int(120 * s), int(340 * s) // 20 * 20, flip=True)

        rects = self._select_rects()
        label_size = max(12, int(28 * s))
        label_h = int(label_size * 1.5)          # vùng dành riêng cho nhãn "Stage N" dưới mỗi card
        pad = 6                                  # viền quanh thumbnail (3 px mỗi phía)
        min_w = int(110 * s)

        def thumb_of(idx: int, r: pygame.Rect):
            return self._thumb(idx, (max(1, int(r.w) - pad * 2), max(1, int(r.h) - label_h - pad * 2)))

        def card_rect(r: pygame.Rect, tw: int, th: int) -> pygame.Rect:
            """Card + nhãn được canh giữa theo chiều dọc trong slot."""
            card = pygame.Rect(0, 0, max(tw + pad, min_w), th + pad)
            card.midtop = (r.centerx, r.y + max(0, (int(r.h) - card.h - label_h) // 2))
            return card

        # Kích thước card chuẩn: lấy từ thumbnail đầu tiên có trên trang này
        ref = None
        for i in range(page * PAGE_SIZE, min(n, (page + 1) * PAGE_SIZE)):
            t = thumb_of(i, rects[0])
            if t:
                ref = (t.get_width(), t.get_height())
                break

        for slot, r in enumerate(rects):
            idx = page * PAGE_SIZE + slot
            if idx >= n:  # ô trống: cùng kích thước với card thật
                empty = card_rect(r, *ref) if ref else r.copy()
                pygame.draw.rect(surf, (34, 37, 48), empty, 1, border_radius=10)
                continue
            sel = idx == self.select_index
            col = COLORS["accent"] if sel else COLORS["muted"]
            thumb = thumb_of(idx, r)
            if thumb:
                card = card_rect(r, thumb.get_width(), thumb.get_height())
                surf.blit(thumb, thumb.get_rect(center=card.center))
                draw_text(surf, f"Stage {idx + 1}", label_size, col, bold=sel,
                          midtop=(card.centerx, card.bottom + int(4 * s)))
            else:
                card = r
                draw_text(surf, f"Stage {idx + 1}", max(12, int(26 * s)), col, bold=sel, center=card.center)
            pygame.draw.rect(surf, COLORS["accent"] if sel else COLORS["border"], card, 2, border_radius=10)

        # mũi tên chuyển page, text hướng dẫn manuver trong menu
        a, b, lw = int(12 * s), int(26 * s), max(2, int(6 * s))
        for rect, glyph, ok in zip(self._arrow_rects(), ("left", "right"), (page > 0, page < pages - 1)):
            col = COLORS["accent"] if ok else (50, 54, 68)
            cx, cy, d = rect.centerx, rect.centery, 1 if glyph == "right" else -1
            pygame.draw.lines(surf, col, False, [(cx - a * d, cy - b), (cx + a * d, cy), (cx - a * d, cy + b)], lw)
        draw_text(surf, f"Page {page + 1}/{pages}", max(12, int(25 * s)), COLORS["muted"],
                  midbottom=(w // 2, h - int(35 * s)))
        draw_text(surf, "Arrow keys: move   Enter: play   Esc: back", max(11, int(20 * s)), COLORS["muted"],
                  midbottom=(w // 2, h - int(8 * s)))
        
    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(60)
            self.screen = pygame.display.get_surface()
            for e in pygame.event.get():
                self.handle_event(e)
            self.update(dt)
            pygame.display.set_caption(f"Bloxorz Solver  {self.clock.get_fps():.0f} fps")
            self.draw()
            pygame.display.flip()
        pygame.quit()