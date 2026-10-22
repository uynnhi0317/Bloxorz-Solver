"""Preview cho T06.

Chạy khi đứng từ thư mục Source/:   python -m gui.demo_renderer stages/simple/some_stage.json
Điều khiển: phím mũi tên để lăn, SPACE để đổi cube khi đang tách, R để chơi lại màn.

NOTE: render dựa trên output của game.rules.get_next_state.

TODO: xóa file này
"""
import sys

import pygame

from game import Action, get_next_state, is_goal, load_stage
from gui.animations import Sequence, animate_fall, animate_move, animate_win, is_bump
from gui.renderer import Renderer

KEYS = {pygame.K_UP: Action.MOVE_UP, pygame.K_DOWN: Action.MOVE_DOWN,
        pygame.K_LEFT: Action.MOVE_LEFT, pygame.K_RIGHT: Action.MOVE_RIGHT,
        pygame.K_SPACE: Action.SWITCH_CONTROL}


def main(path: str) -> None:
    pygame.init()
    screen = pygame.display.set_mode((960, 640))
    clock = pygame.time.Clock()
    board = load_stage(path)
    renderer = Renderer(board, screen.get_rect())
    state = board.initial_state
    anim, pending = None, None

    while True:
        dt = clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
            if e.type == pygame.KEYDOWN and anim is None:
                if e.key == pygame.K_r:
                    state = board.initial_state
                action = KEYS.get(e.key)
                if action is None:
                    continue
                nxt = get_next_state(board, state, action)
                if action == Action.SWITCH_CONTROL:  # chỉ đổi cube active, không cần animation
                    if nxt is not None:
                        state = nxt
                elif nxt is None:
                    if is_bump(state, action):  # cube đâm vào cube kia: bị chặn, không thua
                        continue
                    anim, pending = animate_fall(renderer, state, action), board.initial_state
                elif is_goal(board, nxt):
                    anim = Sequence(animate_move(renderer, state, action, nxt), animate_win(renderer, nxt))
                    pending = board.initial_state
                else:
                    anim, pending = animate_move(renderer, state, action, nxt), nxt

        screen.fill((24, 26, 34))
        if anim:
            anim.update(dt)
            anim.draw(screen)
            if anim.done:
                state, anim = pending, None
        else:
            renderer.draw(screen, state)
        pygame.display.flip()


if __name__ == "__main__":
    main(sys.argv[1])