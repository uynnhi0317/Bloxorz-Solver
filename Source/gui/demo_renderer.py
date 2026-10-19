"""Preview cho T06.

Chạy khi đứng từ thư mục Source/:   python -m gui.demo_renderer stages/simple/some_stage.json
Điều khiển bằng phím mũi tên, bắt đầu lại màn - phím R.

NOTE: render dựa trên output của game.rules.get_next_state, do đó các ô màu cam, công tắc,
cầu đóng hay box tách đôi chưa thể thể hiện qua preview này.

TODO: xóa file này
"""
import sys

import pygame

from game import Direction, get_next_state, is_goal, load_stage
from gui.animations import Sequence, animate_fall, animate_move, animate_win
from gui.renderer import Renderer

KEYS = {pygame.K_UP: Direction.UP, pygame.K_DOWN: Direction.DOWN,
        pygame.K_LEFT: Direction.LEFT, pygame.K_RIGHT: Direction.RIGHT}


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
                d = KEYS.get(e.key)
                if d is None:
                    continue
                nxt = get_next_state(board, state, d)
                if nxt is None:
                    anim, pending = animate_fall(renderer, state, d), board.initial_state
                elif is_goal(board, nxt):
                    anim = Sequence(animate_move(renderer, state, d, nxt), animate_win(renderer, nxt))
                    pending = board.initial_state
                else:
                    anim, pending = animate_move(renderer, state, d, nxt), nxt

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