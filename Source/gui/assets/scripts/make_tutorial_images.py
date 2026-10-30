"""make_tutorial_images.py - Sinh ảnh minh họa cho Tutorial bằng chính renderer của game.

Chạy từ thư mục Source/:
    python -m gui.assets.scripts.make_tutorial_images

Ảnh lưu ở gui/assets/tutorial/. Đổi hình ảnh trong theme thì chạy lại.
"""
import os
import sys
from typing import List, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # không cần mở cửa sổ

import pygame

from game import Board, Orientation, State
from gui.app import SOURCE_DIR
from gui.renderer import Renderer
from gui.theme import Background

IMAGE_SIZE = (640, 440)
F, V, G, L = {"type": "floor"}, {"type": "void"}, {"type": "goal"}, {"type": "fragile"}
B = {"type": "bridge", "id": "b"}
SOFT = {"type": "switch", "switch": "soft", "targets": ["b"]}
HEAVY = {"type": "switch", "switch": "heavy", "targets": ["b"]}
SPLIT = {"type": "switch", "switch": "split", "destinations": [[0, 4], [2, 4]]}


def _scene(name: str, tiles: List[list], start: Tuple[int, int], state=None, closed=(), selected=None) -> None:
    w, h = IMAGE_SIZE
    board = Board.from_dict({"name": name, "start": list(start), "initial_closed_bridges": list(closed), "tiles": tiles})
    surf = pygame.Surface(IMAGE_SIZE)
    bg = Background(IMAGE_SIZE)
    bg.update(4000)
    bg.draw(surf)
    # tile_width lớn: bàn chơi luôn co/giãn vừa khung
    renderer = Renderer(board, pygame.Rect(16, 16, w - 32, h - 32), tile_width=1000.0)
    renderer.selected = selected
    renderer.draw(surf, state or board.initial_state)
    pygame.image.save(surf, str(SOURCE_DIR / "gui" / "assets" / "tutorial" / name))
    print(name)


def main() -> int:
    pygame.init()
    _scene("01_basics.png",
           [[F, F, F, V, V], [F, F, F, F, F], [F, F, F, G, F], [V, V, F, F, F]], (1, 1))
    _scene("02_bridge.png",
           [[F, F, V, V, F, F], [F, SOFT, B, B, F, G], [F, F, V, V, F, F]], (1, 0))
    # soft đang bật (cầu "a" mở, sáng lên), heavy chưa bật (cầu "b" đóng, tối)
    _scene("03_switch_types.png",
           [[F, F, F, F, F], [F, dict(SOFT, targets=["a"]), F, HEAVY, F], [F, {"type": "bridge", "id": "a"}, F, B, G]],
           (2, 2), closed=("b",))
    _scene("04_fragile.png",
           [[F, F, L, L, F], [F, F, L, L, G], [F, F, L, L, F]], (1, 0),
           state=State(1, 2, Orientation.HORIZONTAL))
    _scene("05_split_switch.png",
           [[F, F, F, V, F, F], [F, SPLIT, F, V, F, F], [F, F, F, V, F, G]], (1, 0),
           state=State.split((0, 4), (2, 4)), selected=(0, 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
