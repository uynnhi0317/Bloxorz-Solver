"""make_thumbnails.py - Sinh ảnh thumbnail (trạng thái ban đầu của từng màn) cho menu Load Stage.

Chạy từ thư mục Source/:
    python -m gui.assets.scripts.make_thumbnails                 # đúng danh sách + thứ tự stage mà app đang dùng
    python -m gui.assets.scripts.make_thumbnails a.json b.json   # hoặc chỉ định file (thứ tự = số thứ tự màn)

Ảnh lưu ở gui/assets/thumbs/stage_01.png, stage_02.png, ...
Thêm/đổi/đổi thứ tự stage thì chạy lại.
"""
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # không cần mở cửa sổ

import pygame

from game import Board, load_stage
from gui.app import SOURCE_DIR, discover_stages
from gui.renderer import Renderer
from gui.theme import Background

THUMB_SIZE = (400, 300)  # app tự co ảnh vừa ô (giữ tỉ lệ), nên ảnh nên to hơn ô một chút
PAD = 28                 # lề quanh bàn chơi


def render_thumbnail(board: Board) -> pygame.Surface:
    """Nền theme + bàn chơi + khối ở vị trí xuất phát, co vừa khung (cả màn nhỏ lẫn màn lớn)."""
    w, h = THUMB_SIZE
    surf = pygame.Surface((w, h))
    Background((w, h)).draw(surf)
    # tile_width rất lớn => Renderer luôn co/giãn bàn chơi cho vừa khung thay vì giữ cỡ tối đa 64px
    renderer = Renderer(board, pygame.Rect(PAD, PAD, w - 2 * PAD, h - 2 * PAD), tile_width=1000.0)
    renderer.draw(surf, board.initial_state)
    return surf


def main(argv: List[str]) -> int:
    paths = [Path(a) for a in argv] or discover_stages()
    if not paths:
        print("Không có stage nào để sinh thumbnail (stages/*/*.json trống).")
        return 1
    out_dir = SOURCE_DIR / "gui" / "assets" / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):  # giữ nguyên chỉ số dù có màn bị lỗi -> số thứ tự không lệch
        try:
            board = load_stage(path)
        except Exception as exc:
            print(f"[bỏ qua] {path}: {exc}")
            continue
        out = out_dir / f"stage_{i + 1:02d}.png"
        pygame.image.save(render_thumbnail(board), str(out))
        print(f"{out.relative_to(SOURCE_DIR)}  <-  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))