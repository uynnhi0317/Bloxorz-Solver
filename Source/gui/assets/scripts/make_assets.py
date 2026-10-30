"""make_assets.py - Sinh model OBJ/MTL và texture của game. Toàn bộ do code tạo ra, không dùng ảnh ngoài.

Chạy từ thư mục Source/:
    python -m gui.assets.scripts.make_assets

Kết quả: gui/assets/models/*.obj, *.mtl và gui/assets/textures/*.png.
Có thể thay các file này bằng model/ảnh tự làm: giữ nguyên tên file và tên vật liệu là game tự dùng.
"""
import os
import sys
from pathlib import Path
from typing import Dict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from gui.noface import build_model
from gui.prop_models import build_torii
from gui.resources import ASSET_ROOT, Material, save_obj

TEXTURE_SIZE = 256


def _noise(size: int, octaves, seed: int) -> np.ndarray:
    """Nhiễu lặp được (tile) từ tổng các sóng sin có tần số nguyên, chuẩn hóa về 0..1."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(size) / size, np.arange(size) / size, indexing="ij")
    out = np.zeros((size, size))
    for freq, amp, count in octaves:
        for _ in range(count):
            fx, fy = rng.integers(-freq, freq + 1, 2)
            out += amp * np.sin(2 * np.pi * (fx * x + fy * y) + rng.uniform(0, 6.28))
    return (out - out.min()) / (out.max() - out.min())


def _save(pixels: np.ndarray, name: str) -> None:
    path = ASSET_ROOT / "textures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(pygame.surfarray.make_surface(pixels.clip(0, 255).astype(np.uint8)), str(path))
    print(path.relative_to(ASSET_ROOT))


def make_textures() -> None:
    n = TEXTURE_SIZE
    # đá lát: nền xám ấm, đốm sáng tối mềm
    grain = _noise(n, ((3, 1.0, 6), (9, 0.6, 10), (28, 0.35, 16)), 1)
    stone = 118 + 90 * grain[..., None] * np.array((1.0, 0.98, 0.9))
    _save(stone, "stone.png")
    # gỗ: vân chạy theo một chiều, dùng để nhân với màu đỏ cầu/torii
    waves = _noise(n, ((2, 1.0, 5), (6, 0.5, 8)), 2)
    x = np.arange(n)[:, None] / n
    rings = 0.5 + 0.5 * np.sin(2 * np.pi * (x * 14 + waves * 3))
    wood = 150 + 105 * (0.35 * waves + 0.65 * rings)[..., None] * np.array((1.0, 0.96, 0.9))
    _save(wood, "torii_wood.png")
    # áo Vô Diện: các nếp vải dọc, tương phản thấp
    folds = _noise(n, ((6, 1.0, 8),), 3)
    _save(np.repeat((170 + 85 * folds)[..., None], 3, axis=2), "cloak.png")
    # gợn sóng mặt hồ: chỉ dùng độ sáng, vẽ cộng lên mặt nước
    ripples = _noise(n, ((10, 1.0, 14), (26, 0.6, 20)), 4)
    crest = np.clip((ripples - 0.64) / 0.36, 0, 1) ** 2
    _save(255 * crest[..., None] * np.array((0.85, 0.95, 1.0)), "water.png")


def write_mtl(name: str, materials: Dict[str, Material]) -> None:
    lines = [f"# {name} - sinh bởi make_assets.py"]
    for m in materials.values():
        lines += [f"newmtl {m.name}", "Kd " + " ".join(f"{c / 255:.4f}" for c in m.diffuse),
                  f"d {m.opacity:.3f}", f"illum {m.illum}"]
        if m.texture_ref:
            lines.append(f"map_Kd ../textures/{m.texture_ref}")
        lines.append("")
    path = ASSET_ROOT / "models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path.relative_to(ASSET_ROOT))


def make_models() -> None:
    figures = {f"noface_{k}": build_model(k) for k in ("whole", "head", "body")}
    props = {"torii": build_torii()}
    write_mtl("noface.mtl", next(iter(figures.values())).materials)
    write_mtl("props.mtl", next(iter(props.values())).materials)
    for name, loaded in figures.items():
        save_obj(loaded.model, ASSET_ROOT / "models" / f"{name}.obj", "noface.mtl")
    for name, loaded in props.items():
        save_obj(loaded.model, ASSET_ROOT / "models" / f"{name}.obj", "props.mtl")


def main() -> int:
    pygame.init()
    make_textures()
    make_models()
    return 0


if __name__ == "__main__":
    sys.exit(main())
