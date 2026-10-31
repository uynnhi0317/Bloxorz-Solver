"""make_assets.py - Sinh mô hình 3D Wavefront OBJ/MTL và texture cho Robot.

Chạy từ thư mục Source/:
    python -m gui.assets.scripts.make_assets
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from gui.resources import ASSET_ROOT, Material, save_obj
from gui.robot import ROBOT_MATERIALS, build_model

TEXTURE_SIZE = 256


def _noise(size: int, octaves, seed: int) -> np.ndarray:
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
    print(f"Created texture: {path.relative_to(ASSET_ROOT)}")


def make_textures() -> None:
    n = TEXTURE_SIZE
    # Vỏ kim loại titan/hợp kim: vân kim loại chải mờ
    grain = _noise(n, ((4, 1.0, 8), (16, 0.5, 14), (32, 0.25, 20)), 42)
    metal = 175 + 55 * grain[..., None] * np.array((0.95, 0.98, 1.05))
    _save(metal, "metal.png")


def write_mtl(name: str, materials: Dict[str, Material]) -> None:
    lines = [f"# {name} - sinh bởi make_assets.py"]
    for m in materials.values():
        lines += [
            f"newmtl {m.name}",
            "Kd " + " ".join(f"{c / 255:.4f}" for c in m.diffuse),
            f"d {m.opacity:.3f}",
            f"illum {m.illum}",
        ]
        if m.texture_ref:
            lines.append(f"map_Kd ../textures/{m.texture_ref}")
        lines.append("")
    path = ASSET_ROOT / "models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Created material lib: {path.relative_to(ASSET_ROOT)}")


def make_models() -> None:
    figures = {f"robot_{k}": build_model(k) for k in ("whole", "head", "body")}
    write_mtl("robot.mtl", ROBOT_MATERIALS)
    for name, loaded in figures.items():
        obj_path = ASSET_ROOT / "models" / f"{name}.obj"
        save_obj(loaded.model, obj_path, "robot.mtl")
        print(f"Created 3D model: {obj_path.relative_to(ASSET_ROOT)}")


def main() -> int:
    pygame.init()
    make_textures()
    make_models()
    return 0


if __name__ == "__main__":
    sys.exit(main())

