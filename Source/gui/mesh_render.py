"""mesh_render.py - Biên dịch model đã tải thành mảng numpy và sinh lệnh vẽ theo thuật toán họa sĩ.

Renderer là phần mềm (pygame), nên mỗi mặt có một màu: màu Kd nhân với màu trung bình của texture trên mặt đó.
Lưới nằm trong tọa độ cục bộ, gốc ở tâm khối; `rot` và `origin` đặt nó vào thế giới như một vật rắn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .resources import LoadedModel, get_assets
from .resources.model_loader import material_names

Color = Tuple[int, int, int]
RIM = np.array((150, 130, 120), dtype=float)   # viền sáng để áo tối nổi trên nền
IDENTITY = np.eye(3)
DECAL_PART = "face"     # bộ phận dán lên bề mặt (mặt nạ...): vẽ sau thân, theo thứ tự vật liệu trong file


@dataclass
class Mesh:
    verts: np.ndarray       # (N, 3)
    polys: List[np.ndarray]
    normals: np.ndarray     # (M, 3)
    centers: np.ndarray     # (M, 3)
    colors: np.ndarray      # (M, 3)
    tinted: np.ndarray      # (M,) mặt nhận màu do caller truyền vào (thân Vô Diện)
    lit: np.ndarray         # (M,) mặt chịu ánh sáng
    layer: np.ndarray       # (M,) lớp vẽ: thân trước, chi tiết dán lên sau
    part_center: np.ndarray # (M, 3) tâm của bộ phận chứa mặt


def compile_model(loaded: LoadedModel, tint_material: Optional[str] = None) -> Mesh:
    """tint_material: vật liệu sẽ nhận màu do caller truyền vào lúc vẽ."""
    model = loaded.model
    verts = model.vertices
    order = material_names(model)
    polys, normals, centers, colors, tinted, lit, layer, parts = [], [], [], [], [], [], [], []
    for face in model.faces:
        idx = np.array(face.vertices, dtype=int)
        pts = verts[idx]
        # pháp tuyến Newell, đúng cả với đa giác nhiều đỉnh hơi cong
        next_pts = np.roll(pts, -1, axis=0)
        n = np.array([np.sum((pts[:, 1] - next_pts[:, 1]) * (pts[:, 2] + next_pts[:, 2])),
                      np.sum((pts[:, 2] - next_pts[:, 2]) * (pts[:, 0] + next_pts[:, 0])),
                      np.sum((pts[:, 0] - next_pts[:, 0]) * (pts[:, 1] + next_pts[:, 1]))])
        length = np.linalg.norm(n)
        if length < 1e-9:
            continue
        mat = loaded.material(face.material)
        color = np.array(mat.diffuse, dtype=float)
        tex = loaded.textures.get(face.material)
        if tex is not None and face.uvs:
            sampled = np.array(tex.sample_mean(model.uvs[list(face.uvs)]))
            color = color * sampled / 255.0
        polys.append(idx)
        normals.append(n / length)
        centers.append(pts.mean(axis=0))
        colors.append(color)
        tinted.append(face.material == tint_material)
        lit.append(mat.lit)
        layer.append(order[face.material] if face.part == DECAL_PART else 0)
        parts.append(face.part)
    centers_arr = np.array(centers)
    part_means: Dict[str, np.ndarray] = {}
    for name in set(parts):
        part_means[name] = centers_arr[[i for i, p in enumerate(parts) if p == name]].mean(axis=0)
    return Mesh(verts, polys, np.array(normals), centers_arr, np.array(colors), np.array(tinted), np.array(lit),
                np.array(layer), np.array([part_means[p] for p in parts]))


_MESHES: Dict[str, Mesh] = {}


def get_mesh(name: str, fallback: Callable[[], LoadedModel], tint_material: Optional[str] = None) -> Mesh:
    """Mesh dùng chung trong cả phiên: file chỉ đọc một lần, đổi màn hay chạy solver không tải lại."""
    if name not in _MESHES:
        _MESHES[name] = compile_model(get_assets().load_model(name, fallback=fallback), tint_material)
    return _MESHES[name]


def mesh_ops(mesh: Mesh, rot: np.ndarray, origin: np.ndarray, view_dir: np.ndarray, light: np.ndarray,
             project: Callable[[np.ndarray], np.ndarray], clip: Optional[Callable] = None,
             below: Optional[bool] = None, tint: Optional[Color] = None, dim: float = 1.0, rim: bool = False) -> list:
    """Sinh lệnh vẽ (định dạng renderer._replay: (0, màu, đa giác)). below: chỉ lấy phần dưới/trên z = 0."""
    world = origin + mesh.verts @ rot.T
    normals = mesh.normals @ rot.T
    facing = normals @ view_dir
    vis = np.nonzero(facing > 1e-3)[0]
    if len(vis) == 0:
        return []
    base = mesh.colors.copy()
    if tint is not None:
        base[mesh.tinted] = base[mesh.tinted] * np.array(tint, dtype=float) / 255.0
    untinted_flat = ~mesh.tinted & ~mesh.lit
    base[untinted_flat] *= dim
    shade = np.where(mesh.lit, 0.62 + 0.75 * np.maximum(0.0, normals @ light), 1.0)
    colors = base * shade[:, None]
    if rim:
        colors[mesh.tinted] += RIM * (((1 - np.clip(facing[mesh.tinted], 0.0, 1.0)) ** 3) * 0.75)[:, None]
    colors = colors.clip(0, 255).astype(int)

    centers = origin + mesh.centers @ rot.T
    part_centers = origin + mesh.part_center @ rot.T
    # sắp: lớp -> bộ phận xa trước -> mặt xa trước
    order = vis[np.lexsort((centers[vis] @ view_dir, part_centers[vis] @ view_dir, mesh.layer[vis]))]
    screen = project(world)
    ops: list = []
    for i in order:
        idx = mesh.polys[i]
        col = tuple(int(v) for v in colors[i])
        if below is None:
            ops.append((0, col, screen[idx].tolist()))
            continue
        zs = world[idx, 2]
        lo, hi = zs.min(), zs.max()
        if (below and lo >= 0) or (not below and hi <= 0):
            continue
        if lo < 0 < hi and clip is not None:
            poly, _ = clip(world[idx], below)
            if len(poly) >= 3:
                ops.append((0, col, project(np.array(poly)).tolist()))
        else:
            ops.append((0, col, screen[idx].tolist()))
    return ops
