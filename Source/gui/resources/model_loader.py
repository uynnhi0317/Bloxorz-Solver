"""model_loader.py - Đọc/ghi model định dạng Wavefront OBJ, và ModelBuilder để dựng model bằng code."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("bloxorz.assets")


class AssetError(Exception):
    """Tài nguyên thiếu hoặc không hợp lệ."""


@dataclass(frozen=True)
class Face:
    vertices: Tuple[int, ...]     # chỉ số đỉnh (từ 0)
    uvs: Tuple[int, ...]          # chỉ số uv (từ 0), rỗng nếu không có
    material: str
    part: str


@dataclass
class Model:
    name: str
    vertices: np.ndarray                  # (N, 3)
    uvs: np.ndarray                       # (K, 2)
    faces: List[Face]
    material_lib: Optional[str] = None    # tham chiếu mtllib nguyên văn trong file
    path: Optional[Path] = None


def _resolve_index(token: str, count: int, what: str, line_no: int) -> int:
    i = int(token)
    i = i - 1 if i > 0 else count + i     # OBJ đánh số từ 1, số âm tính từ cuối
    if not 0 <= i < count or token == "0":
        raise AssetError(f"dòng {line_no}: chỉ số {what} {token} ngoài phạm vi (có {count})")
    return i


def parse_obj(path: Path) -> Model:
    """Đọc file OBJ: v, vt, f (v, v/vt, v//vn, v/vt/vn, chỉ số âm), o, g, usemtl, mtllib."""
    path = Path(path)
    if path.suffix.lower() != ".obj":
        raise AssetError(f"{path}: không phải file .obj")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AssetError(f"{path}: không đọc được ({exc})") from exc

    verts: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    faces: List[Face] = []
    material, part, lib = "default", "default", None
    try:
        for line_no, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tag, _, rest = line.partition(" ")
            args = rest.split()
            if tag == "v":
                verts.append((float(args[0]), float(args[1]), float(args[2])))
            elif tag == "vt":
                uvs.append((float(args[0]), float(args[1]) if len(args) > 1 else 0.0))
            elif tag == "f":
                if len(args) < 3:
                    raise AssetError(f"dòng {line_no}: mặt phải có ít nhất 3 đỉnh")
                vi, ti = [], []
                for token in args:
                    bits = token.split("/")
                    vi.append(_resolve_index(bits[0], len(verts), "đỉnh", line_no))
                    if len(bits) > 1 and bits[1]:
                        ti.append(_resolve_index(bits[1], len(uvs), "uv", line_no))
                faces.append(Face(tuple(vi), tuple(ti) if len(ti) == len(vi) else (), material, part))
            elif tag == "usemtl":
                material = rest.strip() or "default"
            elif tag in ("o", "g"):
                part = rest.strip() or "default"
            elif tag == "mtllib":
                lib = rest.strip() or None
    except (ValueError, IndexError) as exc:
        raise AssetError(f"{path}: dòng {line_no} sai định dạng ({exc})") from exc
    except AssetError as exc:
        raise AssetError(f"{path}: {exc}") from exc
    if not verts or not faces:
        raise AssetError(f"{path}: không có đỉnh hoặc mặt nào")
    return Model(path.stem, np.array(verts, dtype=float), np.array(uvs, dtype=float).reshape(-1, 2), faces, lib, path)


def save_obj(model: Model, path: Path, material_lib: Optional[str] = None) -> None:
    """Ghi model ra file OBJ (dùng cho script sinh asset)."""
    path = Path(path)
    out = [f"# {model.name} - sinh bởi gui/assets/scripts/make_assets.py"]
    if material_lib:
        out.append(f"mtllib {material_lib}")
    out += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in model.vertices]
    out += [f"vt {u:.5f} {v:.5f}" for u, v in model.uvs]
    part = material = None
    for face in model.faces:
        if face.part != part:
            part = face.part
            out.append(f"o {part}")
        if face.material != material:
            material = face.material
            out.append(f"usemtl {material}")
        if face.uvs:
            out.append("f " + " ".join(f"{v + 1}/{t + 1}" for v, t in zip(face.vertices, face.uvs)))
        else:
            out.append("f " + " ".join(str(v + 1) for v in face.vertices))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


class ModelBuilder:
    """Dựng Model bằng code: dùng làm phương án dự phòng khi thiếu file, và để xuất ra OBJ."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.verts: List[Tuple[float, float, float]] = []
        self.uvs: List[Tuple[float, float]] = []
        self.faces: List[Face] = []

    def add_vertices(self, pts: Iterable[Sequence[float]]) -> int:
        start = len(self.verts)
        self.verts.extend((float(p[0]), float(p[1]), float(p[2])) for p in pts)
        return start

    def add_face(self, indices: Sequence[int], material: str, part: str,
                 uvs: Optional[Sequence[Sequence[float]]] = None) -> None:
        uv_idx: Tuple[int, ...] = ()
        if uvs is not None:
            start = len(self.uvs)
            self.uvs.extend((float(u), float(v)) for u, v in uvs)
            uv_idx = tuple(range(start, start + len(uvs)))
        self.faces.append(Face(tuple(indices), uv_idx, material, part))

    def add_polygon(self, pts: Sequence[Sequence[float]], material: str, part: str) -> None:
        start = self.add_vertices(pts)
        self.add_face(range(start, start + len(pts)), material, part)

    def add_box(self, lo: Sequence[float], hi: Sequence[float], material: str, part: str) -> None:
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        s = self.add_vertices([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        square = ((0, 0), (1, 0), (1, 1), (0, 1))
        for quad in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            self.add_face([s + i for i in quad], material, part, square)

    def build(self) -> Model:
        return Model(self.name, np.array(self.verts, dtype=float), np.array(self.uvs, dtype=float).reshape(-1, 2),
                     list(self.faces))


def material_names(model: Model) -> Dict[str, int]:
    """Thứ tự xuất hiện của vật liệu trong model."""
    order: Dict[str, int] = {}
    for face in model.faces:
        order.setdefault(face.material, len(order))
    return order

