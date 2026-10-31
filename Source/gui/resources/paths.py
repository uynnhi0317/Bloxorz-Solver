"""paths.py - Đường dẫn tài nguyên, tính từ vị trí file này nên không phụ thuộc CWD."""
from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional, Sequence

ASSET_ROOT = Path(__file__).resolve().parent.parent / "assets"


def _is_absolute(ref: str) -> bool:
    return PureWindowsPath(ref).is_absolute() or PurePosixPath(ref).is_absolute() or ref.startswith("\\")


def resolve_reference(ref: str, base_dir: Path, root: Path, search: Sequence[str] = ()) -> Optional[Path]:
    """Tìm file được tham chiếu bên trong OBJ/MTL (mtllib, map_Kd...). Trả về None nếu không thấy."""
    ref = ref.strip().strip('"').replace("\\", "/")
    if not ref:
        return None
    root = root.resolve()
    name = PurePosixPath(ref).name
    candidates = []
    if not _is_absolute(ref):
        candidates.append(base_dir / ref)
    candidates.append(base_dir / name)
    candidates += [root / folder / name for folder in search]
    for path in candidates:
        path = path.resolve()
        if path.is_file() and path.is_relative_to(root):
            return path
    return None

